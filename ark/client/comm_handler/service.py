
import socket
import struct
import threading
from typing import Callable, Type
from ark.client.comm_handler.comm_handler import CommHandler
import json
import time
from ark.tools.log import log
from typing import Any

class Service(CommHandler):
    def __init__(
        self,
        name: str,
        req_type: Type,
        resp_type: Type,
        callback: Callable[[str, object], object],
        registry_host: str,
        registry_port: int,
        host: str = None,
        port: int = None,
        is_default = False
    ):
        """
        Initialize the service.

        :param name: Name of the service.
        :param req_type: Request message class with encode/decode methods.
        :param resp_type: Response message class with encode/decode methods.
        :param callback: Function to handle the request and return a response.
        :param registry_host: Host of the registry server.
        :param registry_port: Port of the registry server.
        :param host: Host to bind the service. If None, binds to the local network interface.
        :param port: Port to bind the service. If None, a random free port is chosen.
        """
        self.service_name = name
        self.comm_type = "Service"
        self.req_type = req_type
        self.resp_type = resp_type
        self.callback = callback
        self.host = host if host is not None else self._get_local_ip()
        self.port = port if port is not None else self._find_free_port()
        self.registry_host = registry_host
        self.registry_port = registry_port
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._serve)
        self.is_default_service = is_default
        self.thread.daemon = True
        self.thread.start()
        self.registered = self.register_with_registry()

    def _get_local_ip(self) -> str:
        """Get the local IP address of the machine."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(('10.254.254.254', 1))  # Connect to a non-local address
            local_ip = s.getsockname()[0]
        except Exception:
            local_ip = "0.0.0.0"  # If it fails, use a fallback IP
        finally:
            s.close()
        return local_ip

    def _find_free_port(self) -> int:
        """Find a free port to bind the service."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, 0))
            return s.getsockname()[1]

    def register_with_registry(self):
        """Register the service with the registry server."""
        registration = {
            "type": "REGISTER",
            "service_name": self.service_name,
            "host": self.host,  # Use the local IP address for registration
            "port": self.port,
        }
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
               
                s.connect((self.registry_host, self.registry_port))
                encoded_req = json.dumps(registration).encode("utf-8")
                
                s.sendall(struct.pack("!I", len(encoded_req)))
           
                s.sendall(encoded_req)
                # Receive response
                raw_resp_len = self._recvall(s, 4)

                if not raw_resp_len:
                    log.error("Service: Failed to receive registration response length.")
                    return False
                resp_len = struct.unpack("!I", raw_resp_len)[0]
                data = self._recvall(s, resp_len)
                if not data:
                    log.error("Service: Failed to receive registration response data.")
                    return False
                response = json.loads(data.decode("utf-8"))
                if response.get("status") == "OK":
                    log.info(
                        f"Service: Successfully registered '{self.service_name}' with registry."
                    )
                else:
                    log.error(f"Service: Registration failed - {response.get('message')}")
                    return False
        except Exception as e:
            # log.error(f"Service: Error registering with registry - {e}")
            return 
        return True

    def _serve(self):
        """Serve incoming service requests."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            while not self._stop_event.is_set():
                try:
                    s.settimeout(1.0)
                    conn, addr = s.accept()
                
                except socket.timeout:
                    continue
                with conn:
            
                    try:
                        # Receive message length
                        raw_msglen = self._recvall(conn, 4)
                        if not raw_msglen:
                            print("Service: No message length received.")
                            continue
                        msglen = struct.unpack("!I", raw_msglen)[0]
                        
                        # Receive the actual message
                        data = self._recvall(conn, msglen)
                        if not data:
                            print("Service: No data received.")
                            continue
                        # Decode the request
                        request = self.req_type.decode(data)
                        
                        # Process the request
                        response = self.callback(self.service_name, request)
                        # Encode the response
                        encoded_resp = response.encode()
                        
                        # Send the length of the response first
                        conn.sendall(struct.pack("!I", len(encoded_resp)))
                        # Then send the actual response
                        conn.sendall(encoded_resp)
                    except Exception as e:
                        log.error(f"Service: Error handling request: {e}")

    def _recvall(self, conn, n):
        """Helper function to receive n bytes or return None if EOF is hit."""
        data = bytearray()
        while len(data) < n:
            packet = conn.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)

    def __repr__(self):
        """
        Returns a string representation of the communication handler, including 
        the channel name and the types of messages it handles.

        The string is formatted as:
        "channel_name[request_type, response_type]".

        @return: A string representation of the handler, formatted as 
                "channel_name[request_type,response_type]".
        """
        return f"{self.service_name}[{self.req_type},{self.resp_type}]"
    
    def restart(self):
        return super().restart()

    def suspend(self):
        """Shut down the service and deregister from the registry."""
        if self.deregister_from_registry():
            self._stop_event.set()  # Stop the serving thread
            self.thread.join()  # Wait for the serving thread to terminate
            print(f"Service '{self.service_name}' stopped.")
        else:
            print("Service shutdown un-gracefully.")

    def deregister_from_registry(self) -> bool:
        """Deregister the service from the registry server and validate the response."""
        deregistration = {
            "type": "DEREGISTER",
            "service_name": self.service_name,
            "host": self.host,
            "port": self.port,
        }
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.registry_host, self.registry_port))
                encoded_req = json.dumps(deregistration).encode("utf-8")
                s.sendall(struct.pack("!I", len(encoded_req)))
                s.sendall(encoded_req)
                log.info(f"Service: Sending deregistration request for '{self.service_name}'.")

                # Receive response length
                raw_resp_len = self._recvall(s, 4)
                if not raw_resp_len:
                    log.error("Service: Failed to receive deregistration response length.")
                    return False
                resp_len = struct.unpack("!I", raw_resp_len)[0]

                # Receive the actual response data
                data = self._recvall(s, resp_len)
                if not data:
                    log.error("Service: Failed to receive deregistration response data.")
                    return False

                # Parse the response
                response = json.loads(data.decode("utf-8"))
                if response.get("status") == "OK":
                    log.info(f"Service: Successfully deregistered '{self.service_name}' from registry.")
                    return True
                else:
                    log.error(f"Service: Deregistration failed - {response.get('message')}")
                    return False
        except Exception as e:
            log.error(f"Service: Error deregistering from registry - {e}")
            return False
        
    def get_info(self):
        info = {
            "comms_type": "Service",
            "service_name": self.service_name,
            "service_host": self.host,
            "service_port": self.port,
            "registry_host": self.registry_host,
            "registry_port": self.registry_port,
            "request_type": self.req_type.__name__, 
            "response_type": self.resp_type.__name__,
            "default_service": self.is_default_service,
        }

        return info


def send_service_request(registry_host, registry_port, service_name: str, request: object, response_type: type, timeout: int = 1) -> Any:
        """
        Sends a request to a service discovered from a registry.

        This method first discovers the target service by querying the registry,
        then sends a request to the discovered service, and returns the response.

        Args:
            registry_host (str): The host address of the service registry.
            registry_port (int): The port of the service registry.
            service_name (str): The name of the service to be discovered.
            request (object): The request object to be sent to the discovered service.

        Returns:
            Any: The response from the service.

        Raises:
            Exception: If there is an error during discovery or while calling the service.
        """
        # TODO timeout addition
        try:
            # Discover the host and port of the service from the registry
            host, port = __discover_service(registry_host, registry_port, service_name)
            # Call the discovered service with the provided request
            response = __call_service(host, port, request, response_type)
            return response
        except Exception as e:
            log.error(f"Client Error: {e}")
        pass

def __call_service(service_host: str, service_port: int, request, response_type: type) -> Any:
    """
    Calls a specific service with the given request and receives a response.

    This method establishes a socket connection with the service, sends the request,
    and waits for the response. It then decodes the response and returns it.

    Args:
        service_host (str): The host address of the service.
        service_port (int): The port of the service.
        request (object): The request to send to the service.
        response_type (type): The expected type of the response.

    Returns:
        Any: The response from the service, decoded into the specified response type.

    Raises:
        RuntimeError: If there is an error while sending the request or receiving the response.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Connect to the service
        s.connect((service_host, service_port))
        
        # Encode the request into bytes
        encoded_req = request.encode()

        # Send the length of the request first
        s.sendall(struct.pack("!I", len(encoded_req)))
        
        # Then send the actual request data
        s.sendall(encoded_req)

        # Receive the length of the response (first 4 bytes)
        raw_resp_len = __recvall(s, 4)
        if not raw_resp_len:
            raise RuntimeError("Client: Failed to receive response length.")
        resp_len = struct.unpack("!I", raw_resp_len)[0]
        
        # Receive the actual response data
        data = __recvall(s, resp_len)
        if not data:
            raise RuntimeError("Client: Failed to receive response data.")
        
        # Decode the response into the specified response type
        response = response_type.decode(data)
        return response

def __recvall(conn: socket.socket, n: int) -> bytes:
    """
    Receives `n` bytes of data from the socket connection.

    This helper function reads from the socket in chunks until `n` bytes
    are received. If EOF (End of File) is encountered before `n` bytes are
    received, it returns None.

    Args:
        conn (socket.socket): The socket connection to receive data from.
        n (int): The number of bytes to receive.

    Returns:
        bytes: The received data as a byte string.

    Raises:
        None: Returns None if the connection is closed before receiving `n` bytes.
    """
    data = bytearray()
    while len(data) < n:
        # Receive the remaining bytes
        packet = conn.recv(n - len(data))
        if not packet:
            return None  # EOF hit
        data.extend(packet)
    return bytes(data)

def __discover_service(registry_host: str, registry_port: int, service_name: str):
    """
    Discovers the host and port of a service by querying the service registry.

    This method sends a discovery request to the registry and receives the
    host and port details of the requested service.

    Args:
        registry_host (str): The host address of the registry.
        registry_port (int): The port of the registry.
        service_name (str): The name of the service to be discovered.

    Returns:
        tuple: A tuple containing the host and port of the discovered service.

    Raises:
        RuntimeError: If there is an error during the discovery process.
    """
    discovery_request = {"type": "DISCOVER", "service_name": service_name}
    try:
        # Create a socket connection to the registry
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((registry_host, registry_port))
            
            # Encode the discovery request to send it over the socket
            encoded_req = json.dumps(discovery_request).encode("utf-8")
            
            # Send the length of the request first
            s.sendall(struct.pack("!I", len(encoded_req)))
            
            # Send the actual discovery request
            s.sendall(encoded_req)
            
            # Receive the length of the response (first 4 bytes)
            raw_resp_len = __recvall(s, 4)
            if not raw_resp_len:
                raise RuntimeError(
                    "Client: Failed to receive discovery response length."
                )
            resp_len = struct.unpack("!I", raw_resp_len)[0]
            
            # Receive the actual response data
            data = __recvall(s, resp_len)
            if not data:
                raise RuntimeError("Client: Failed to receive discovery response data.")
            
            # Decode the response
            response = json.loads(data.decode("utf-8"))
            
            # If the service was successfully discovered, return the host and port
            if response.get("status") == "OK":
                host = response.get("host")
                port = response.get("port")
                return host, port
            else:
                raise RuntimeError(
                    f"Client: Service discovery failed - {response.get('message')}"
                )
    except Exception as e:
        log.error(f"Client: Error during service discovery - {e}")
        raise
