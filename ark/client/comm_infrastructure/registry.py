
import argparse
import socket
import threading
import struct
import json
import sys
import typer

from ark.client.comm_handler.service import Service, send_service_request
from ark.client.comm_infrastructure.endpoint import EndPoint
from ark.tools.log import log
from ark.global_constants import *
from arktypes import flag_t, network_info_t, node_info_t

app = typer.Typer()

class Registry(EndPoint):
    def __init__(self, registry_host: str = "127.0.0.1", registry_port: int = 1234, lcm_network_bounces: int = 1):
        """!
        Initialize the Registry server instance.

        @param registry_host: Host address for the registry.
        @param registry_port: Port on which the registry listens.
        @param lcm_network_bounces: TTL for LCM multicast messages.
        """
        global_config = {
            "network": {
                "registry_host": registry_host,
                "registry_port": registry_port,
                "lcm_network_bounces": lcm_network_bounces
            }
        }

        super().__init__(global_config)

        self.services = {}  # Maps service_name to (host, port)
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self.error_flag = False
        self.thread = None

        self.get_info_service = None  # Placeholder for service

    def _callback_get_network_info(self, channel, msg):
        """!
        Aggregate information about all nodes in the network.

        @param channel: Unused service channel name.
        @param msg: Service request message.
        @return: Populated :class:`network_info_t` message.
        """
        nodes_info = []
        req = flag_t()
        for service in self.services:
            if service.startswith(f"{DEFAULT_SERVICE_DECORATOR}/GetInfo"):
                node_info = send_service_request(self.registry_host, self.registry_port, service, req, node_info_t)
                if node_info is not None:
                    nodes_info.append(node_info)
        
        res = network_info_t()
        res.n_nodes = len(nodes_info)
        for node in nodes_info:
            res.nodes.append(node)
        return res

    def _serve(self):
        """!
        Main loop handling incoming registry requests.

        This method listens on the configured host and port, processing
        registration, deregistration and discovery requests from clients.
        """
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        #with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            self.s.bind((self.registry_host, self.registry_port))
            log.ok(f"Registry Server started on {self.registry_host} : {self.registry_port}")
        except OSError as e:
            log.error(f"Error: {e}")
            self.error_flag = True  # Set the error flag to true on error
            self._stop_event.set()  # Trigger shutdown
            self.s.close()  # Close the socket
            return  # Exit the method to stop the server

        self.s.listen()
        self.s.settimeout(1.0)  # Allow periodic check for stop event

        while not self._stop_event.is_set():
            try:
                conn, addr = self.s.accept()
            except socket.timeout:
                continue
            with conn:
                log.info(f"Registry: Connected via client (ip, port): {addr}")
                try:
                    # Receive message length
                    raw_msglen = self._recvall(conn, 4)
                    if not raw_msglen:
                        log.error("Registry: No message length received.")
                        continue
                    msglen = struct.unpack("!I", raw_msglen)[0]
                    # Receive the actual message
                    data = self._recvall(conn, msglen)
                    if not data:
                        log.error("Registry: No data received.")
                        continue
                    # Parse the request
                    request = json.loads(data.decode("utf-8"))
                    response = self._handle_request(request)
                    # Send response
                    encoded_resp = json.dumps(response).encode("utf-8")
                    conn.sendall(struct.pack("!I", len(encoded_resp)))
                    conn.sendall(encoded_resp)
                except Exception as e:
                    log.error(f"Registry: Error handling request: {e}")
                    continue  # Continue with the next request

    def _recvall(self, conn, n):
        """!
        Receive ``n`` bytes from a connection.

        @param conn: Socket connection.
        @param n: Number of bytes to receive.
        @return: The received bytes or ``None`` if EOF is hit.
        """
        data = bytearray()
        while len(data) < n:
            packet = conn.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)

    def _handle_request(self, request):
        """!
        Handle an incoming registry request.

        @param request: Parsed request dictionary.
        @return: Response dictionary to send back to the client.
        """
        req_type = request.get("type")
        if req_type == "REGISTER":
            service_name = request.get("service_name")
            host = request.get("host")
            port = request.get("port")
            if not all([service_name, host, port]):
                return {"status": "ERROR", "message": "Missing fields in REGISTER"}
            with self.lock:
                self.services[service_name] = (host, port)
            log.info(f"Registry: Registered service '{service_name}' at {host}:{port}")
            return {"status": "OK", "message": "Service registered successfully"}
        elif req_type == "DISCOVER":
            service_name = request.get("service_name")
            if not service_name:
                return {"status": "ERROR", "message": "Missing service_name in DISCOVER"}
            with self.lock:
                service = self.services.get(service_name)
            if service:
                host, port = service
                log.info(f"Registry: Service '{service_name}' found at {host}:{port}")
                return {"status": "OK", "host": host, "port": port}
            else:
                log.warning(f"Registry: Service '{service_name}' not found")
                return {"status": "ERROR", "message": "Service not found"}
        elif req_type == "DEREGISTER":
            service_name = request.get("service_name")
            if not service_name:
                return {"status": "ERROR", "message": "Missing service_name in DEREGISTER"}
            with self.lock:
                if service_name in self.services:
                    del self.services[service_name]  # Remove service from registry
                    log.info(f"Registry: Deregistered service '{service_name}'")
                    return {"status": "OK", "message": "Service deregistered successfully"}
                else:
                    log.warning(f"Registry: Service '{service_name}' not found")
                    return {"status": "ERROR", "message": "Service not found"}
        else:
            return {"status": "ERROR", "message": "Unknown request type"}

    def _stop(self):
        """!
        Stop the server and wait for the serving thread to finish.

        @return: ``None``
        """
        log.info("Shutting down server...")
        # Shutdown the info service
        if self.get_info_service:
            self.get_info_service.suspend()


        if self.thread and self.thread.is_alive():
            self._stop_event.set()
            self.thread.join()  # Ensure the server thread is stopped
            log.info("Server thread stopped.")
        self.s.close()

        log.info("Registry Server stopped.")

    def start(self):
        """!
        Start the server and monitor for errors.

        This method blocks until the server stops or encounters a fatal error.
        """
        
        try:
            # Initialize thread to serve requests
            self.thread = threading.Thread(target=self._serve, daemon=True)
            self.thread.start()

            self.get_info_service = Service(f"{DEFAULT_SERVICE_DECORATOR}/GetNetworkInfo", flag_t, network_info_t, self._callback_get_network_info, self.registry_host, self.registry_port, is_default=True)

            while not self.error_flag:
                if not self.thread.is_alive():
                    log.error("Server thread terminated unexpectedly.")
                    self.error_flag = True
                    self._stop()
                    sys.exit(1)
                self.thread.join(1)  # Wait for the thread to finish (or periodically check for errors)

            self._stop()
        except KeyboardInterrupt:
            log.error("Program interrupted by user.")
            self._stop()  # Gracefully stop the server
            sys.exit(0)  # Exit gracefully

@app.command()
def start(
    registry_host: str = typer.Option("127.0.0.1", "--host", help="The host address for the registry server."),
    registry_port: int = typer.Option(1234, "--port", help="The port for the registry server.")
):
    """!
    Start the Registry server with the specified host and port.

    @param registry_host: Host address for the registry server.
    @param registry_port: Port for the registry server.
    """
    server = Registry(registry_host=registry_host, registry_port=registry_port)
    server.start()

def main():
    """! Entry point for the CLI. """
    app()  # Initializes the Typer CLI

if __name__ == "__main__":
    main()
# def parse_args():
#     """Parse command-line arguments."""
#     parser = argparse.ArgumentParser(description="Registry Server")
#     parser.add_argument("--registry_host", type=str, default="127.0.0.1", help="The host address for the registry server.")
#     parser.add_argument("--registry_port", type=int, default=1234, help="The port for the registry server.")
#     return parser.parse_args()


# @app.command()
# def start_server(registry_host: str = "127.0.0.1", registry_port: int = 1234):
#     """Starts the Registry server with specified host and port."""
#     server = Registry(registry_host=registry_host, registry_port=registry_port)
#     server.start()

# def main():
#     """Entry point for the CLI."""
#     app()  # Initializes the Typer CLI

# if __name__ == "__main__":
#     main()
    # # Parse command-line arguments
    # args = parse_args()
    
    # # Create Registry server instance with command-line arguments
    # server = Registry(registry_host=args.registry_host, registry_port=args.registry_port)
    # server.start()
