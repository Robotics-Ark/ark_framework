
import ast
import os
import time
import signal
import uuid
import sys
from abc import ABC, abstractmethod
from typing import Any, Generator, Dict, Union
from pathlib import Path
import yaml

import lcm
from lcm import LCM

from ark.client.comm_infrastructure.endpoint import EndPoint
from ark.client.comm_handler.comm_handler import CommHandler
from ark.client.comm_handler.multi_comm_handler import MultiCommHandler
from ark.client.comm_handler.publisher import Publisher
from ark.client.comm_handler.subscriber import Subscriber
from ark.client.comm_handler.listener import Listener
from ark.client.comm_handler.service import Service, send_service_request
from ark.client.comm_handler.multi_channel_publisher import MultiChannelPublisher
from ark.client.comm_handler.multi_channel_listener import MultiChannelListener
from ark.client.frequencies.stepper import Stepper
from ark.tools.log import log

from arktypes import flag_t, node_info_t, comms_info_t, service_info_t, listener_info_t, subscriber_info_t, publisher_info_t


DEFAULT_SERVICE_DECORATOR = "__DEFAULT_SERVICE"

class CommEndpoint(EndPoint):
    """!
    Base class for nodes that interact with the LCM system. Handles the subscription,
    publishing, and communication processes for the node.

    The `BaseNode` class manages the LCM instance and communication handlers, and provides
    methods for creating publishers, subscribers, listeners, and steppers. It also provides
    functionality for handling command-line arguments and the graceful shutdown of the node.

    @param lcm: The LCM instance used for communication.
    @param channel_name: The name of the channel to subscribe to.
    @param channel_type: The type of the message expected for the channel.
    """    
    #def __init__(self, node_name: str, registry_host: str = "127.0.0.1", registry_port: int = 1234, lcm_network_bounces: int = 1) -> None:
    def __init__(self,
                 node_name: str,
                 global_config: Union[str, Dict[str, Any], Path]) -> None:
        """!
        Initialize a communication endpoint for a node.

        @param node_name: The name for the node.
        param unique: True when the node is unique, False otherwise. When True the node name is appended with a unique stamp.

        @raises SystemExit: If "--help" or "-h" is passed in command-line arguments.
        """

        # system_config = self.load_system_config(global_config, node_name)



        super().__init__(global_config)

        self.name = node_name
        self.node_id = str(uuid.uuid4())

        self._done: bool = False
        self._comm_handlers: list[CommHandler] = []
        self._multi_comm_handlers: list[MultiCommHandler] = []
        self._steppers: list[Stepper] = []

        # Create default service for get info of the node
        get_info_service_name = f"{DEFAULT_SERVICE_DECORATOR}/GetInfo/{self.name}_{self.node_id}"
        get_info_service = self.create_service(get_info_service_name, flag_t, node_info_t, self._callback_get_info, True)
        
        suspend_node_service_name = f"{DEFAULT_SERVICE_DECORATOR}/SuspendNode/{self.name}_{self.node_id}"
        suspend_node_service = self.create_service(suspend_node_service_name, flag_t, flag_t, self._callback_suspend_node, True)

        restart_node_service_name = f"{DEFAULT_SERVICE_DECORATOR}/RestartNode/{self.name}_{self.node_id}"
        restart_node_service = self.create_service(restart_node_service_name, flag_t, flag_t, self._callback_restart_node, True)
        
        self.registered = self.check_registration()

        if self.registered == False: 
            log.error("Unable to connect to Registry. Please check network configuration setting / start a registry")
            sys.exit(1)

    def check_registration(self):
        """!
        Check whether all default services have been registered.

        @return: ``True`` if all services are registered, ``False`` otherwise.
        """
        # check if default services are registered
        n_service_channels = 0
        n_service_channels_registered = 0
        for ch in self._comm_handlers:
            # check if type is a service
            if ch.comm_type == "Service":
                n_service_channels += 1
                if ch.registered == True: 
                    n_service_channels_registered += 1
        
        if n_service_channels != n_service_channels_registered and n_service_channels_registered == 0:
            # log.error("ARK Registry has not been started, use 'ark registry start', to start ")
            return False
        elif n_service_channels != n_service_channels_registered and n_service_channels_registered > 0:
            log.error("FATAL: Some services are not registered, please check the registry and network settings")
            return False
        else:
            return True
        
    def _load_config_section(self, global_config: Union[str, Dict[str, Any], Path], name: str, type: str) -> Dict:
        """!
        Load the configuration section for a component.

        @param global_config: Global configuration source.
        @param name: Name of the component.
        @param type: Section type within the configuration file.
        @return: Dictionary containing the configuration for the component.
        """
        
        if isinstance(global_config, str):
            global_config = Path(global_config)
            if not global_config.exists():
                log.error("Given configuration file path does not exist.")
            if not global_config.is_absolute():
                global_config = global_config.resolve()
        if isinstance(global_config, Path):
            config_path = str(global_config)
            with open(config_path, 'r') as file:
                cfg = yaml.safe_load(file) 
            for item in cfg.get(type, []):
                if isinstance(item, dict):  # If it's an inline configuration
                    config = item["config"]
                    return config
                # Make sure the yaml config has the same name with "name"
                elif isinstance(item, str) and item.endswith('.yaml'):  # If it's a path to an external file
                    if item.split('.')[0] == type+'/'+name:
                        if os.path.isabs(item):  # Check if the path is absolute
                            external_path = item
                        else:  # Relative path, use the directory of the main config file
                            external_path = os.path.join(os.path.dirname(config_path), item)
                        # Load the YAML file and return its content
                        with open(external_path, 'r') as file:
                            item_config = yaml.safe_load(file)
                            config = item_config["config"]
                        return config
                else:
                    log.error(f"Invalid entry in '{type}': {self.name}. Please provide either a config or a path to another (.yaml) config.")
                    return  # Skip invalid entries
        if isinstance(global_config, dict):
            config = {}
            for component, component_config in global_config[type].items():
                if component == self.name:
                    if not component_config:
                        log.error(f"Please provide a config for the {type}: {self.name}")
                    return component_config
            if not config:
                log.error(f"Couldn't find type '{self.name}' in config.")
            return config
        else:
            log.error(f"Couldn't load config for {type} '{self.name}'")
    
    def get_info(self) -> dict:
        """!
        Gather information about all registered communication handlers.

        @return: Dictionary describing listeners, publishers, subscribers and services.
        """
        listener_info = []
        subscriber_info = []
        publisher_info = []
        service_info = []
        for ch in self._comm_handlers: 
            ch_info = ch.get_info()
            if ch_info["comms_type"] == "Listener":
                listener_info.append(ch_info)
            elif ch_info["comms_type"] == "Subscriber":
                subscriber_info.append(ch_info)
            elif ch_info["comms_type"] == "Publisher":
                publisher_info.append(ch_info)
            elif ch_info["comms_type"] == "Service":
                service_info.append(ch_info)
            else: 
                raise NameError
        
        for m_ch in self._multi_comm_handlers: 
            m_ch_info = m_ch.get_info()
            for ch_info in m_ch_info: 
                if ch_info["comms_type"] == "Listener":
                    listener_info.append(ch_info)
                elif ch_info["comms_type"] == "Subscriber":
                    subscriber_info.append(ch_info)
                elif ch_info["comms_type"] == "Publisher":
                    publisher_info.append(ch_info)
                elif ch_info["comms_type"] == "Service":
                    service_info.append(ch_info)
                else: 
                    raise NameError

        info = {
            "node_name": self.name,
            "node_id": self.node_id,
            "comms": {
                "listeners": listener_info,
                "subscribers": subscriber_info,
                "publishers": publisher_info,
                "services": service_info
            }
        }

        return info
    
    def _callback_get_info(self, channel, msg):
        """!
        Callback for the default GetInfo service.

        @param channel: Unused service channel name.
        @param msg: Service request message.
        @return: Node information message.
        """

        print("Get info service called")


        # Create an instance of node_info_t
        node_info = node_info_t()

        data = self.get_info()
        # Populate node_info
        node_info.node_name = data['node_name']
        node_info.node_id = data['node_id']

        # Create comms_info_t
        comms_info = comms_info_t()

        # Populate listeners
        comms_info.n_listeners = len(data['comms']['listeners'])
        comms_info.listeners = [listener_info_t() for _ in range(comms_info.n_listeners)]
        for i, listener in enumerate(data['comms']['listeners']):
            comms_info.listeners[i].comms_type = listener['comms_type']
            comms_info.listeners[i].channel_name = listener['channel_name']
            comms_info.listeners[i].channel_type = listener['channel_type']
            comms_info.listeners[i].channel_status = listener['channel_status']

        # Populate subscribers
        comms_info.n_subscribers = len(data['comms']['subscribers'])
        comms_info.subscribers = [subscriber_info_t() for _ in range(comms_info.n_subscribers)]
        for i, subscriber in enumerate(data['comms']['subscribers']):
            comms_info.subscribers[i].comms_type = subscriber['comms_type']
            comms_info.subscribers[i].channel_name = subscriber['channel_name']
            comms_info.subscribers[i].channel_type = subscriber['channel_type']
            comms_info.subscribers[i].channel_status = subscriber['channel_status']

        # Populate publishers
        comms_info.n_publishers = len(data['comms']['publishers'])
        comms_info.publishers = [publisher_info_t() for _ in range(comms_info.n_publishers)]
        for i, publisher in enumerate(data['comms']['publishers']):
            comms_info.publishers[i].comms_type = publisher['comms_type']
            comms_info.publishers[i].channel_name = publisher['channel_name']
            comms_info.publishers[i].channel_type = publisher['channel_type']
            comms_info.publishers[i].channel_status = publisher['channel_status']

        # Populate services
        comms_info.n_services = len(data['comms']['services'])
        comms_info.services = [service_info_t() for _ in range(comms_info.n_services)]
        for i, service in enumerate(data['comms']['services']):
            comms_info.services[i].comms_type = service['comms_type']
            comms_info.services[i].service_name = service['service_name']
            comms_info.services[i].service_host = service['service_host']
            comms_info.services[i].service_port = service['service_port']
            comms_info.services[i].registry_host = service['registry_host']
            comms_info.services[i].registry_port = service['registry_port']
            comms_info.services[i].request_type = service['request_type']
            comms_info.services[i].response_type = service['response_type']

        # Assign comms_info to node_info
        node_info.comms = comms_info

        return node_info

    def create_publisher(self, channel_name: str, channel_type: type) -> Publisher:
        """!
        Creates and returns a publisher for the specified channel.

        @param channel_name: The name of the channel to publish to.
        @type channel_name: str
        @param channel_type: The type of the message to publish.
        @type channel_type: type
        @return: The created Publisher instance.
        @rtype: Publisher
        """
        pub = Publisher(self._lcm, channel_name, channel_type)
        self._comm_handlers.append(pub)
        return pub
    
    def create_multi_channel_publisher(self, channels):
        """!
        Create a publisher that manages multiple channels.

        @param channels: List of ``(channel_name, channel_type)`` tuples.
        @return: The created :class:`MultiChannelPublisher` instance.
        """
        multi_pub = MultiChannelPublisher(channels, self._lcm)
        self._multi_comm_handlers.append(multi_pub)
        return multi_pub
    
    def create_multi_channel_listener(self, channels):
        """!
        Create listeners for multiple channels.

        @param channels: List of ``(channel_name, channel_type)`` tuples.
        @return: The created :class:`MultiChannelListener` instance.
        """
        multi_listeners = MultiChannelListener(channels, lcm_instance=self._lcm)
        self._multi_comm_handlers.append(multi_listeners)
        return multi_listeners

    def create_subscriber(
        self,
        channel_name: str,
        channel_type: type,
        callback: callable,
        callback_args: list = [],
    ) -> Subscriber:
        """!
        Creates and returns a subscriber for the specified channel.

        @param channel_name: The name of the channel to subscribe to.
        @type channel_name: str
        @param channel_type: The type of the message expected from the channel.
        @type channel_type: type
        @param callback: The callback function to be invoked when a message is received.
        @type callback: callable
        @param callback_args: Additional arguments to pass to the callback.
        @type callback_args: list
        @return: The created Subscriber instance.
        @rtype: Subscriber
        """
        sub = Subscriber(
            self._lcm,
            channel_name,
            channel_type,
            callback,
            callback_args=callback_args,
        )
        self._comm_handlers.append(sub)
        return sub
    
    def create_service(self, service_name: str, request_type: type, response_type: type, callback: callable, is_default_service = False):
        """!
        Create and register a service.

        @param service_name: Name of the service.
        @param request_type: Message type of the request.
        @param response_type: Message type of the response.
        @param callback: Callback invoked to handle the request.
        @param is_default_service: Mark service as an internal default.
        @return: The created :class:`Service` instance.
        """
        service = Service(service_name= service_name,
                          req_type= request_type,
                          resp_type= response_type,
                          callback= callback,
                          registry_host= self.registry_host,
                          registry_port= self.registry_port,
                          is_default= is_default_service)
        self._comm_handlers.append(service)
        return service

    def create_listener(self, channel_name: str, channel_type: type) -> Listener:
        """!
        Creates and returns a listener for the specified channel.

        @param channel_name: The name of the channel to listen to.
        @type channel_name: str
        @param channel_type: The type of the message expected from the channel.
        @type channel_type: type
        @return: The created Listener instance.
        @rtype: Listener
        """
        listener = Listener(self._lcm, channel_name, channel_type)
        self._comm_handlers.append(listener)
        return listener

    def wait_for_message(
        self, channel_name: str, channel_type: type, timeout: int = 10
    ) -> Any:
        """!
        Waits for a single message on the specified channel within a timeout period.

        @param channel_name: The name of the channel to listen for messages.
        @type channel_name: str
        @param channel_type: The type of the message to expect.
        @type channel_type: type
        @param timeout: The number of seconds to wait before timing out.
        @type timeout: int
        @return: The received message, or None if the timeout was reached.
        @rtype: Any
        @raises TimeoutError: If the timeout is reached before receiving a message.
        """

        def timeout_handler(signum, frame):
            raise TimeoutError(
                f"Timeout reached while waiting for a message on channel '{channel_name}'."
            )

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        msg = None
        listener = Listener(self._lcm, channel_name, channel_type)

        try:
            while not listener.received():
                self._lcm.handle()  # Blocking call.
            msg = listener.get()
        except TimeoutError:
            log.warning(
                f"Listener {listener} did not receive a message within the specified timeout."
            )
        finally:
            signal.alarm(0)  # Cancel the alarm.
            listener.shutdown()

        return msg

    def create_stepper(
        self,
        hz: float,
        callback: callable,
        oneshot: bool = False,
        reset: bool = True,
        callback_args: list = [],
    ) -> Stepper:
        """!
        Creates and returns a stepper that calls the specified callback at the specified rate.

        @param hz: The frequency (in Hz) at which the callback will be invoked.
        @type hz: float
        @param callback: The callback function to be called.
        @type callback: callable
        @param oneshot: If True, the callback is fired only once. Otherwise, it fires continuously.
        @type oneshot: bool
        @param reset: If True, the timer is reset when time moves backward.
        @type reset: bool
        @param callback_args: Additional arguments to pass to the callback.
        @type callback_args: list
        @return: The created Stepper instance.
        @rtype: Stepper
        """
        stepper = Stepper(
            hz, callback, oneshot=oneshot, reset=reset, callback_args=callback_args
        )
        self._steppers.append(stepper)
        return stepper

    def now(self) -> float:
        """!
        Returns the current time in seconds since the epoch.

        @return: The current time.
        @rtype: float
        """
        return time.time()

    def now_ns(self) -> int:
        """!
        Returns the current time in nanoseconds since the epoch.

        @return: The current time in nanoseconds.
        @rtype: int
        """
        return time.time_ns()

    def spin(self) -> None:
        """!
        Runs the node’s main loop, handling LCM messages continuously until the node is finished.

        The loop calls `self._lcm.handle()` to process incoming messages. If an OSError is encountered,
        the loop will stop and the node will shut down.
        """
        while not self._done:
            try:
                self._lcm.handle_timeout(0)
            except OSError as e:
                log.warning(f"LCM threw OSError {e}")
                self._done = True

    def _callback_suspend_node(self, channel, msg):
        """!
        Callback that suspends the node when triggered.

        @param channel: Unused service channel.
        @param msg: Service request message.
        @return: Empty :class:`flag_t` response.
        """
        self.suspend_node()
        return flag_t()
    
    def _callback_restart_node(self, channel, msg):
        """!
        Callback that restarts the node when triggered.

        @param channel: Unused service channel.
        @param msg: Service request message.
        @return: Empty :class:`flag_t` response.
        """
        self.restart_node()
        return flag_t()


    def suspend_communications(self, services=True) -> None:
        """!
        Suspends the node stopping comms handellers

        """
        # Unsubscribe all comm handlers
        for ch in self._comm_handlers:
            if ch.comm_type != 'Service':
                ch.suspend()
            elif ch.comm_type == 'Service' and services == True:
                ch.suspend()
        
        for m_ch in self._multi_comm_handlers:
            m_ch.suspend()


    def resume_communications(self, services=True) -> None:
        """!
        Resumes the node's communication handlers.
        """
        for ch in self._comm_handlers:
            if ch.comm_type != 'Service':
                ch.restart()
            elif ch.comm_type == 'Service' and services == True:
                ch.restart()
        
        for m_ch in self._multi_comm_handlers:
            m_ch.restart()

    def kill_node(self) -> None:
        """!
        Terminate the node process immediately.

        This method suspends the node and exits the program.
        """

        self.suspend_node()
        log.ok(f"Killing {self.name} Node")
        sys.exit(0)

    def suspend_node(self) -> None:
        """!
        Shuts down the node by stopping all communication handlers and steppers.

        Iterates through all registered communication handlers and steppers, shutting them down.
        """
        for ch in self._comm_handlers:
            if ch.comm_type != 'Service':
                ch.suspend()
            elif ch.comm_type == 'Service' and ch.register_with_registry == True:
                ch.suspend()

        for m_ch in self._multi_comm_handlers:
            m_ch.suspend()

        for s in self._steppers:
            s.suspend()
            
            
    def restart_node(self) -> None:
        """!
        Restart all communication handlers and steppers for the node.
        """
        for ch in self._comm_handlers:
            ch.restart()

        for m_ch in self._multi_comm_handlers:
            m_ch.restart()

        for s in self._steppers:
            s.restart()
    
    def send_service_request(self, service_name: str, request: object, response_type: type, timeout: int = 30) -> Any:
        """!
        Convenience wrapper around :func:`send_service_request`.

        @param service_name: Name of the service to call.
        @param request: Request object to send.
        @param response_type: Expected response type.
        @param timeout: Timeout in seconds.
        @return: The decoded response from the service.
        """
        return send_service_request(self.registry_host, self.registry_port, service_name, request, response_type, timeout)

