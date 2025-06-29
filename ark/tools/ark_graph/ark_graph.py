
"""! Tools for visualising node graphs and communications.

This module generates mermaid diagrams representing the connections between ARK
nodes.  It can also query running nodes for their communication details.
"""

import argparse
import json
import sys

# --- Third-party/project-specific imports ---
from ark.client.comm_handler.service import Service, send_service_request
from ark.client.comm_infrastructure.endpoint import EndPoint
from ark.tools.log import log
from ark.global_constants import *
from arktypes import flag_t, network_info_t

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import mermaid as mmd
from mermaid.graph import Graph
import typer

from dataclasses import dataclass
from pathlib import Path

# Render the image with matplotlib
import base64
import io
from PIL import Image

app = typer.Typer()

DEFAULT_SERVICE_DECORATOR = "__DEFAULT_SERVICE"
# ----------------------------------------------------------------------
#                             DATA CLASSES
# ----------------------------------------------------------------------
@dataclass
class BaseGraph:
    """
    Graph base class.

    This class serves as a base for other classes representing different
    types of diagrams like `Flowchart`, `ERDiagram`, etc.

    Attributes:
        title (str): The title of the diagram.
        script (str): The main script to create the diagram.
    """
    title: str
    script: str

    def save(self, path=None) -> None:
        """
        Save the diagram to a file.

        Args:
            path (Optional[Union[Path,str]]): The path to save the diagram. If not
                provided, the diagram will be saved in the current directory
                with the title as the filename.

        Raises:
            ValueError: If the file extension is not '.mmd' or '.mermaid'.
        """
        if path is None:
            path = Path(f"./{self.title}.mmd")
        if isinstance(path, str):
            path = Path(path)

        if path.suffix not in [".mmd", ".mermaid"]:
            raise ValueError("File extension must be '.mmd' or '.mermaid'")

        with open(path, "w") as file:
            file.write(self.script)

    def _build_script(self) -> None:
        """
        Internal helper to finalize the script content for the diagram.
        """
        script: str = f"---\ntitle: {self.title}\n---"
        script += self.script
        self.script = script


class ServiceInfo:
    """
    Encapsulates service-related information for a node.

    Attributes:
        comms_type (str): The communications type (e.g., TCP, UDP, etc.).
        service_name (str): The name of the service.
        service_host (str): The hostname/IP of the service.
        service_port (int): The port used by the service.
        registry_host (str): The registry host for service discovery.
        registry_port (int): The registry port for service discovery.
        request_type (str): The request LCM type.
        response_type (str): The response LCM type.
    """

    def __init__(
        self,
        comms_type: str,
        service_name: str,
        service_host: str,
        service_port: int,
        registry_host: str,
        registry_port: int,
        request_type: str,
        response_type: str
    ):
        """! Construct a ``ServiceInfo`` describing one service endpoint."""
        self.comms_type = comms_type
        self.service_name = service_name
        self.service_host = service_host
        self.service_port = service_port
        self.registry_host = registry_host
        self.registry_port = registry_port
        self.request_type = request_type
        self.response_type = response_type


class ListenerInfo:
    """
    Encapsulates listener-related information for a node.

    Attributes:
        comms_type (str): The communications type (e.g., LCM).
        channel_name (str): The name of the channel.
        channel_type (str): The message type on that channel.
        channel_status (str): The status (e.g., active/inactive).
    """

    def __init__(
        self,
        comms_type: str,
        channel_name: str,
        channel_type: str,
        channel_status: str
    ):
        """! Construct ``ListenerInfo`` for a subscribed LCM channel."""
        self.comms_type = comms_type
        self.channel_name = channel_name
        self.channel_type = channel_type
        self.channel_status = channel_status


class SubscriberInfo:
    """
    Encapsulates subscriber-related information for a node.

    Attributes:
        comms_type (str): The communications type (e.g., LCM).
        channel_name (str): The name of the channel.
        channel_type (str): The message type on that channel.
        channel_status (str): The status (e.g., active/inactive).
    """

    def __init__(
        self,
        comms_type: str,
        channel_name: str,
        channel_type: str,
        channel_status: str
    ):
        """! Construct ``SubscriberInfo`` for an outgoing subscription."""
        self.comms_type = comms_type
        self.channel_name = channel_name
        self.channel_type = channel_type
        self.channel_status = channel_status


class PublisherInfo:
    """
    Encapsulates publisher-related information for a node.

    Attributes:
        comms_type (str): The communications type (e.g., LCM).
        channel_name (str): The name of the channel.
        channel_type (str): The message type on that channel.
        channel_status (str): The status (e.g., active/inactive).
    """

    def __init__(
        self,
        comms_type: str,
        channel_name: str,
        channel_type: str,
        channel_status: str
    ):
        """! Construct ``PublisherInfo`` for a published channel."""
        self.comms_type = comms_type
        self.channel_name = channel_name
        self.channel_type = channel_type
        self.channel_status = channel_status


class CommsInfo:
    """
    Encapsulates all communications (listeners/subscribers/publishers/services) for a node.

    Attributes:
        n_listeners (int): Number of listeners on this node.
        listeners (List[ListenerInfo]): A list of listener info objects.
        n_subscribers (int): Number of subscribers on this node.
        subscribers (List[SubscriberInfo]): A list of subscriber info objects.
        n_publishers (int): Number of publishers on this node.
        publishers (List[PublisherInfo]): A list of publisher info objects.
        n_services (int): Number of services on this node.
        services (List[ServiceInfo]): A list of service info objects.
    """

    def __init__(
        self,
        n_listeners: int,
        listeners: list,
        n_subscribers: int,
        subscribers: list,
        n_publishers: int,
        publishers: list,
        n_services: int,
        services: list
    ):
        """! Aggregate all communication endpoints of a node."""
        self.n_listeners = n_listeners
        self.listeners = listeners
        self.n_subscribers = n_subscribers
        self.subscribers = subscribers
        self.n_publishers = n_publishers
        self.publishers = publishers
        self.n_services = n_services
        self.services = services


class NodeInfo:
    """
    Encapsulates information about a single node in the network.

    Attributes:
        node_name (str): The name of the node (e.g., "Camera").
        node_id (str): A unique identifier for the node.
        comms (CommsInfo): Communication details for the node.
    """

    def __init__(self, node_name: str, node_id: str, comms: CommsInfo):
        """! Container for a node and its communication details."""
        self.name = node_name
        self.node_id = node_id
        self.comms = comms


class NetworkInfo:
    """
    Encapsulates network-level information for multiple nodes.

    Attributes:
        num_nodes (int): The number of nodes in the network.
        nodes (List[NodeInfo]): A list of NodeInfo objects.
    """

    def __init__(self, n_nodes: int, nodes: list):
        """! Wrapper for the entire network graph."""
        self.num_nodes = n_nodes
        self.nodes = nodes


# ----------------------------------------------------------------------
#                        DECODING & HELPER FUNCTIONS
# ----------------------------------------------------------------------
def decode_network_info(lcm_message) -> NetworkInfo:
    """!
    Converts an LCM network info message into a NetworkInfo object.

    @param lcm_message The raw ``network_info_t`` LCM message.
    @return ``NetworkInfo`` object with detailed node and comms information.
    """
    return NetworkInfo(
        n_nodes=lcm_message.n_nodes,
        nodes=[
            NodeInfo(
                node_name=node.node_name,
                node_id=node.node_id,
                comms=CommsInfo(
                    n_listeners=node.comms.n_listeners,
                    listeners=[
                        ListenerInfo(
                            comms_type=listener.comms_type,
                            channel_name=listener.channel_name,
                            channel_type=listener.channel_type,
                            channel_status=listener.channel_status
                        )
                        for listener in node.comms.listeners
                    ],
                    n_subscribers=node.comms.n_subscribers,
                    subscribers=[
                        SubscriberInfo(
                            comms_type=subscriber.comms_type,
                            channel_name=subscriber.channel_name,
                            channel_type=subscriber.channel_type,
                            channel_status=subscriber.channel_status
                        )
                        for subscriber in node.comms.subscribers
                    ],
                    n_publishers=node.comms.n_publishers,
                    publishers=[
                        PublisherInfo(
                            comms_type=publisher.comms_type,
                            channel_name=publisher.channel_name,
                            channel_type=publisher.channel_type,
                            channel_status=publisher.channel_status
                        )
                        for publisher in node.comms.publishers
                    ],
                    n_services=node.comms.n_services,
                    services=[
                        ServiceInfo(
                            comms_type=service.comms_type,
                            service_name=service.service_name,
                            service_host=service.service_host,
                            service_port=service.service_port,
                            registry_host=service.registry_host,
                            registry_port=service.registry_port,
                            request_type=service.request_type,
                            response_type=service.response_type
                        )
                        for service in node.comms.services
                    ]
                )
            )
            for node in lcm_message.nodes
        ]
    )


def network_info_lcm_to_dict(lcm_message) -> dict:
    """
    Converts an LCM network info message into a Python dictionary,
    allowing easy serialization or manipulation.

    Args:
        lcm_message (network_info_t): The LCM message containing network information.

    Returns:
        dict: A dictionary representation of the network info.
    """
    network_info_obj = decode_network_info(lcm_message)
    return json.loads(json.dumps(network_info_obj, default=lambda o: o.__dict__))


# ----------------------------------------------------------------------
#                         MERMAID VISUALIZATION
# ----------------------------------------------------------------------
def mermaid_plot(data: dict):
    """
    Generate a Mermaid diagram from the given network data and display it using Matplotlib.

    Args:
        data (dict): A dictionary containing the network information.
                     Typically the output of `network_info_lcm_to_dict(...)`.

    Returns:
        Mermaid: A mermaid diagram object (useful if further manipulation is needed).
    """
    graph_lines = ["graph LR"]
    # Define some default class styles
    graph_lines.append("classDef node fill:#add8e6,stroke:#000,stroke-width:2px;")
    graph_lines.append("classDef channel fill:#ffffff,stroke:#000,stroke-width:2px;")

    # We will map channel names to IDs so every time we see the same channel name,
    # we reuse the same ID, ensuring consistent links in the diagram.
    channel_id_map = {}
    id_counter = 1

    def get_channel_id(channel_name: str) -> str:
        """
        Return a unique channel ID if it doesn't exist,
        or return the existing one from our dictionary.
        """
        nonlocal id_counter
        if channel_name not in channel_id_map:
            channel_id_map[channel_name] = f"ch_{id_counter}"
            id_counter += 1
        return channel_id_map[channel_name]

    # For each node, define its communication and connections
    for node in data["nodes"]:
        node_id = node["node_id"]
        node_name = node["name"]

        publishers = [pub["channel_name"] for pub in node["comms"]["publishers"]]
        subscribers = [sub["channel_name"] for sub in node["comms"]["subscribers"]]
        listeners = [lis["channel_name"] for lis in node["comms"]["listeners"]]
        services = [ser["service_name"] for ser in node["comms"]["services"]]

        # Publisher edges: node -> channel
        for pub in publishers:
            pub_id = get_channel_id(pub)
            graph_lines.append(
                f"{node_id}([{node_name}]):::node --> {pub_id}[{pub}]:::channel"
            )

        # Subscriber edges: channel -> node
        for sub in subscribers:
            sub_id = get_channel_id(sub)
            graph_lines.append(
                f"{sub_id}[{sub}]:::channel --> {node_id}([{node_name}]):::node"
            )

        # Listener edges: channel -> node
        for lis in listeners:
            lis_id = get_channel_id(lis)
            graph_lines.append(
                f"{lis_id}[{lis}]:::channel --> {node_id}([{node_name}]):::node"
            )

        # Service edges: node -> service
        for ser in services:
            # Skip default service placeholders
            if ser.startswith(DEFAULT_SERVICE_DECORATOR):
                continue
            ser_id = get_channel_id(ser)
            graph_lines.append(
                f"{node_id}([{node_name}]):::node --> {ser_id}[[{ser}]]:::channel"
            )

    # Combine all lines into a single Mermaid script
    script = "\n".join(graph_lines) + "\n"

    # Create a mermaid graph and convert to image
    mermaid_graph = Graph("Ark Graph", script)
    mermaid_obj = mmd.Mermaid(mermaid_graph)
    graph_image = mermaid_obj.img_response.content



    image_stream = io.BytesIO(graph_image)
    image = Image.open(image_stream)

    return image


# ----------------------------------------------------------------------
#                              MAIN CLASS
# ----------------------------------------------------------------------
class ArkGraph(EndPoint):
    """
    ArkGraph is an EndPoint that retrieves network information from
    a registry service and displays it as a Mermaid diagram.

    Attributes:
        registry_host (str): The host of the registry server.
        registry_port (int): The port of the registry server.
        lcm_network_bounces (int): LCM network bounces for deeper network queries.
    """

    def __init__(
        self,
        registry_host: str = "127.0.0.1",
        registry_port: int = 1234,
        lcm_network_bounces: int = 1
    ):
        """
        Initializes the ArkGraph endpoint with registry configuration.

        Args:
            registry_host (str): The host address for the registry server.
            registry_port (int): The port for the registry server.
            lcm_network_bounces (int): LCM network bounces for deeper network queries.
        """
        config = { "network": {
            "registry_host": registry_host,
            "registry_port": registry_port,
            "lcm_network_bounces": lcm_network_bounces
            }
        }
        super().__init__(config)

        # Query the registry for network information
        req = flag_t()
        response_lcm = send_service_request(
            self.registry_host, 
            self.registry_port,
            f"{DEFAULT_SERVICE_DECORATOR}/GetNetworkInfo",
            req, 
            network_info_t
        )

        # Convert LCM response to a dictionary
        data = network_info_lcm_to_dict(response_lcm)

        # Generate the Mermaid diagram and display it
        plot_image = mermaid_plot(data)
        self.display_image(plot_image)

    @staticmethod
    def get_cli_doc() -> str:
        """
        Return CLI help documentation.
        """
        return __doc__
    
    def display_image(self, plot_image):
        """
        Display the Mermaid diagram image using Matplotlib.

        Args:
            plot_image (Mermaid): The Mermaid object containing the diagram.
        """
        plt.imshow(plot_image)
        plt.axis("off")
        plt.show()


# ----------------------------------------------------------------------
#                           COMMAND-LINE INTERFACE
# ----------------------------------------------------------------------
def parse_args():
    """
    Parse command-line arguments for running ArkGraph as a script.

    Returns:
        argparse.Namespace: The parsed arguments with `registry_host` and `registry_port`.
    """
    parser = argparse.ArgumentParser(description="ArkGraph - Visualize your NOAHR network.")
    parser.add_argument(
        "--registry_host",
        type=str,
        default="127.0.0.1",
        help="The host address for the registry server."
    )
    parser.add_argument(
        "--registry_port",
        type=int,
        default=1234,
        help="The port for the registry server."
    )
    return parser.parse_args()


# ----------------------------------------------------------------------
#                                MAIN
# ----------------------------------------------------------------------
@app.command()
def start(
    registry_host: str = typer.Option("127.0.0.1", "--host", help="The host address for the registry server."),
    registry_port: int = typer.Option(1234, "--port", help="The port for the registry server.")
):
    """! Starts the graph with specified host and port."""
    server = ArkGraph(registry_host=registry_host, registry_port=registry_port)

def main():
    """! Entry point for the CLI."""
    app()  # Initializes the Typer CLI

if __name__ == "__main__":
    main()