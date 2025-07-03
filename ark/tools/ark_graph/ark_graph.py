
import argparse
import json
import sys

# --- Third-party/project-specific imports ---
from ark.client.comm_handler.service import send_service_request
from ark.client.comm_infrastructure.endpoint import EndPoint
from ark.tools.log import log
from ark.global_constants import *
from arktypes import flag_t, network_info_t

import matplotlib.pyplot as plt
from graphviz import Digraph
import typer

from dataclasses import dataclass
from pathlib import Path

# Render the image with matplotlib
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
            ValueError: If the file extension is not '.gv' or '.dot'.
        """
        if path is None:
            path = Path(f"./{self.title}.gv")
        if isinstance(path, str):
            path = Path(path)

        if path.suffix not in [".gv", ".dot"]:
            raise ValueError("File extension must be '.gv' or '.dot'")

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
        self.num_nodes = n_nodes
        self.nodes = nodes


# ----------------------------------------------------------------------
#                        DECODING & HELPER FUNCTIONS
# ----------------------------------------------------------------------
def decode_network_info(lcm_message) -> NetworkInfo:
    """
    Converts an LCM network info message into a NetworkInfo object.

    Args:
        lcm_message (network_info_t): The LCM message containing network information.

    Returns:
        NetworkInfo: A NetworkInfo object with detailed node and comms information.
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
#                         GRAPHVIZ VISUALIZATION
# ----------------------------------------------------------------------
def graph_viz_plot(data: dict):
    """
    Generate a GraphViz diagram from the given network data and display it using Matplotlib.

    Args:
        data (dict): A dictionary containing the network information.
                     Typically the output of `network_info_lcm_to_dict(...)`.

    Returns:
        Image: The generated PIL Image containing the graph visualisation.

    Notes:
        Service nodes are drawn with single borders on the top and bottom and
        double borders on the left and right to resemble ``|| service ||``.
    """
    dot = Digraph(format="png")
    dot.attr("graph", fontname="Helvetica")
    dot.attr("node", fontname="Helvetica")
    dot.attr("edge", fontname="Helvetica")

    channel_id_map = {}
    id_counter = 1

    def get_channel_id(channel_name: str) -> str:
        nonlocal id_counter
        if channel_name not in channel_id_map:
            channel_id_map[channel_name] = f"ch_{id_counter}"
            id_counter += 1
        return channel_id_map[channel_name]

    # Build the graph
    for node in data["nodes"]:
        node_id = node["node_id"]
        node_name = node["name"]

        dot.node(node_id, node_name, shape="box", style="filled", fillcolor="lightblue")

        publishers = [pub["channel_name"] for pub in node["comms"]["publishers"]]
        subscribers = [sub["channel_name"] for sub in node["comms"]["subscribers"]]
        listeners = [lis["channel_name"] for lis in node["comms"]["listeners"]]
        services = [ser["service_name"] for ser in node["comms"]["services"]]

        for pub in publishers:
            pub_id = get_channel_id(pub)
            dot.node(pub_id, pub, shape="box", style="rounded,filled", fillcolor="white")
            dot.edge(node_id, pub_id)

        for sub in subscribers:
            sub_id = get_channel_id(sub)
            dot.node(sub_id, sub, shape="box", style="rounded,filled", fillcolor="white")
            dot.edge(sub_id, node_id)

        for lis in listeners:
            lis_id = get_channel_id(lis)
            dot.node(lis_id, lis, shape="box", style="rounded,filled", fillcolor="white")
            dot.edge(lis_id, node_id)

        for ser in services:
            if ser.startswith(DEFAULT_SERVICE_DECORATOR):
                continue
            ser_id = get_channel_id(ser)
            service_label = (
                            "<"
        "<TABLE BORDER='0' CELLBORDER='0' CELLPADDING='0' BGCOLOR='white'>"
        "  <TR><TD>"
        "    <TABLE BORDER='0' CELLBORDER='1' SIDES='LR' CELLPADDING='4' BGCOLOR='white'>"
        f"      <TR><TD>{ser}</TD></TR>"
        "    </TABLE>"
        "  </TD></TR>"
        "</TABLE>"
        ">"
            )
            dot.node(ser_id, label=service_label, shape="plaintext")
            dot.edge(node_id, ser_id)

    graph_image = dot.pipe()
    image_stream = io.BytesIO(graph_image)
    image = Image.open(image_stream)

    return image


# ----------------------------------------------------------------------
#                              MAIN CLASS
# ----------------------------------------------------------------------
class ArkGraph(EndPoint):
    """Endpoint that retrieves network info and renders a GraphViz diagram.

    The diagram can either be displayed immediately or saved for later use.

    Attributes:
        registry_host (str): The host of the registry server.
        registry_port (int): The port of the registry server.
        lcm_network_bounces (int): LCM network bounces for deeper network queries.
    """

    def __init__(
        self,
        registry_host: str = "127.0.0.1",
        registry_port: int = 1234,
        lcm_network_bounces: int = 1,
        *,
        display: bool = True,
    ):
        """
        Initializes the ArkGraph endpoint with registry configuration.

        Args:
            registry_host (str): The host address for the registry server.
            registry_port (int): The port for the registry server.
            lcm_network_bounces (int): LCM network bounces for deeper network queries.
            display (bool, optional): Whether to immediately display the diagram.
                If ``False``, the image can still be saved via :meth:`save_image`.
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

        # Generate the GraphViz diagram
        self.plot_image = graph_viz_plot(data)
        if display:
            self.display_image(self.plot_image)

    def save_image(self, file_path: str | Path) -> None:
        """Save the generated diagram image to ``file_path``.

        Only ``.png`` files are supported.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        if file_path.suffix.lower() != ".png":
            raise ValueError("File extension must be '.png'")

        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.plot_image.save(file_path)

    @staticmethod
    def get_cli_doc() -> str:
        """
        Return CLI help documentation.
        """
        return __doc__
    
    def display_image(self, plot_image):
        """
        Display the GraphViz diagram image using Matplotlib.

        Args:
            plot_image (Image.Image): The PIL Image containing the diagram.
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
    """Starts the graph with specified host and port."""
    server = ArkGraph(registry_host=registry_host, registry_port=registry_port)


@app.command()
def save(
    file_path: str,
    registry_host: str = typer.Option("127.0.0.1", "--host", help="The host address for the registry server."),
    registry_port: int = typer.Option(1234, "--port", help="The port for the registry server."),
):
    """Save the graph image to ``FILE_PATH`` without displaying it.

    ``FILE_PATH`` must end with ``.png``.
    """
    server = ArkGraph(registry_host=registry_host, registry_port=registry_port, display=False)
    try:
        server.save_image(file_path)
    except ValueError as exc:
        log.error(str(exc))
        raise typer.Exit(code=1)
    else:
        log.ok(f"Graph saved to {file_path}")

def main():
    """Entry point for the CLI."""
    app()  # Initializes the Typer CLI

if __name__ == "__main__":
    main()
