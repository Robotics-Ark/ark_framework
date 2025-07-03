import typer

from ark.client.comm_handler.service import send_service_request
from ark.global_constants import DEFAULT_SERVICE_DECORATOR
from ark.tools.ark_graph.ark_graph import network_info_lcm_to_dict
from arktypes import flag_t, network_info_t

info = typer.Typer(help="Query information about the running ARK network")


def _fetch_network_info(host: str, port: int) -> dict:
    req = flag_t()
    lcm_msg = send_service_request(
        host,
        port,
        f"{DEFAULT_SERVICE_DECORATOR}/GetNetworkInfo",
        req,
        network_info_t,
    )
    return network_info_lcm_to_dict(lcm_msg)


@info.command()
def nodes(host: str = "127.0.0.1", port: int = 1234):
    """List active nodes."""
    data = _fetch_network_info(host, port)
    for node in data.get("nodes", []):
        print(node.get("name"))


@info.command()
def channels(host: str = "127.0.0.1", port: int = 1234):
    """List active channels."""
    data = _fetch_network_info(host, port)
    channels = set()
    for node in data.get("nodes", []):
        comms = node.get("comms", {})
        for comp in ("listeners", "subscribers", "publishers"):
            for ch in comms.get(comp, []):
                if ch.get("channel_status"):
                    channels.add(ch.get("channel_name"))
    for ch in sorted(channels):
        print(ch)


@info.command()
def services(host: str = "127.0.0.1", port: int = 1234):
    """List available services."""
    data = _fetch_network_info(host, port)
    services = set()
    for node in data.get("nodes", []):
        comms = node.get("comms", {})
        for srv in comms.get("services", []):
            services.add(srv.get("service_name"))
    for srv in sorted(services):
        print(srv)


def main():
    info()


if __name__ == "__main__":
    main()
