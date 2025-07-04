import typing
import typer

from ark.client.comm_handler.service import send_service_request
from ark.global_constants import DEFAULT_SERVICE_DECORATOR
from ark.tools.ark_graph.ark_graph import network_info_lcm_to_dict
from arktypes import flag_t, network_info_t

node = typer.Typer(help="Interact with nodes", invoke_without_command=True)
channel = typer.Typer(help="Inspect channels")
service = typer.Typer(help="Inspect services")


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


@node.callback(invoke_without_command=True)
def show_node(
    ctx: typer.Context,
    name: typing.Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Name of the node to inspect",
    ),
    host: str = "127.0.0.1",
    port: int = 1234,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show default services"),
):
    """Show information about NODE if provided via ``-n/--name``."""
    if ctx.invoked_subcommand is not None:
        return
    if name is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()
    data = _fetch_network_info(host, port)
    for node_info in data.get("nodes", []):
        if node_info.get("name") == name:
            comms = node_info.get("comms", {})
            def _print_section(title: str, items: list, key: str):
                print(f"{title}:")
                if not items:
                    print("  <none>")
                else:
                    for it in items:
                        print(f"  {it.get(key)} ({it.get('channel_type') if 'channel_type' in it else it.get('request_type')}" +
                              (f" -> {it.get('response_type')}" if 'response_type' in it else "") + ")")
            _print_section("Listeners", comms.get("listeners", []), "channel_name")
            _print_section("Publishers", comms.get("publishers", []), "channel_name")
            _print_section("Subscribers", comms.get("subscribers", []), "channel_name")
            print("Services:")
            services = comms.get("services", [])
            if not services:
                print("  <none>")
            else:
                for srv in services:
                    name = srv.get("service_name")
                    if not verbose and name.startswith(DEFAULT_SERVICE_DECORATOR):
                        continue
                    print(
                        f"  {name} ({srv.get('request_type')} -> {srv.get('response_type')})"
                    )
            return
    typer.echo(f"Node '{name}' not found.")


@node.command("list")
def list_nodes(host: str = "127.0.0.1", port: int = 1234):
    """List active nodes."""
    data = _fetch_network_info(host, port)
    for node_info in data.get("nodes", []):
        print(node_info.get("name"))


@channel.command("list")
def list_channels(host: str = "127.0.0.1", port: int = 1234):
    """List active channels."""
    data = _fetch_network_info(host, port)
    channels = set()
    for node_info in data.get("nodes", []):
        comms = node_info.get("comms", {})
        for comp in ("listeners", "subscribers", "publishers"):
            for ch in comms.get(comp, []):
                if ch.get("channel_status"):
                    channels.add(ch.get("channel_name"))
    for ch in sorted(channels):
        print(ch)


@service.command("list")
def list_services(
    host: str = "127.0.0.1",
    port: int = 1234,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show default services"),
):
    """List available services."""
    data = _fetch_network_info(host, port)
    services = set()
    for node_info in data.get("nodes", []):
        for srv in node_info.get("comms", {}).get("services", []):
            name = srv.get("service_name")
            if not verbose and name.startswith(DEFAULT_SERVICE_DECORATOR):
                continue
            services.add(name)
    for srv in sorted(services):
        print(srv)


if __name__ == "__main__":
    app = typer.Typer()
    app.add_typer(node, name="node")
    app.add_typer(channel, name="channel")
    app.add_typer(service, name="service")
    app()
