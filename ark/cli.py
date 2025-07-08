
import typer

from ark.client.comm_infrastructure import registry
from ark.tools.ark_graph import ark_graph
from ark.tools import launcher
from ark.tools import network
from ark.tools.visualization import image_viewer

app = typer.Typer()

# Core tooling
app.add_typer(registry.app, name="registry")
app.add_typer(ark_graph.app, name="graph")
app.add_typer(launcher.app, name="launcher")
# Network inspection utilities
app.add_typer(network.node, name="node")
app.add_typer(network.channel, name="channel")
app.add_typer(network.service, name="service")
app.add_typer(image_viewer.app, name="view")

def main():
    """Main CLI entry point."""
    app()  

if __name__ == "__main__":
    main()
