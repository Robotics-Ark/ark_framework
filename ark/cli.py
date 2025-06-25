
import typer

from ark.client.comm_infrastructure import registry
from ark.tools.ark_graph import ark_graph
from ark.tools import launcher
# from ark.tools.image_viewer import image_viewer




app = typer.Typer()

app.add_typer(registry.app, name="registry")
app.add_typer(ark_graph.app, name="graph")
app.add_typer(launcher.app, name="launcher")
# app.add_typer(image_viewer.app, name="image_viewer")

def main():
    """Main CLI entry point."""
    app()  

if __name__ == "__main__":
    main()
