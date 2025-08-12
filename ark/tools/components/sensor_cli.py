import importlib
import inspect
from functools import wraps
import typer
from ark.tools.log import log
from ark.tools.components.cli_helpers import package_installation

app = typer.Typer()


@app.command()  # Let Typer see the (wrapped) signature
@package_installation("package_name")  # Our check runs before the command body
def intel_real_sense(
    name: str = typer.Option(
        "intel_real_sense", "--name", "-n", help="Name of the robot"
    ),
    config: str = typer.Option(
        None, "--config", "-c", help="Path to the configuration file"
    ),
    package_name: str = typer.Option("intel_real_sense", hidden=True),
):
    typer.echo(f"start {name}")
    print(f"Configuration file: {config if config else 'None'}")
    print(f"Package name: {package_name}")
