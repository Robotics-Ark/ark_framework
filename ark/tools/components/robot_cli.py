import importlib
import inspect
from functools import wraps
import typer
from ark.tools.log import log
from ark.tools.components.cli_helpers import package_installation
from ark.client.comm_infrastructure.base_node import main

app = typer.Typer()


@app.command()  # Let Typer see the (wrapped) signature
@package_installation("package_name")  # Our check runs before the command body
def franka(
    name: str = typer.Option("franka", "--name", "-n", help="Name of the robot"),
    config: str = typer.Option(
        None, "--config", "-c", help="Path to the configuration file"
    ),
    package_name: str = typer.Option("ark_franka", hidden=True),
):
    typer.echo(f"start {name}")

    from ark_franka.franka_panda import FrankaPanda
    from ark_franka.franka_driver import FrankaResearch3Driver


    driver = FrankaResearch3Driver(name, config)
    main(FrankaPanda, name, config, driver)

    print(f"Configuration file: {config if config else 'None'}")
    print(f"Package name: {package_name}")