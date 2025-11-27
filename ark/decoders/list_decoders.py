import typer
from ark.decoders.registry import DECODER_REGISTRY

app = typer.Typer()


@app.command("list")
def list_decoders() -> None:
    """
    Display all registered observation and action decoders.
    Returns:
        None
    """
    if not DECODER_REGISTRY:
        typer.echo("No decoders registered.")
        return

    typer.echo("Available decoders:")
    for name in sorted(DECODER_REGISTRY.keys()):
        typer.echo(f"- {name}")


def main():
    app()


if __name__ == "__main__":
    main()
