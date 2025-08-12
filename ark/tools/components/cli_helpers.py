import importlib
import inspect
from functools import wraps
import typer
from ark.tools.log import log


def is_package_installed(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def package_installation(package_param: str = "package_name"):
    """Decorator that checks a package argument is importable before running the command."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the package name from kwargs or from defaults
            pkg = kwargs.get(package_param)
            if pkg is None:
                sig = inspect.signature(func)
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                pkg = bound.arguments.get(package_param)

            if pkg and not is_package_installed(pkg):
                log.error(
                    f" Required package '{pkg}' is not installed.\n"
                    f"   Try: pip install ark-robotics[{pkg}]"
                )
                raise typer.Exit(code=1)

            return func(*args, **kwargs)

        return wrapper

    return decorator
