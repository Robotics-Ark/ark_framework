from importlib import import_module


def load_from_spec(spec: str):
    """
    Load and return an object from a string like:

        "mypackage.module:MyClass"

    Parameters
    ----------
    spec: str
        The string specification of the object to load.
        Should be in the format "package.module:ClassName", where
        the part before the colon is the module path and the part
        after the colon is the attribute path within that module.

    Returns
    -------
    object: type
        The resolved object itself, e.g. MyClass.

    Raises
    ------
    ValueError
        If the spec string is not in the correct format.
    ImportError
        If the module cannot be imported or the attribute cannot be resolved.
    """
    try:
        module_path, attr_path = spec.split(":", 1)
    except ValueError as e:
        raise ValueError(
            f"Invalid spec {spec!r}. Expected format 'package.module:ClassName'"
        ) from e

    module = import_module(module_path)

    obj = module
    for part in attr_path.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as e:
            raise ImportError(
                f"Could not resolve attribute path {attr_path!r} in module {module_path!r}"
            ) from e

    return obj
