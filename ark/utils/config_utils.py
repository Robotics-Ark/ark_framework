def resolve_class(path: str) -> type:
    """Resolve a fully-qualified class path like 'module.submodule:Class'."""

    module_name, _, class_name = path.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"Invalid class path '{path}'")
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"Class '{class_name}' not found in module '{module_name}'")
    return cls
