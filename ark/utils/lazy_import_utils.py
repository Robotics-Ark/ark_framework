import importlib
from types import ModuleType


class LazyImporter(ModuleType):
    """A lazily-loading proxy for a module and its submodules.

    This class replaces a module's global namespace to support lazy loading
    of submodules and member attributes. When an attribute is accessed, the
    importer first attempts to treat the name as a submodule. If that fails,
    it then attempts to retrieve the attribute from the wrapped module.

    """

    def __init__(self, module_name: str, module: ModuleType):
        """Initializes the lazy importer.

        Args:
            module_name (str): Name of the module being wrapped.
            module (ModuleType): The imported module instance.
        """
        super().__init__("lazy_" + module_name)
        self._module_path = module_name
        self._module = module
        self._not_module = set()
        self._submodules = {}

    def __getattr__(self, name: str):
        """Resolves an attribute access lazily.

        The attribute name is first checked as a possible submodule. If importing fails, the name is
        assumed to be a regular attribute of the wrapped module.

        Args:
            name : Name of attribute or submodule.

        Returns:
            Any: The resolved submodule or attribute.

        """
        if name not in self._not_module:
            submodule = self._try_load_submodule(name)
            if submodule:
                return submodule

            self._not_module.add(name)

        try:
            return getattr(self._module, name)
        except:
            raise AttributeError(
                f"module {self.__name__} has no attribute {name}"
            ) from None

    def _try_load_submodule(self, module_name: str):
        """Attempts to load a submodule lazily.

        Args:
            module_name: Submodule name relative to this module.

        Returns:
            LazyImporter | None: A LazyImporter for the submodule if found,
            otherwise None.
        """

        if self._module_path:
            module_name = f"{self._module_path}.{module_name}"

        if module_name in self._submodules:
            return self._submodules[module_name]

        try:
            wrapper = LazyImporter(module_name, importlib.import_module(module_name))
            self._submodules[module_name] = wrapper
            return wrapper
        except ModuleNotFoundError:
            return None
