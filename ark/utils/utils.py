import json
from pathlib import Path

import yaml
from ark.tools.log import log


class ConfigPath(type(Path())):
    """
    A Path subclass with convenience methods for reading configuration files.
    Works cross-platform (inherits PosixPath on Linux/macOS or WindowsPath on Windows).
    """

    @property
    def str(self) -> str:
        """
        Return the string representation of this path.
        Equivalent to calling str(self).
        """
        return str(self)

    def read_yaml(self, raise_fnf_error: bool = True) -> dict:
        """
        Load a YAML configuration schema from this path.

        Args:
            raise_fnf_error: If True, raise FileNotFoundError when the file is missing.

        Returns:
            Parsed YAML as a dictionary. Returns {} if empty.
        """
        if self.exists():
            with self.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        else:
            if raise_fnf_error:
                raise FileNotFoundError(f"Config file not found: {self}")
            log.error(f"Config file {self} does not exist.")
            return {}

    def read_json(self, raise_fnf_error: bool = True) -> dict:
        """
        Load a JSON configuration schema from this path.

        Args:
            raise_fnf_error: If True, raise FileNotFoundError when the file is missing.

        Returns:
            Parsed JSON as a dictionary.
        """
        if self.exists():
            with self.open("r", encoding="utf-8") as f:
                return json.load(f)
        else:
            if raise_fnf_error:
                raise FileNotFoundError(f"Config file not found: {self}")
            log.error(f"Config file {self} does not exist.")
            return {}

    def __repr__(self) -> str:
        return f"<ConfigPath path={super().__str__()}>"
