from pathlib import Path

import yaml
from ark.tools.log import log


def load_yaml(config_path: str, raise_fnf_error=True) -> dict:
    """
    Load a YAML configuration schema from a file.
    Args:
        config_path: Path to the YAML configuration file.
        raise_fnf_error: Raise FileNotFoundError if the YAML configuration file does not exist.

    Returns:
        The parsed configuration schema as a dictionary. If the file
        contains no data, an empty dictionary is returned.

    """
    cfg_path = Path(config_path)
    cfg_dict = {}
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            cfg_dict = yaml.safe_load(f) or {}
    else:
        if raise_fnf_error:
            raise FileNotFoundError(f"Config file could not found {cfg_path}")
        log.error(f"Config file {cfg_path} does not exist.")

    return cfg_dict
