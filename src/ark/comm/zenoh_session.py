import os
import zenoh
import json
from pathlib import Path

_SHM_AVAILABLE = hasattr(zenoh, "shm")


def _default_cfg() -> dict:
    router = os.environ.get("ARK_ZENOH_ROUTER", "").strip()
    if router:
        cfg: dict = {"mode": "client", "connect": {"endpoints": [f"tcp/{router}"]}}
    else:
        cfg = {"mode": "peer"}
    if _SHM_AVAILABLE:
        cfg["transport"] = {"shared_memory": {"enabled": True}}
    return cfg


def default_session() -> zenoh.Session:
    return zenoh.open(zenoh.Config.from_json5(json.dumps(_default_cfg())))


def load_session(config_path: str | Path | None) -> zenoh.Session:
    if config_path is None:
        return default_session()
    z_cfg = zenoh.Config.from_json5(Path(str(config_path)).read_text())
    return zenoh.open(z_cfg)
