import zenoh
import json

z_cfg = {"mode": "peer", "connect": {"endpoints": ["udp/127.0.0.1:7447"]}}


def default_session() -> zenoh.Session:
    """Create and return a default zenoh session."""
    return zenoh.open(zenoh.Config.from_json5(json.dumps(z_cfg)))
