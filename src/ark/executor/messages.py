import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class RunRequest:
    host: str
    env_name: str
    command: str | list[str]
    log_file: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def from_json(cls, data) ->"RunRequest":
        return cls(**json.loads(bytes(data)))


@dataclass
class RunNodeRequest:
    host: str
    env_name: str
    conda_env: str
    script: str
    node_name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    channel_remaps: dict[str, str] = field(default_factory=dict)
    log_file: str = ""

    @classmethod
    def from_json(cls, data) ->"RunNodeRequest":
        return cls(**json.loads(bytes(data)))


@dataclass
class KillRequest:
    id: str

    @classmethod
    def from_json(cls, data) ->"KillRequest":
        return cls(**json.loads(bytes(data)))


@dataclass
class KillEnvRequest:
    env_name: str

    @classmethod
    def from_json(cls, data) ->"KillEnvRequest":
        return cls(**json.loads(bytes(data)))


@dataclass
class ProcessInfo:
    id: str
    env_name: str
    is_node: bool
    node_name: str | None
    running: bool


@dataclass
class RunReply:
    success: bool
    error: str = ""

    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode()
