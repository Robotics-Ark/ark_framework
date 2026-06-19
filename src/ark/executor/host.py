import os
import platform
from dataclasses import dataclass

SUPPORTED_OS = {"windows", "linux", "darwin"}


@dataclass(frozen=True, slots=True)
class Host:
    name: str  # unique identifier for this host
    os: str  # "windows", "linux", "darwin"
    ssh_alias: str  # alias from ~/.ssh/config, e.g. "pc1"; unused for local hosts
    conda_path: str  # full path to conda executable on remote; unused for local hosts
    is_local: bool = False

    def __post_init__(self):
        if self.os not in SUPPORTED_OS:
            raise ValueError(
                f"Unsupported OS '{self.os}'. Supported OS:\n{SUPPORTED_OS}"
            )

    @classmethod
    def local(cls) -> "Host":
        return cls(
            name="local",
            os=platform.system().lower(),
            ssh_alias="",
            conda_path=os.environ.get("CONDA_EXE", ""),
            is_local=True,
        )
