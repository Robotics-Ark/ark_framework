import shlex
import subprocess
from dataclasses import dataclass, field
from ark.executor.host import Host


@dataclass(frozen=True)
class NodeRunner:
    node_name: str
    env_name: str
    host: Host
    main: str
    parameters: dict[str, str | bool | int | float] = field(default_factory=dict)
    channel_remaps: dict[str, str] = field(default_factory=dict)
    conda_env: str | None = None
    z_config_path: str | None = None

    def launch(self) -> subprocess.Popen:
        params = [f"{name}:={value}" for name, value in self.parameters.items()]
        remaps = [f"{src}--{dst}" for src, dst in self.channel_remaps.items()]

        node_cmd = [
            "python",
            "-m",
            self.main,
            f"env_name:={self.env_name}",
            f"node_name:={self.node_name}",
            *params,
            *remaps,
        ]

        if self.z_config_path:
            node_cmd.append(f"z_config_path:={self.z_config_path}")

        if self.conda_env:
            conda_executable = "conda" if self.host.is_local else self.host.conda_path
            cmd = [conda_executable, "run", "-n", self.conda_env, *node_cmd]
        else:
            cmd = node_cmd

        if self.host.is_local:
            return subprocess.Popen(cmd)

        if self.host.os == "windows":
            remote_cmd = subprocess.list2cmdline(cmd)
        else:
            remote_cmd = shlex.join(cmd)

        return subprocess.Popen(["ssh", self.host.ssh_alias, remote_cmd])
