"""Parse a YAML launch file into typed config objects."""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from ark.executor.host import Host
from ark.executor.node import NodeRunner


NodeMode = Literal["sim", "real", "both"]


@dataclass(frozen=True)
class NodeSpec:
    """Parsed representation of one node entry in the launch YAML."""

    name: str
    host_name: str
    main: str
    mode: NodeMode = "both"
    conda_env: str | None = None
    z_config_path: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    channel_remaps: dict[str, str] = field(default_factory=dict)


@dataclass
class LaunchConfig:
    """Parsed launch YAML.

    YAML schema::

        env_name: my_env
        sim: true          # default mode; overridable at launch time

        hosts:
          local:
            os: linux      # "linux" | "darwin" | "windows"
            is_local: true
          robot_pc:
            os: linux
            ssh_alias: robotpc
            conda_path: /home/user/miniconda3/bin/conda

        nodes:
          simulator:
            host: local
            main: ark.simulator.impl.pybullet   # python -m entry point
            mode: sim                            # "sim" | "real" | "both"
            conda_env: ark_sim                   # optional
            z_config_path: /path/to/cfg.json5   # optional
            parameters:
              time_step_sec: 0.001
            channel_remaps:
              old_name: new_name

          franka_driver:
            host: robot_pc
            main: my_robot_pkg.franka_driver
            mode: real
    """

    env_name: str
    sim: bool
    hosts: dict[str, Host]
    node_specs: list[NodeSpec]

    @classmethod
    def from_file(cls, path: str | Path) -> "LaunchConfig":
        text = Path(path).read_text()
        data = yaml.safe_load(text)
        return cls._parse(data)

    @classmethod
    def from_dict(cls, data: dict) -> "LaunchConfig":
        return cls._parse(data)

    @classmethod
    def _parse(cls, data: dict) -> "LaunchConfig":
        env_name: str = str(data["env_name"])
        sim: bool = bool(data.get("sim", False))

        hosts: dict[str, Host] = {}
        for host_name, host_data in data.get("hosts", {}).items():
            is_local = bool(host_data.get("is_local", False))
            hosts[host_name] = Host(
                name=host_name,
                os=str(host_data.get("os", platform.system().lower())),
                ssh_alias=str(host_data.get("ssh_alias", "")),
                conda_path=str(host_data.get("conda_path", "")),
                is_local=is_local,
            )

        if not hosts:
            hosts["local"] = Host.local()

        node_specs: list[NodeSpec] = []
        for node_name, node_data in data.get("nodes", {}).items():
            node_specs.append(NodeSpec(
                name=node_name,
                host_name=str(node_data.get("host", "local")),
                main=str(node_data["main"]),
                mode=str(node_data.get("mode", "both")),  # type: ignore[arg-type]
                conda_env=node_data.get("conda_env"),
                z_config_path=node_data.get("z_config_path"),
                parameters=dict(node_data.get("parameters", {})),
                channel_remaps=dict(node_data.get("channel_remaps", {})),
            ))

        return cls(
            env_name=env_name,
            sim=sim,
            hosts=hosts,
            node_specs=node_specs,
        )

    def node_runners(
        self, sim: bool, extra_params: dict[str, Any] | None = None
    ) -> list[NodeRunner]:
        """Return NodeRunner objects for the nodes active in the given mode."""
        runners = []
        for spec in self.node_specs:
            if spec.mode == "sim" and not sim:
                continue
            if spec.mode == "real" and sim:
                continue
            host = self.hosts.get(spec.host_name)
            if host is None:
                raise ValueError(
                    f"Node '{spec.name}' references unknown host '{spec.host_name}'. "
                    f"Defined hosts: {list(self.hosts)}"
                )
            params = {"sim": sim, **spec.parameters}
            if extra_params:
                params.update(extra_params)
            runners.append(NodeRunner(
                node_name=spec.name,
                env_name=self.env_name,
                host=host,
                main=spec.main,
                parameters=params,
                channel_remaps=spec.channel_remaps,
                conda_env=spec.conda_env,
                z_config_path=spec.z_config_path,
            ))
        return runners
