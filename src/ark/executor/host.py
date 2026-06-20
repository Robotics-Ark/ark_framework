import os
import platform
import shlex
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO

_FILE = int | IO[bytes] | None

SUPPORTED_OS = {"windows", "linux", "darwin"}

_SSH_OPTS = ["-o", "ConnectTimeout=5", "-o", "BatchMode=yes"]


def _ssh_config_aliases() -> set[str]:
    config_path = os.path.expanduser("~/.ssh/config")
    if not os.path.exists(config_path):
        return set()
    aliases = set()
    with open(config_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts and parts[0].lower() == "host":
                for alias in parts[1:]:
                    if not any(c in alias for c in ("*", "?", "!")):
                        aliases.add(alias)
    return aliases


class Host(ABC):

    def __init__(self, name: str, os: str, conda_path: str):
        self.name = name
        self.os = os
        self.conda_path = self._ensure_conda_path(conda_path)

    def _ensure_conda_path(self, conda_path: str) -> str:
        try:
            if self.os == "windows":
                proc = self.run(
                    f"if (Test-Path '{conda_path}') {{ exit 0 }} else {{ exit 1 }}"
                )
            else:
                proc = self.run(f"[ -x '{conda_path}' ]")
            proc.communicate(timeout=10)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Conda path '{conda_path}' is not executable on host '{self.name}'."
                )
            return conda_path
        except (subprocess.TimeoutExpired, OSError):
            raise RuntimeError(
                f"Failed to verify conda path '{conda_path}' on host '{self.name}'."
            )

    @abstractmethod
    def run(
        self,
        args: str | list[str],
        stdout: _FILE = subprocess.PIPE,
        stderr: _FILE = subprocess.PIPE,
        log_file: str = "",
        env_vars: dict[str, str] | None = None,
        pid_file: str = "",
    ) -> subprocess.Popen: ...

    def _env_python(self, env: str) -> str:
        conda_dir = os.path.dirname(os.path.dirname(self.conda_path))
        return os.path.join(conda_dir, "envs", env, "bin", "python")

    def run_in_env(
        self,
        env: str,
        args: str | list[str],
        stdout: _FILE = subprocess.PIPE,
        stderr: _FILE = subprocess.PIPE,
        log_file: str = "",
        env_vars: dict[str, str] | None = None,
        pid_file: str = "",
    ) -> subprocess.Popen:
        python = self._env_python(env)
        if isinstance(args, str):
            full_args = [python, "-c", args]
        else:
            args_list = list(args)
            if args_list and args_list[0] in ("python", "python3"):
                args_list = args_list[1:]
            full_args = [python] + args_list
        return self.run(full_args, stdout=stdout, stderr=stderr, log_file=log_file, env_vars=env_vars, pid_file=pid_file)

    def __repr__(self) -> str:
        return f"Host(name={self.name!r}, os={self.os!r})"


class LocalHost(Host):

    os = platform.system().lower()

    def __init__(self, name: str = "localhost", conda_path: str = ""):
        super().__init__(
            name=name,
            os=self.os,
            conda_path=conda_path or os.environ.get("CONDA_EXE", ""),
        )

    def run(
        self,
        args: str | list[str],
        stdout: _FILE = subprocess.PIPE,
        stderr: _FILE = subprocess.PIPE,
        log_file: str = "",
        env_vars: dict[str, str] | None = None,
        pid_file: str = "",
    ) -> subprocess.Popen:
        env = {**os.environ, **(env_vars or {})}
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            f = open(log_file, "w")
            return subprocess.Popen(
                args, shell=isinstance(args, str), stdout=f, stderr=f, env=env
            )
        return subprocess.Popen(
            args,
            shell=isinstance(args, str),
            stdout=stdout,
            stderr=stderr,
            env=env,
        )


class ExternalHost(Host):

    def __init__(
        self,
        name: str,
        ssh_alias: str,
        conda_path: str,
        ssh_tunnel: bool = False,
        router_ip: str | None = None,
    ):
        self.ssh_alias = self._ensure_ssh_alias(ssh_alias)
        self.ssh_tunnel = ssh_tunnel
        self.router_ip = router_ip
        self.router_addr: str | None = None  # set at runtime by Core

        self._os = ""
        if not (self._check_unix() or self._check_windows()):
            raise RuntimeError(
                f"Could not determine OS for host '{name}': "
                "uname and cmd /c ver both failed"
            )
        if self._os not in SUPPORTED_OS:
            raise RuntimeError(
                f"Unsupported OS '{self._os}' for host '{name}'. "
                f"Supported: {SUPPORTED_OS}"
            )

        super().__init__(name=name, os=self._os, conda_path=conda_path)

    def _ensure_ssh_alias(self, ssh_alias: str) -> str:
        known = _ssh_config_aliases()
        if ssh_alias not in known:
            raise ValueError(
                f"SSH alias '{ssh_alias}' not found in ~/.ssh/config. "
                f"Known aliases: {sorted(known)}"
            )
        return ssh_alias

    def _check_unix(self) -> bool:
        stdout, _ = self.run("uname -s").communicate(timeout=10)
        uname = stdout.strip().lower().decode()
        if uname in ("linux", "darwin"):
            self._os = uname
            return True
        return False

    def _check_windows(self) -> bool:
        stdout, _ = self.run("cmd /c ver").communicate(timeout=10)
        if b"windows" in stdout.lower():
            self._os = "windows"
            return True
        return False

    def run(
        self,
        args: str | list[str],
        stdout: _FILE = subprocess.PIPE,
        stderr: _FILE = subprocess.PIPE,
        log_file: str = "",
        env_vars: dict[str, str] | None = None,
        pid_file: str = "",
    ) -> subprocess.Popen:
        cmd = shlex.join(args) if isinstance(args, list) else args
        if env_vars:
            prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_vars.items())
            cmd = f"{prefix} {cmd}"
        if log_file:
            mkdir_cmd = f"mkdir -p {shlex.quote(str(Path(log_file).parent))}"
            if pid_file:
                exec_cmd = f"exec env {cmd}" if env_vars else f"exec {cmd}"
                cmd = f"{mkdir_cmd} && echo $$ > {shlex.quote(pid_file)}; {exec_cmd} > {shlex.quote(log_file)} 2>&1"
            else:
                cmd = f"{mkdir_cmd} && {cmd} > {shlex.quote(log_file)} 2>&1"
            return subprocess.Popen(
                ["ssh", *_SSH_OPTS, self.ssh_alias, cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return subprocess.Popen(
            ["ssh", *_SSH_OPTS, self.ssh_alias, cmd],
            stdout=stdout,
            stderr=stderr,
        )

    def kill_proc(self, pid_file: str) -> None:
        cmd = f"kill $(cat {shlex.quote(pid_file)} 2>/dev/null) 2>/dev/null; rm -f {shlex.quote(pid_file)}"
        subprocess.run(
            ["ssh", *_SSH_OPTS, self.ssh_alias, cmd],
            capture_output=True,
            timeout=10,
        )


def load_hosts(config: dict) -> dict[str, Host]:
    local_spec = config.get("local", {})
    local = LocalHost(
        name=local_spec.get("name", "localhost"),
        conda_path=local_spec.get("conda_path", ""),
    )
    hosts = {local.name: local}

    external_specs = config.get("external", {})
    for name, spec in external_specs.items():
        hosts[name] = ExternalHost(
            name=name,
            ssh_alias=spec["ssh_alias"],
            conda_path=spec["conda_path"],
            ssh_tunnel=spec.get("ssh_tunnel", False),
            router_ip=spec.get("router_ip", None),
        )

    return hosts
