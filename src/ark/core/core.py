import os
import socket
import shutil
import subprocess
import time
from pathlib import Path
from ..logging import log
from ..base import Spinner
from ..executor import Executor
from ..executor.host import Host, ExternalHost
from ..parameters import ParameterServer
from ..reset import ResetCoordinator
from ..comm.zenoh_session import load_session


class Core(Spinner):

    def __init__(
        self,
        hosts: dict[str, Host],
        sim_envs: dict[str, dict],
        zenoh_config: Path | None = None,
        router_port: int = 7447,
    ):
        super().__init__()
        self._router_proc, router_env = self._start_router(hosts, router_port)
        self._session = load_session(zenoh_config)
        self._sim_env_param_servers = self._init_sim_env_param_servers(sim_envs)
        self._reset_coordinator = ResetCoordinator(self._session)
        self._executor = Executor(hosts, self._session, env_vars=router_env)
        log.info("core initialized")

    def _start_router(
        self, hosts: dict[str, Host], port: int
    ) -> tuple[subprocess.Popen | None, dict[str, str]]:
        if not any(isinstance(h, ExternalHost) for h in hosts.values()):
            return None, {}
        if shutil.which("zenohd") is None:
            raise RuntimeError(
                "External hosts detected but 'zenohd' not found in PATH."
            )
        ip = self._local_ip()
        router_addr = f"{ip}:{port}"
        os.environ["ARK_ZENOH_ROUTER"] = router_addr
        proc = subprocess.Popen(
            ["zenohd"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info("started zenoh router at %s (pid %d)" % (router_addr, proc.pid))
        time.sleep(1.0)  # wait for the router to start
        return proc, {"ARK_ZENOH_ROUTER": router_addr}

    @staticmethod
    def _local_ip() -> str:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]

    def _init_sim_env_param_servers(
        self, sim_envs: dict[str, dict]
    ) -> dict[str, ParameterServer]:

        init_ps = lambda en, s: ParameterServer(
            f"{en}/parameters",
            {"sim": True, "simulator": s},
            self._session,
            read_only=True,
        )

        ps = {}
        for env_ns, env_config in sim_envs.items():
            for i in range(env_config["n_envs"]):
                env_name = f"{env_ns}_{i}"
                ps[env_name] = init_ps(env_name, env_config["simulator"])

        return ps

    def close(self):
        for ps in self._sim_env_param_servers.values():
            ps.close()
        self._reset_coordinator.close()
        self._executor.close()
        self._session.close()
        if self._router_proc is not None:
            self._router_proc.terminate()
            try:
                self._router_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._router_proc.kill()
        log.info("core closed")
