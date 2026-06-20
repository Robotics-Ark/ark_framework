import os
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
        self._managed_procs = self._start_router(hosts, router_port)
        self._session = load_session(zenoh_config)
        self._sim_env_param_servers = self._init_sim_env_param_servers(sim_envs)
        self._reset_coordinator = ResetCoordinator(self._session)
        self._executor = Executor(hosts, self._session)
        log.info("core initialized")

    def _start_router(self, hosts: dict[str, Host], port: int) -> list[subprocess.Popen]:
        external_hosts = [h for h in hosts.values() if isinstance(h, ExternalHost)]
        if not external_hosts:
            return []
        if shutil.which("zenohd") is None:
            raise RuntimeError("External hosts detected but 'zenohd' not found in PATH.")

        router = subprocess.Popen(
            ["zenohd"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info("started zenoh router (pid %d)" % router.pid)
        time.sleep(1.0)

        os.environ["ARK_ZENOH_ROUTER"] = f"127.0.0.1:{port}"

        procs = [router]
        for host in external_hosts:
            if host.ssh_tunnel:
                tunnel = subprocess.Popen(
                    ["ssh", "-N",
                     "-o", "ExitOnForwardFailure=yes",
                     "-o", "ServerAliveInterval=30",
                     "-R", f"{port}:127.0.0.1:{port}",
                     host.ssh_alias],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                host.router_addr = f"127.0.0.1:{port}"
                log.info("opened reverse tunnel to '%s' on port %d (pid %d)" % (host.name, port, tunnel.pid))
                procs.append(tunnel)
            elif host.router_ip:
                host.router_addr = f"{host.router_ip}:{port}"
                log.info("host '%s' will connect directly via %s" % (host.name, host.router_addr))

        return procs

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

    @staticmethod
    def _stop_proc(proc: subprocess.Popen) -> None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def close(self):
        for ps in self._sim_env_param_servers.values():
            ps.close()
        self._reset_coordinator.close()
        self._executor.close()
        self._session.close()
        for proc in self._managed_procs:
            self._stop_proc(proc)
        log.info("core closed")
