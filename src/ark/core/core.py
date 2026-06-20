import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from ..logging import log
from ..base import Spinner
from ..executor import Executor
from ..executor.host import Host, ExternalHost
from ..executor.messages import RunNodeRequest, RunReply
from ..parameters import ParameterServer
from ..reset import ResetCoordinator
from ..comm.zenoh_session import load_session
from ..comm.codec.payload import payload_bytes


class Core(Spinner):

    def __init__(
        self,
        hosts: dict[str, Host],
        sim_envs: dict[str, dict],
        zenoh_config: Path | None = None,
        router_port: int = 7447,
        real_env: str | None = None,
    ):
        super().__init__()
        self._managed_procs = self._start_router(hosts, router_port)
        self._session = load_session(zenoh_config)
        self._sim_env_param_servers = self._init_sim_env_param_servers(sim_envs)
        self._real_env_name = real_env
        self._real_env_param_server = (
            ParameterServer(f"{real_env}/parameters", {"sim": False}, self._session, read_only=True)
            if real_env else None
        )
        if real_env:
            log.info("real env '%s' initialized (sim=False)" % real_env)
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

    def _all_env_names(self) -> list[str]:
        names = list(self._sim_env_param_servers.keys())
        if self._real_env_name:
            names.append(self._real_env_name)
        return names

    def launch_nodes(self, nodes_config: dict) -> None:
        qr = self._session.declare_querier("executor/run_node")
        for env_name in self._all_env_names():
            for node_name, spec in nodes_config.items():
                module = spec["main"].split(":")[0]
                req = RunNodeRequest(
                    host=spec["host"],
                    env_name=env_name,
                    conda_env=spec["conda_env"],
                    script=module,
                    node_name=node_name,
                    parameters=spec.get("param", {}),
                    channel_remaps=spec.get("remap", {}),
                    log_file=f"/tmp/{env_name}_{node_name}.log",
                )
                for reply in qr.get(payload=json.dumps(asdict(req)).encode()):
                    if reply.ok:
                        result = RunReply(**json.loads(payload_bytes(reply.ok)))
                        if result.success:
                            log.info("launched node '%s' in env '%s'" % (node_name, env_name))
                        else:
                            log.error("failed to launch '%s' in '%s': %s" % (node_name, env_name, result.error))
        qr.undeclare()

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
        if self._real_env_param_server is not None:
            self._real_env_param_server.close()
        self._reset_coordinator.close()
        self._executor.close()
        self._session.close()
        for proc in self._managed_procs:
            self._stop_proc(proc)
        log.info("core closed")
