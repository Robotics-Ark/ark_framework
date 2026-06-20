import zenoh
import json
import subprocess
import threading
from dataclasses import dataclass, asdict
from ark.executor.host import Host, ExternalHost
from ark.executor.messages import (
    RunRequest,
    RunNodeRequest,
    KillRequest,
    KillEnvRequest,
    ProcessInfo,
    RunReply,
)
from ark.logging import log


@dataclass
class _ProcessEntry:
    proc: subprocess.Popen
    env_name: str
    is_node: bool
    node_name: str | None = None


class Executor:

    def __init__(self, hosts: dict[str, Host], session: zenoh.Session):
        self._hosts = hosts
        self._processes: dict[str, _ProcessEntry] = {}
        self._node_ids: set[tuple[str, str]] = set()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._closed = False
        self._monitor_thread = threading.Thread(target=self._monitor, daemon=True)
        self._monitor_thread.start()
        self._session = session
        self._qr_run = self._session.declare_queryable("executor/run", self._on_run)
        self._qr_run_node = self._session.declare_queryable(
            "executor/run_node", self._on_run_node
        )
        self._qr_list = self._session.declare_queryable("executor/list", self._on_list)
        self._qr_kill = self._session.declare_queryable("executor/kill", self._on_kill)
        self._qr_kill_env = self._session.declare_queryable(
            "executor/kill_env", self._on_kill_env
        )
        log.info(
            "executor initialized with hosts:\n"
            + "\n".join(
                "%d: %s" % (i, str(h))
                for i, h in enumerate(self._hosts.values(), start=1)
            )
        )

    def _check_node_unique(self, env_name: str, node_name: str) -> None:
        nodes_in_env = [n for e, n in self._node_ids if e == env_name]
        if node_name in nodes_in_env:
            raise ValueError(
                f"Node name '{node_name}' already in use in env '{env_name}'. "
                f"Running nodes: {nodes_in_env}"
            )

    def _deregister(self, proc_id: str) -> "_ProcessEntry | None":
        entry = self._processes.pop(proc_id, None)
        if entry is None:
            return None
        if entry.is_node:
            self._node_ids.discard((entry.env_name, entry.node_name))
        return entry

    def _terminate(self, proc: subprocess.Popen) -> None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def _on_run(self, query: zenoh.Query) -> None:
        with query:
            try:
                req = RunRequest.from_json(query.payload)
                host = self._hosts[req.host]
                router = {"ARK_ZENOH_ROUTER": host.router_addr} if isinstance(host, ExternalHost) and host.router_addr else None
                proc = host.run(
                    req.command,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    log_file=req.log_file,
                    env_vars=router,
                )
                proc_id = f"{req.env_name}/{req.id}"
                with self._lock:
                    self._processes[proc_id] = _ProcessEntry(
                        proc=proc, env_name=req.env_name, is_node=False
                    )
                query.reply(query.key_expr, RunReply(success=True).to_json())
                log.info(
                    "Started process '%s' in env '%s' on host '%s'"
                    % (proc_id, req.env_name, req.host)
                )
            except Exception as e:
                query.reply(
                    query.key_expr, RunReply(success=False, error=str(e)).to_json()
                )
                log.error(
                    "Failed to start process in env '%s' on host '%s': %s"
                    % (req.env_name, req.host, str(e))
                )

    def _on_run_node(self, query: zenoh.Query) -> None:
        with query:
            reserved = False
            node_key = None
            try:
                req = RunNodeRequest.from_json(query.payload)
                node_key = (req.env_name, req.node_name)
                proc_id = f"{req.env_name}/{req.node_name}"

                with self._lock:
                    self._check_node_unique(req.env_name, req.node_name)
                    self._node_ids.add(node_key)
                    reserved = True

                args = [
                    "python",
                    "-u",
                    req.script,
                    f"env_name:={req.env_name}",
                    f"node_name:={req.node_name}",
                    *[f"{k}:={v}" for k, v in req.parameters.items()],
                    *[f"{f}--{t}" for f, t in req.channel_remaps.items()],
                ]
                host = self._hosts[req.host]
                router = {"ARK_ZENOH_ROUTER": host.router_addr} if isinstance(host, ExternalHost) and host.router_addr else None
                proc = host.run_in_env(
                    req.conda_env,
                    args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    log_file=req.log_file,
                    env_vars=router,
                )
                with self._lock:
                    self._processes[proc_id] = _ProcessEntry(
                        proc=proc,
                        env_name=req.env_name,
                        is_node=True,
                        node_name=req.node_name,
                    )
                query.reply(query.key_expr, RunReply(success=True).to_json())
                log.info("Started node '%s' on host '%s'" % (proc_id, req.host))

            except Exception as e:
                if reserved:
                    with self._lock:
                        self._node_ids.discard(node_key)
                query.reply(
                    query.key_expr, RunReply(success=False, error=str(e)).to_json()
                )
                log.error("Failed to start node '%s': %s" % (proc_id, str(e)))

    def _on_list(self, query: zenoh.Query) -> None:
        with query:
            with self._lock:
                info = [
                    asdict(
                        ProcessInfo(
                            id=proc_id,
                            env_name=entry.env_name,
                            is_node=entry.is_node,
                            node_name=entry.node_name,
                            running=entry.proc.poll() is None,
                        )
                    )
                    for proc_id, entry in self._processes.items()
                ]
            query.reply(query.key_expr, json.dumps(info).encode())
        log.debug("request to list processes received, %d processes listed" % len(info))

    def _on_kill(self, query: zenoh.Query) -> None:
        with query:
            try:
                req = KillRequest.from_json(query.payload)
                with self._lock:
                    entry = self._deregister(req.id)
                if entry is None:
                    raise ValueError(f"No process with id '{req.id}'")
                self._terminate(entry.proc)
                query.reply(query.key_expr, RunReply(success=True).to_json())
                log.info("Killed process '%s' in env '%s'" % (req.id, entry.env_name))
            except Exception as e:
                query.reply(
                    query.key_expr, RunReply(success=False, error=str(e)).to_json()
                )

    def _on_kill_env(self, query: zenoh.Query) -> None:
        with query:
            try:
                req = KillEnvRequest.from_json(query.payload)
                with self._lock:
                    ids = [
                        pid
                        for pid, e in self._processes.items()
                        if e.env_name == req.env_name
                    ]
                    if not ids:
                        raise ValueError(f"No processes found in env '{req.env_name}'")
                    entries = [self._deregister(pid) for pid in ids]
                for entry in entries:
                    self._terminate(entry.proc)
                query.reply(query.key_expr, RunReply(success=True).to_json())
                log.info("Killed all processes in env '%s'" % req.env_name)
            except Exception as e:
                query.reply(
                    query.key_expr, RunReply(success=False, error=str(e)).to_json()
                )

    def _monitor(self) -> None:
        while not self._stop_event.wait(timeout=1.0):
            with self._lock:
                for proc_id, entry in self._processes.items():
                    if entry.proc.poll() is not None:
                        log.info(
                            "Process '%s' exited (code %d) — shutting down"
                            % (proc_id, entry.proc.returncode)
                        )
                        self._stop_event.set()
                        return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        with self._lock:
            procs = [entry.proc for entry in self._processes.values()]
            self._processes.clear()
            self._node_ids.clear()
        for proc in procs:
            self._terminate(proc)
        self._qr_run.undeclare()
        self._qr_run_node.undeclare()
        self._qr_list.undeclare()
        self._qr_kill.undeclare()
        self._qr_kill_env.undeclare()
