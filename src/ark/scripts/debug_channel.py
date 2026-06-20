import time
import signal
import argparse
import threading
import collections

from ark.comm.zenoh_session import default_session
from ark.comm.queryable_space import query_space
from ark.comm.codec.registry import sample_codec, space_codec


def cmd_hz(session, channel: str, window: float) -> None:
    timestamps: collections.deque = collections.deque()
    lock = threading.Lock()

    def on_sample(_):
        with lock:
            timestamps.append(time.monotonic())

    sub = session.declare_subscriber(channel, on_sample)
    print(f"Subscribing to '{channel}' — Ctrl-C to stop\n")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    try:
        while not stop.wait(1.0):
            now = time.monotonic()
            cutoff = now - window
            with lock:
                while timestamps and timestamps[0] < cutoff:
                    timestamps.popleft()
                n = len(timestamps)
                span = (timestamps[-1] - timestamps[0]) if n >= 2 else 0.0
            if n >= 2:
                hz = (n - 1) / span
                print(
                    f"  rate: {hz:.2f} Hz  ({n} msgs in last {window:.0f}s window)",
                    flush=True,
                )
            else:
                print(f"  waiting for messages...", flush=True)
    finally:
        sub.undeclare()


_ROLE_SUFFIXES = [
    ("/publisher/get_space", "pub"),
    ("/subscriber/get_space", "sub"),
    ("/request/get_space", "queryable"),
]

# Known internal channel patterns and their roles (regex suffix → roles list).
# Used to annotate _ark/** channels discovered via wildcard subscription or
# to show static infrastructure channels even when no messages are flowing.
import re as _re

_INTERNAL_KNOWN: list[tuple[str, list[str]]] = [
    # executor
    (r"^_ark/executor/run$", ["queryable"]),
    (r"^_ark/executor/run_node$", ["queryable"]),
    (r"^_ark/executor/list$", ["queryable"]),
    (r"^_ark/executor/kill$", ["queryable"]),
    (r"^_ark/executor/kill_env$", ["queryable"]),
    # reset coordinator (global)
    (r"^_ark/reset/register_reset_object$", ["queryable"]),
    (r"^_ark/reset/initiate_reset$", ["queryable"]),
    # reset per-env
    (r"^_ark/reset/[^/]+/initiate_reset$", ["pub", "sub"]),
    (r"^_ark/reset/[^/]+/reset_completed$", ["pub", "sub"]),
    # simulated time
    (r"^_ark/[^/]+/time$", ["pub", "sub"]),
    (r"^_ark/[^/]+/sim$", ["sub"]),
]

# All static infrastructure channels (always present when core is running).
_ARK_STATIC = [
    "_ark/executor/run",
    "_ark/executor/run_node",
    "_ark/executor/list",
    "_ark/executor/kill",
    "_ark/executor/kill_env",
]


def _roles_for_internal(key: str) -> list[str]:
    for pattern, roles in _INTERNAL_KNOWN:
        if _re.match(pattern, key):
            return roles
    return []


def _query_roles(session) -> dict[str, dict]:
    """Return {channel_full: {roles: set[str], space: str}} from QueryableSpace registrations."""
    channels: dict[str, dict] = {}
    for suffix, role in _ROLE_SUFFIXES:
        qr = session.declare_querier(f"**{suffix}")
        for reply in qr.get():
            if reply.ok is None:
                continue
            key = str(reply.ok.key_expr)
            if not key.endswith(suffix):
                continue
            ch = key[: -len(suffix)]
            entry = channels.setdefault(ch, {"roles": set(), "space": ""})
            entry["roles"].add(role)
            if not entry["space"]:
                try:
                    space = space_codec.get().decode(bytes(reply.ok.payload))
                    entry["space"] = str(space)
                except Exception:
                    pass
        qr.undeclare()
    return channels


def cmd_ls(session, include_internal: bool, env_name: str | None = None) -> None:
    print("Discovering channels...\n")
    channels = _query_roles(session)

    # {key: set[str]} for internal channels
    internal: dict[str, set[str]] = {}

    if include_internal:
        # Capture active publishers via 2s wildcard subscription
        print("Listening for active publishers (2s)...", end="", flush=True)
        active_keys: set[str] = set()

        def _on_msg(s):
            active_keys.add(str(s.key_expr))

        sub = session.declare_subscriber("**", _on_msg)
        time.sleep(2.0)
        sub.undeclare()
        print(" done\n")

        # Keep only _ark/** keys not already in user channels
        for k in active_keys:
            if k.startswith("_ark/") and k not in channels and "/get_space" not in k:
                roles = _roles_for_internal(k)
                internal[k] = set(roles) if roles else {"pub"}

        # Confirm executor is running by querying the read-only list endpoint.
        # If it responds, all static infrastructure channels are active.
        executor_active = False
        try:
            qr = session.declare_querier("_ark/executor/list")
            for reply in qr.get():
                if reply.ok is not None:
                    executor_active = True
                    break
            qr.undeclare()
        except Exception:
            pass

        if executor_active:
            for k in _ARK_STATIC:
                if k not in internal:
                    internal[k] = set(_roles_for_internal(k))

    # Filter by env_name if requested
    if env_name:
        channels = {k: v for k, v in channels.items() if k.startswith(f"{env_name}/")}

    if not channels and not internal:
        print("No active channels found.")
        return

    # Group user channels by env_name (first path segment)
    grouped: dict[str, list] = {}
    for ch_full, info in sorted(channels.items()):
        grp, _, ch_name = ch_full.partition("/")
        roles_str = ", ".join(sorted(info["roles"]))
        grouped.setdefault(grp, []).append(
            (ch_name or ch_full, roles_str, info["space"])
        )

    for grp in sorted(grouped):
        print(f"{grp}/")
        for ch_name, roles, space in sorted(grouped[grp]):
            print(f"  {ch_name:<30}  [{roles}]  {space}")
        print()

    if internal:
        # Group by second segment (strip _ark/ prefix for display)
        int_grouped: dict[str, list[tuple[str, str]]] = {}
        for k in sorted(internal):
            # k is e.g. "_ark/executor/run" or "_ark/reset/env1/foo"
            without_prefix = k[len("_ark/") :]  # "executor/run"
            segment, _, rest = without_prefix.partition("/")
            roles_str = ", ".join(sorted(internal[k]))
            int_grouped.setdefault(segment, []).append(
                (rest or without_prefix, roles_str)
            )
        print("_ark/")
        for segment in sorted(int_grouped):
            print(f"  {segment}/")
            for ch, roles_str in sorted(int_grouped[segment]):
                print(f"    {ch:<34}  [{roles_str}]")
        print()


def cmd_echo(session, channel: str) -> None:
    try:
        space = query_space(channel, "publisher", session)
        codec = sample_codec.get(space)
        decode = codec.decode
        print(f"Subscribing to '{channel}'  space={space}  — Ctrl-C to stop\n")
    except Exception as e:
        decode = bytes
        print(
            f"Subscribing to '{channel}'  (raw bytes — publisher space not found: {e})\n"
        )

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    def on_sample(sample):
        data = decode(sample.payload)
        print(f"  {data!r}", flush=True)

    sub = session.declare_subscriber(channel, on_sample)
    try:
        stop.wait()
    finally:
        sub.undeclare()


def main():
    parser = argparse.ArgumentParser(
        prog="ark-debug-channel",
        description="Inspect ark channels at runtime.",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    ls_p = subs.add_parser("ls", help="List all active channels grouped by env.")
    ls_p.add_argument(
        "--include-internal",
        action="store_true",
        help="Also show internal channels (listens for active publishers for 2s).",
    )
    ls_p.add_argument(
        "--env_name",
        default=None,
        metavar="ENV",
        help="Filter to channels in a specific env, e.g. pybullet_envs_0.",
    )

    hz_p = subs.add_parser("hz", help="Print the publish rate of a channel.")
    hz_p.add_argument(
        "channel", help="Full channel path, e.g. pybullet_envs_0/mychatter"
    )
    hz_p.add_argument(
        "--window",
        type=float,
        default=5.0,
        help="Sliding average window in seconds (default: 5)",
    )

    echo_p = subs.add_parser("echo", help="Print messages received on a channel.")
    echo_p.add_argument(
        "channel", help="Full channel path, e.g. pybullet_envs_0/mychatter"
    )

    args = parser.parse_args()
    session = default_session()

    try:
        if args.command == "ls":
            cmd_ls(
                session, include_internal=args.include_internal, env_name=args.env_name
            )
        elif args.command == "hz":
            cmd_hz(session, args.channel, args.window)
        else:
            cmd_echo(session, args.channel)
    finally:
        session.close()


if __name__ == "__main__":
    main()
