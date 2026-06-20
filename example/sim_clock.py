import time
import yaml
import signal
import argparse
from pathlib import Path
from ark.time import SimulatedTime
from ark.comm.zenoh_session import default_session
from ark.logging import configure_logging, log


def _load_env_names(sim_envs_path: Path) -> list[str]:
    with open(sim_envs_path) as f:
        config = yaml.safe_load(f)
    return [
        f"{ns}_{i}"
        for ns, spec in config.items()
        for i in range(spec["n_envs"])
    ]


def _parse_hz(hz_args: list[str], env_names: list[str]) -> dict[str, float]:
    """Parse --hz args into a per-env hz dict.

    Accepted forms:
      --hz 100               apply 100 Hz to all envs
      --hz pybullet_envs:100 mujoco_envs:50   per-namespace rates
    Unspecified namespaces fall back to the plain default (default: 100.0).
    """
    default_hz = 100.0
    ns_overrides: dict[str, float] = {}

    for token in hz_args:
        if ":" in token:
            ns, rate = token.split(":", 1)
            ns_overrides[ns.strip()] = float(rate)
        else:
            default_hz = float(token)

    return {
        name: next(
            (rate for ns, rate in ns_overrides.items() if name.startswith(ns)),
            default_hz,
        )
        for name in env_names
    }


def main():
    configure_logging(env="sim_clock")

    parser = argparse.ArgumentParser(
        description="Step simulated time for ark envs. "
        "Use --hz for a global rate or per-namespace rates (e.g. pybullet_envs:100 mujoco_envs:50)."
    )
    parser.add_argument("--sim_envs", type=Path, required=True, help="Path to sim_envs.yaml")
    parser.add_argument(
        "--hz", nargs="+", default=["100.0"],
        metavar="[NAMESPACE:]HZ",
        help="Step rate(s) in Hz. Single value applies to all envs; "
             "use namespace:rate pairs for per-env rates.",
    )
    args = parser.parse_args()

    env_names = _load_env_names(args.sim_envs)
    env_hz = _parse_hz(args.hz, env_names)
    session = default_session()

    clocks = {name: SimulatedTime(name, 1.0 / env_hz[name], session) for name in env_names}
    for c in clocks.values():
        c.reset()

    for name in env_names:
        log.info("env '%s' stepping at %.0f Hz" % (name, env_hz[name]))

    now = time.monotonic()
    next_tick = {name: now for name in env_names}
    intervals = {name: 1.0 / env_hz[name] for name in env_names}

    stop = False

    def _shutdown(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while not stop:
        now = time.monotonic()
        for name, clock in clocks.items():
            if now >= next_tick[name]:
                clock.tick()
                next_tick[name] += intervals[name]
        sleep_until = min(next_tick.values())
        remaining = sleep_until - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)

    for c in clocks.values():
        c.close()
    session.close()
    log.info("sim clock stopped")


if __name__ == "__main__":
    main()
