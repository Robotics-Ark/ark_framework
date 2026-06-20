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


def _parse_per_ns(tokens: list[str], env_names: list[str], default: float) -> dict[str, float]:
    """Parse a list of tokens into a per-env float dict.

    Accepted forms:
      ["2.0"]                          — apply 2.0 to all envs
      ["pybullet_envs:2.0", "mujoco_envs:0.5"]  — per-namespace values
    Unspecified namespaces fall back to the plain default.
    """
    ns_overrides: dict[str, float] = {}
    for token in tokens:
        if ":" in token:
            ns, val = token.split(":", 1)
            ns_overrides[ns.strip()] = float(val)
        else:
            default = float(token)

    return {
        name: next(
            (v for ns, v in ns_overrides.items() if name.startswith(ns)),
            default,
        )
        for name in env_names
    }


def main():
    configure_logging(env="sim_clock")

    parser = argparse.ArgumentParser(
        description="Step simulated time for ark envs. "
        "Use --hz and --speed for global or per-namespace values."
    )
    parser.add_argument("--sim_envs", type=Path, required=True, help="Path to sim_envs.yaml")
    parser.add_argument(
        "--hz", nargs="+", default=["100.0"],
        metavar="[NAMESPACE:]HZ",
        help="Tick rate in Hz (real time). Single value or namespace:rate pairs. Default: 100.",
    )
    parser.add_argument(
        "--speed", nargs="+", default=["1.0"],
        metavar="[NAMESPACE:]SPEED",
        help="Sim-time speed multiplier relative to real time. "
             "> 1 runs faster, < 1 runs slower. Default: 1.0.",
    )
    args = parser.parse_args()

    env_names = _load_env_names(args.sim_envs)
    env_hz = _parse_per_ns(args.hz, env_names, default=100.0)
    env_speed = _parse_per_ns(args.speed, env_names, default=1.0)

    session = default_session()

    clocks = {
        name: SimulatedTime(name, env_speed[name] / env_hz[name], session)
        for name in env_names
    }
    for c in clocks.values():
        c.reset()

    for name in env_names:
        log.info(
            "env '%s'  hz=%.0f  speed=%.2f×  (sim step=%.4fs)"
            % (name, env_hz[name], env_speed[name], env_speed[name] / env_hz[name])
        )

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
