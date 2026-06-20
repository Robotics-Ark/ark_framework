import sys
from loguru import logger as _logger

_logger.remove()


def _patcher(record: dict) -> None:
    record["extra"]["t"] = f"{record['elapsed'].total_seconds():8.3f}s"


def _build_fmt(env: str | None, node: str | None) -> str:
    parts = [
        "<dim>+{extra[t]}</dim>",
        "<level>{level:<8}</level>",
        "<dim>{process}</dim>",
    ]
    if env is not None:
        parts.append("<cyan>{extra[env]:<12}</cyan>")
    if node is not None:
        parts.append("<cyan>{extra[node]:<12}</cyan>")
    parts.append("<dim>{name}:{function}:{line}</dim> - {message}")
    return " | ".join(parts)


def configure_logging(
    level: str = "DEBUG",
    env: str | None = None,
    node: str | None = None,
) -> None:
    _logger.remove()
    extras: dict = {"t": "   0.000s"}
    if env is not None:
        extras["env"] = env
    if node is not None:
        extras["node"] = node
    _logger.configure(extra=extras, patcher=_patcher)
    _logger.add(sys.stderr, format=_build_fmt(env, node), level=level, colorize=True)


configure_logging()

log = _logger
