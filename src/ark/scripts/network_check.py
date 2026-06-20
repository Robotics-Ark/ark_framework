import subprocess
import sys
import yaml
import argparse
from copy import deepcopy
from pathlib import Path


def _resolve_ssh_hostname(ssh_alias: str) -> str:
    result = subprocess.run(["ssh", "-G", ssh_alias], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("hostname "):
            return line.split()[1]
    return ssh_alias


def _local_ips() -> list[str]:
    result = subprocess.run(["hostname", "-I"], capture_output=True, text=True)
    return result.stdout.strip().split()


def _remote_can_reach(ssh_alias: str, ip: str, port: int) -> bool:
    py = (
        f"import socket,errno as e;"
        f"s=socket.socket();s.settimeout(3);"
        f"err=s.connect_ex(('{ip}',{port}));s.close();"
        f"print('ok' if err in (0,e.ECONNREFUSED) else 'fail')"
    )
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "BatchMode=yes",
                ssh_alias,
                f'python3 -c "{py}"',
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return "ok" in result.stdout
    except subprocess.TimeoutExpired:
        return False


def _check_host(name: str, ssh_alias: str, port: int) -> dict:
    host_ip = _resolve_ssh_hostname(ssh_alias)
    print(f"\n  Host: {name}  ({ssh_alias} → {host_ip})")

    local_ips = _local_ips()
    reachable_ip = None
    for ip in local_ips:
        print(f"    {name} → laptop ({ip}):{port} ... ", end="", flush=True)
        if _remote_can_reach(ssh_alias, ip, port):
            print("reachable")
            reachable_ip = ip
            break
        else:
            print("blocked")

    if reachable_ip:
        print(f"    → direct connection  (router_ip: {reachable_ip})")
        return {"ssh_tunnel": False, "router_ip": reachable_ip}
    else:
        print(f"    → SSH reverse tunnel required")
        return {"ssh_tunnel": True}


def main():
    parser = argparse.ArgumentParser(
        description="Check zenoh network connectivity between ark hosts."
    )
    parser.add_argument("--hosts", type=Path, required=True, help="Path to hosts.yaml")
    parser.add_argument(
        "--port", type=int, default=7447, help="Zenoh router port (default: 7447)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output hosts.yaml path (default: overwrites input)",
    )
    args = parser.parse_args()

    with open(args.hosts) as f:
        config = yaml.safe_load(f)

    external = config.get("external", {})
    if not external:
        print("No external hosts found in hosts.yaml — nothing to check.")
        sys.exit(0)

    msg = "Ark network connectivity check"
    print(msg)
    print("=" * len(msg))

    updated = deepcopy(config)
    for name, spec in external.items():
        result = _check_host(name, spec["ssh_alias"], args.port)
        updated["external"][name].update(result)

    out_path = args.output or args.hosts
    with open(out_path, "w") as f:
        yaml.dump(updated, f, default_flow_style=False, sort_keys=False)

    print(f"\nUpdated config written to: {out_path}")
    print("\nSummary:")
    for name, spec in updated["external"].items():
        mode = (
            "ssh_tunnel"
            if spec.get("ssh_tunnel")
            else f"direct (router_ip: {spec.get('router_ip')})"
        )
        print(f"  {name}: {mode}")


if __name__ == "__main__":
    main()
