import json
import ntpath
import posixpath
import subprocess
from ark.executor.host import Host


def _env_basename(host_os: str, path: str) -> str:
    if host_os == "windows":
        return ntpath.basename(path)
    return posixpath.basename(path)


class CondaHost:

    def __init__(self, host: Host):
        self._host = host

    def env_exists(self, env_name: str) -> bool:
        if self._host.is_local:
            cmd = ["conda", "env", "list", "--json"]
        else:
            cmd = [
                "ssh",
                self._host.ssh_alias,
                self._host.conda_path,
                "env",
                "list",
                "--json",
            ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )

        payload = json.loads(result.stdout)

        for env_path in payload.get("envs", []):
            if env_name == _env_basename(self._host.os, env_path):
                return True

        return False
