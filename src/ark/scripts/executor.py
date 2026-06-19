import sys
import traceback
import zenoh
import yaml
from pathlib import Path
from ark.executor.host import load_hosts
from ark.executor import Executor
from ark.comm.default_z_session import default_session


def main():
    path_to_hosts_yaml = sys.argv[1]
    with open(path_to_hosts_yaml, "r") as f:
        config = yaml.safe_load(f)
    hosts = load_hosts(config)

    try:
        z_config_path = sys.argv[2]
        z_cfg = zenoh.Config.from_json5(Path(str(z_config_path)).read_text())
        session = zenoh.open(z_cfg)
    except IndexError:
        session = default_session()

    try:
        executor = Executor(hosts, session)
        executor.spin()
    except KeyboardInterrupt:
        pass
    except:
        tb = traceback.format_exc()
        print(f"Executor failed with exception:\n{tb}")
    finally:
        executor.close()


if __name__ == "__main__":
    main()
