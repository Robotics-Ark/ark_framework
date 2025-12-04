import os
import subprocess
import sys
from pathlib import Path


def configure_isaac_setup():
    # Get ARK_ISSAC_PATH from environment
    ark_path = os.environ.get("ARK_ISSAC_PATH")

    # Terminate if not defined or empty
    if not ark_path:
        print(
            "ERROR: ARK_ISSAC_PATH is not defined. Please set it before running the isaac simulator."
        )
        sys.exit(1)

    # Resolve the expected setup script path
    setup_script = Path(ark_path) / "setup_conda_env.sh"

    if not setup_script.is_file():
        print(
            f"ERROR: Could not Configure isaac environment from the provided path: {setup_script}"
        )
        sys.exit(1)

    subprocess.run(f"source {setup_script}", shell=True, executable="/bin/bash", check=True)
