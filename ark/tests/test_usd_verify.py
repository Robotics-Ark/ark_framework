import os
from pathlib import Path

import pytest

# Allow overriding the USD path via env var for different environments; default to the requested asset.
USD_ASSET_PATH = Path(
    os.environ.get(
        "ARK_USD_PATH",
        "/home/refinath/ark/ark_franka/franka_panda/panda_with_gripper.usd",
    )
)

pxr = pytest.importorskip(
    "pxr",
    reason="USD validation requires the Pixar USD Python bindings (pxr).",
)
from pxr import Usd


@pytest.mark.skipif(
    not USD_ASSET_PATH.exists(),
    reason=(
        f"USD asset not found at {USD_ASSET_PATH}. "
        "Set ARK_USD_PATH to point at a local copy."
    ),
)
def test_usd_file_opens_and_has_valid_prims():
    """Basic sanity check that the USD file loads and contains usable prims."""
    stage = Usd.Stage.Open(str(USD_ASSET_PATH))
    assert stage is not None, f"Failed to open USD stage at {USD_ASSET_PATH}"

    prims = list(stage.Traverse())
    for prim in prims:
        print(prim)
    assert prims, f"USD stage at {USD_ASSET_PATH} has no prims when traversed"

    any_valid_non_root = any(
        prim.IsValid() and str(prim.GetPath()) != "/" for prim in prims
    )
    assert any_valid_non_root or prims[0].IsValid(), "USD stage contains no valid prims"

    default_prim = stage.GetDefaultPrim()
    if default_prim:
        assert default_prim.IsValid(), "Default prim in USD stage is not valid"

if __name__ == "__main__":
    test_usd_file_opens_and_has_valid_prims()
