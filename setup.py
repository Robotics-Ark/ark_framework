import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop

try:
    from setuptools.command.editable_wheel import editable_wheel as _editable_wheel
except ImportError:
    _editable_wheel = None


PROTO_ROOT = Path("proto/_msgs")
PY_ROOT = Path("src/ark/_msgs")


def clean_generated_protos() -> None:
    for pattern in ("*_pb2.py", "*_pb2.pyi", "*_pb2_grpc.py", "*_pb2_grpc.pyi"):
        for generated_file in PY_ROOT.rglob(pattern):
            generated_file.unlink()


def compile_protos() -> None:
    protos = sorted(PROTO_ROOT.rglob("*.proto"))
    if not protos:
        return

    PY_ROOT.mkdir(parents=True, exist_ok=True)
    clean_generated_protos()

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{PROTO_ROOT}",
        f"--python_out={PY_ROOT}",
        f"--pyi_out={PY_ROOT}",
        *[str(proto.relative_to(PROTO_ROOT)) for proto in protos],
    ]
    subprocess.check_call(cmd)


class build_py(_build_py):
    def run(self):
        compile_protos()
        super().run()


class develop(_develop):
    def run(self):
        compile_protos()
        super().run()


cmdclass = {"build_py": build_py, "develop": develop}

if _editable_wheel is not None:

    class editable_wheel(_editable_wheel):
        def run(self):
            compile_protos()
            super().run()

    cmdclass["editable_wheel"] = editable_wheel


setup(cmdclass=cmdclass)
