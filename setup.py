from setuptools import setup, find_packages
import platform

# Optional: check for specific distro like Ubuntu
try:
    import distro
    is_ubuntu = distro.id() == "ubuntu"
except ImportError:
    is_ubuntu = False  # Safe fallback if 'distro' is not installed

# Basic dependencies
install_requires = [
    "colorlog",
    "opencv-python",
    "gymnasium",
    "matplotlib",
    "pandas",
    "numpy==1.24.3",
    # "pybullet",
    "PyYAML",
    "typer"
]

# # Add lcm only on Ubuntu
# if platform.system() == "Linux" and is_ubuntu:
#     install_requires.append("lcm")

setup(
    name="ark",
    version="0.0.5",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={
        "test": ["pytest"],
    },
)
