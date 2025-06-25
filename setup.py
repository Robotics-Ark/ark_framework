
from setuptools import setup, find_packages

setup(
    name="ark",
    version="0.0.5",
    packages=find_packages(),
    install_requires=[
        "lcm",
        "colorlog",
        "opencv-python",
        "gymnasium",
        "matplotlib",
        "pandas",
        "numpy==1.24.3",  
        "pybullet",
        "PyYAML",
        "typer"
    ],
    extras_require={
        "test": ["pytest"],
    },
)
