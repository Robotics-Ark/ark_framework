# ark

**Ark will be actively updated weekly - we iterate so you dont have to wade through a single monster-commit 😄**

This repository contains the ARK framework for machine learning and robotics.

## Installation

The framework depends on [ARK Types](https://github.com/Robotics-Ark/ark_types) and
requires a Python environment managed with Conda. The steps below describe how
to set up the repositories on **Ubuntu** and **macOS**.

### Ubuntu

```bash
# create a workspace and enter it
mkdir Ark
cd Ark

# create and activate the environment
conda create -n ark_env python=3.10
conda activate ark_env

# clone and install the framework
git clone https://github.com/Robotics-Ark/ark_framework.git
cd ark_framework
pip install -e .
cd ..

# clone and install ark_types
git clone https://github.com/Robotics-Ark/ark_types.git
cd ark_types
pip install -e .
```

### macOS

```bash
# create a workspace and enter it
mkdir Ark
cd Ark

# create and activate the environment
conda create -n ark_env python=3.11
conda activate ark_env

# clone and install the framework
git clone https://github.com/Robotics-Ark/ark_framework.git
cd ark_framework
pip install -e .

# pybullet must be installed via conda on macOS
conda install -c conda-forge pybullet
cd ..

# clone and install ark_types
git clone https://github.com/Robotics-Ark/ark_types.git
cd ark_types
pip install -e .
```

After installation, verify the command-line tool is available:

```bash
ark --help
```


