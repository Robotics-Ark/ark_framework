# Ark


# Robotics-Ark

[![Release](https://img.shields.io/github/v/release/Robotics-Ark/Robotics-Ark)](https://github.com/Robotics-Ark/Robotics-Ark/releases)
[![License](https://img.shields.io/github/license/Robotics-Ark/Robotics-Ark)](https://github.com/Robotics-Ark/Robotics-Ark/blob/main/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/Robotics-Ark/Robotics-Ark)](https://github.com/Robotics-Ark/Robotics-Ark/commits/main)
[![GitHub stars](https://img.shields.io/github/stars/Robotics-Ark/Robotics-Ark?style=social)](https://github.com/Robotics-Ark/Robotics-Ark/stargazers)

[![Build](https://img.shields.io/github/actions/workflow/status/Robotics-Ark/Robotics-Ark/ci.yml?branch=main)](https://github.com/Robotics-Ark/Robotics-Ark/actions)

---

⭐ Star us on GitHub — your support motivates us a lot! 🙏😊

---

### 🔗 Share this project

[![Share on Twitter](https://img.shields.io/badge/share-twitter-blue?logo=twitter)](https://twitter.com/intent/tweet?text=Check+out+this+cool+project:+https://github.com/Robotics-Ark/Robotics-Ark)
[![Share on Facebook](https://img.shields.io/badge/share-facebook-blue?logo=facebook)](https://www.facebook.com/sharer/sharer.php?u=https://github.com/Robotics-Ark/Robotics-Ark)
[![Share on Telegram](https://img.shields.io/badge/share-telegram-blue?logo=telegram)](https://t.me/share/url?url=https://github.com/Robotics-Ark/Robotics-Ark&text=Check+out+this+cool+project!)

---


 
This is part of the **Ark project** see [website](https://robotics-ark.github.io/ark_robotics.github.io/).


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

## Cite

If you find Ark useful for your work please cite:

```bibtex
 @misc{robotark2025,
      title        = {Ark: An Open-source Python-based Framework for Robot Learning},
      author       = {Magnus Dierking, Christopher E. Mower, Sarthak Das, Huang Helong, Jiacheng Qiu, Cody Reading, 
                      Wei Chen, Huidong Liang, Huang Guowei, Jan Peters, Quan Xingyue, Jun Wang, Haitham Bou-Ammar},
      year         = {2025},
      howpublished = {\url{https://ark_robotics.github.io/}},
      note         = {Technical report}
    }
```
