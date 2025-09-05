# A Python framework for robotics research and development.

<p align="center">
    <a href="https://pepy.tech/project/ark-robotics"> <img src="https://static.pepy.tech/badge/ark-robotics" alt="PyPI Downloads"> </a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="https://github.com/Robotics-Ark/ark_framework/commits/main"><img src="https://img.shields.io/github/last-commit/Robotics-Ark/ark_framework"></a>
  <a href="https://github.com/Robotics-Ark/ark_framework/stargazers"><img src="https://img.shields.io/github/stars/Robotics-Ark/ark_framework?style=social"></a>
</p>


⭐ **Star us on GitHub — your support motivates us a lot!** 🙏😊

### 🔗 Share this project

<p align="center">
  <a href="https://twitter.com/intent/tweet?text=Check+out+ark_framework:+https://github.com/Robotics-Ark/ark_framework">
    <img src="https://img.shields.io/badge/Share%20on-X-black?logo=x&logoColor=white">
  </a>
  <a href="https://www.facebook.com/sharer/sharer.php?u=https://github.com/Robotics-Ark/ark_framework">
    <img src="https://img.shields.io/badge/Share%20on-Facebook-1877F2?logo=facebook&logoColor=white">
  </a>
  <a href="https://t.me/share/url?url=https://github.com/Robotics-Ark/ark_framework&text=Check+out+ark_framework">
    <img src="https://img.shields.io/badge/Share%20on-Telegram-0088CC?logo=telegram&logoColor=white">
  </a>
  <a href="https://www.linkedin.com/sharing/share-offsite/?url=https://github.com/Robotics-Ark/ark_framework">
    <img src="https://img.shields.io/badge/Share%20on-LinkedIn-0A66C2?logo=linkedin&logoColor=white">
  </a>
</p>

 🔥 Why Ark is the way to go in robotics — find out about the [Ark project](https://robotics-ark.github.io/ark_robotics.github.io/).
 
 👨‍🏫 How to get started with Ark step by step — check out the [Tutorials](https://arkrobotics.notion.site/ARK-Home-22be053d9c6f8096bcdbefd6276aba61).
 
 📄 How to navigate our software - read the [Documentation](https://robotics-ark.github.io/ark_robotics.github.io/docs/html/index.html).


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
