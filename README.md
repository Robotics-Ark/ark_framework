<p align="center">
  <img src="Logo.png" alt="Ark Framework Logo" width="1500">
</p>

<h1 align="center">A Python framework for robotics research and development.</h1>

<p align="center">
  <em>Lightweight, flexible, and designed for researchers and developers in robot learning.</em>
</p>

<p align="center">
  <a href="https://pepy.tech/project/ark-robotics">
    <img src="https://static.pepy.tech/badge/ark-robotics" alt="PyPI Downloads">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-green.svg">
  </a>
  <a href="https://github.com/Robotics-Ark/ark_framework/commits/main">
    <img src="https://img.shields.io/github/last-commit/Robotics-Ark/ark_framework">
  </a>
  <a href="https://github.com/Robotics-Ark/ark_framework/stargazers">
    <img src="https://img.shields.io/github/stars/Robotics-Ark/ark_framework?style=social">
  </a>
</p>

<p align="center">
  ⭐ <b>Star us on GitHub — your support motivates us a lot!</b> 🙏😊
</p>

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

<p align="center">
  Join us on Discord!
</p>
<p align="center">
 <a target="_blank" href="https://discord.gg/Mj9HPrUYcf"><img src="https://dcbadge.limes.pink/api/server/zkspfFwqDg" alt="" /></a>
</p>

## What is this about? 

Ark is a Python-first playground for robot learning. 
Instead of wrestling with C++ and fragmented tools, you can collect data, train policies, and switch between simulation and real robots with just a few lines of code. 
Think of it as the PyTorch + Gym for robotics — simple, modular, and built for rapid prototyping of intelligent robots.

📚 **Learn more:**  
- [📖 Tutorials]()  
- [⚙️ Documentation]()  
- [📄 Research Paper]()

# Installation

1. Create and activate a conda environment.
  - Python 3.12 is recommended.
  - E.g, `conda create -n ark python=3.12`
2. Clone this repository and change directory `cd ark_framework`.
3. Install [zenoh-python](https://github.com/eclipse-zenoh/zenoh-python)
   - Installation instructions are found [here](https://github.com/eclipse-zenoh/zenoh-python#how-to-install-it).
   - It is recommended to [enable zenoh features](https://github.com/eclipse-zenoh/zenoh-python#enable-zenoh-features).
4. Install: `pip install -e .`

## Cite

```bibtex
@misc{robotark2025,
  title = {An Open-source Python-based Framework for Embodied AI},
  author = {Magnus Dierking, Christopher E. Mower, Refinath S N, Abhineet Kumar, Huang Helong, Jiacheng Qiu, Wei Chen, Huidong Liang, Huang Guowei, Jan Peters, Quan Xingyue, Jun Wang, Haitham Bou-Ammar},
  year = {2025},
  howpublished = {\url{https://robotics-ark.github.io/ark_robotics.github.io/}},
  note = {Technical report}
}
```
