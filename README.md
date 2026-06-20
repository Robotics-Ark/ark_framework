<p align="center">
  <img src="docs/logo.png" alt="Ark Framework Logo" width="1500">
</p>

<h1 align="center">A python framework for robotics research and development.</h1>

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

1. Create and activate a Conda environment. Python 3.12 is recommended.

   ```bash
   conda create -n ark python=3.12
   conda activate ark
   python -m pip install --upgrade pip
   ```

2. Create an Ark workspace, then clone Ark as `ark_framework`.

   ```bash
   mkdir -p path/to/ark
   cd path/to/ark
   git clone <ark-repository-url> ark_framework
   cd ark_framework
   ```

   Related repositories should be kept as siblings:

   ```text
   path/to/ark/
   ├── ark_framework/
   ├── zenoh/
   └── other-repositories/
   ```

3. Install Zenoh:

   - To use Ark on one PC, follow the [single-PC Zenoh installation](#single-pc-zenoh-installation).
   - To use Ark across multiple PCs, follow the [multi-PC Zenoh installation](#multi-pc-zenoh-installation).

4. Install [Graphviz](https://graphviz.org/download/). This optional dependency is used to generate graph images in several modules. On Ubuntu:

   ```bash
   sudo apt install graphviz
   ```

5. Install Ark in editable mode.

   ```bash
   python -m pip install -e .
   ```

## Single-PC Zenoh installation

Install a Rust toolchain using [rustup](https://rustup.rs/), then build and install the Zenoh Python bindings with shared-memory support:

```bash
python -m pip install \
  --no-binary eclipse-zenoh \
  --config-settings 'build-args=--features=zenoh/shared-memory' \
  'eclipse-zenoh==1.7.2'
```

## Multi-PC Zenoh installation

Complete the [single-PC installation](#single-pc-zenoh-installation) on every participating PC. Then clone the matching Zenoh router source alongside `ark_framework` and install `zenohd` into the active Conda environment:

```bash
cd path/to/ark
git clone --depth 1 --branch release/1.7.2 \
  https://github.com/eclipse-zenoh/zenoh.git zenoh
cargo install \
  --locked \
  --path zenoh/zenohd \
  --features shared-memory \
  --root "$CONDA_PREFIX"
zenohd --version
cd ark_framework
```

The `eclipse-zenoh` Python package does not include `zenohd`, which is why this additional installation is required for multi-PC deployments.

## Multi-PC network setup

After installing zenohd on every PC, run the network check tool from the machine that will run `ark core`. It tests connectivity between your laptop and each external host and writes the recommended configuration back into your `hosts.yaml`:

```bash
ark-network-check --hosts path/to/hosts.yaml
```

Example output:

```
Ark network connectivity check
========================================

  Host: mycluster  (cluster → 135.84.176.142)
    mycluster → laptop (192.168.1.10):7447 ... reachable
    → direct connection  (router_ip: 192.168.1.10)

  Host: gpu_node  (gpu → 135.84.176.200)
    gpu_node → laptop (192.168.1.10):7447 ... blocked
    → SSH reverse tunnel required

Updated config written to: path/to/hosts.yaml

Summary:
  mycluster: direct (router_ip: 192.168.1.10)
  gpu_node: ssh_tunnel
```

The tool adds `ssh_tunnel` and `router_ip` fields to each external host entry. The `ark core` command reads these automatically — no further configuration needed.

**How connectivity is handled at runtime:**

| Scenario | Method | Requires |
|---|---|---|
| Host can reach laptop on port 7447 | Direct TCP | Open port 7447 on laptop |
| Host cannot reach laptop | SSH reverse tunnel | Only SSH (port 22) |

`ark core` starts `zenohd` on the local machine and, for each host requiring a tunnel, opens a reverse SSH connection that forwards port 7447 on the remote host back to the local `zenohd`. All communication flows through port 22.
