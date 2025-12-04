# Isaac Sim integration (ARK)

Run the ARK Isaac Simulator with the standalone Isaac Sim 5.0.0 build.

## Prerequisites
- Python environment set up for this repository (install ARK deps first).
- NVIDIA driver that meets Isaac Sim requirements.

## Setup & run
1) Download Isaac Sim 5.0.0 standalone  
   `https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.0.0-linux-x86_64.zip`
2) Extract the archive anywhere on your machine.
3) Point ARK to the extracted Isaac Sim root by setting `ARK_ISSAC_PATH` (note: path must point to the root containing `python.sh`/`python.exe` and `kit`):
   - Linux/macOS: `export ARK_ISSAC_PATH=/path/to/isaac-sim-5.0.0-linux-x86_64`
   - Windows (PowerShell): `$env:ARK_ISSAC_PATH="C:\\path\\to\\isaac-sim-5.0.0-windows-x86_64"`
4) From Ark, launch the simulator node from where you have it:  
   `python sim_node.py`

