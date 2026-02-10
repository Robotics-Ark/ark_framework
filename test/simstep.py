from ark.time.simtime import SimTime
from common import z_cfg
import json
import zenoh
import time

def main():
    z_config = zenoh.Config.from_json5(json.dumps(z_cfg))
    with zenoh.open(z_config) as z:
        sim_time = SimTime(z, "clock", 1000)
        sim_time.reset()
        while True:
            current_time = time.time()
            print(f"Simulated Time: {current_time:.2f} seconds")
            sim_time.tick()

if __name__ == "__main__":
    main()
