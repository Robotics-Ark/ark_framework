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
        start_time = time.time()
        while True:
            sim_time.tick()
            elapsed = time.time() - start_time
            sim_elapsed = sim_time._sim_time_ns / 1e9
            print(f"Real: {elapsed:.2f} s | Sim: {sim_elapsed:.3f} s")


if __name__ == "__main__":
    main()
