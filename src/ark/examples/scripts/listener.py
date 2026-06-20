import numpy as np
import gymnasium as gym
from ark.node import Node, main
from ark.comm.stamped_sample import StampedSample


class Listener(Node):

    def __init__(self, env_name, node_name, parameters, channel_remaps, session):
        super().__init__(env_name, node_name, parameters, channel_remaps, session)
        space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        self._sub = self.create_subscriber("chatter", space, self._on_chatter)

    def _on_chatter(self, sample: StampedSample):
        print(f"[listener] received count={sample.sample[0]:.0f}", flush=True)


if __name__ == "__main__":
    main(Listener)
