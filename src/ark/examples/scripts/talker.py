import numpy as np
import gymnasium as gym
from ark.node import Node, main


class Talker(Node):

    def __init__(self, env_name, node_name, parameters, channel_remaps, session):
        super().__init__(env_name, node_name, parameters, channel_remaps, session)
        space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        self._pub = self.create_publisher("chatter", space)
        self._count = 0
        self._max_count = int(self.get_parameter("max_count", 1000))
        self._hello = str(self.get_parameter("hello_word", "hello"))
        self._stepper = self.create_stepper(
            float(self.get_parameter("hz", 1.0)),
            self._step,
        )

    def _step(self, t):
        if self._count >= self._max_count:
            self.stop_spinning()
            return
        self._pub.publish(np.array([float(self._count)]))
        print(f"[talker] {self._hello} count={self._count}", flush=True)
        self._count += 1


if __name__ == "__main__":
    main(Talker)
