import zenoh
from .end_point import EndPoint
from gymnasium.spaces import Dict as GymDict
from .serialization import Encoder, Decoder


class Query(EndPoint):

    def __init__(self, encoder: Encoder, decoder: Decoder, session: zenoh.Session):
        if encoder.channel != decoder.channel:
            raise ValueError(
                f"Encoder and decoder channels do not match: {encoder.channel} vs {decoder.channel}"
            )
        super().__init__(encoder.channel, session)
        self._encode = encoder
        self._decode = decoder
        self._query_space = GymDict({"request": encoder.space, "reply": decoder.space})
