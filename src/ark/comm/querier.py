import zenoh
from typing import Any
from gymnasium import Space
from .end_point import EndPoint
from .stamped_sample import StampedSample
from .codec.registry import sample_codec
from .channel import Channel
from ark.noise import NOISE_TYPE, normalise_noise


class Querier(EndPoint):

    def __init__(
        self,
        channel: Channel,
        req_space: Space,
        res_space: Space,
        session: zenoh.Session,
        check_req: bool = False,
        check_res: bool = False,
        req_noise: NOISE_TYPE = None,
        res_noise: NOISE_TYPE = None,
    ):
        super().__init__(channel, session)
        self._req_space = req_space
        self._res_space = res_space
        self._req_codec = sample_codec.get(self._req_space)
        self._res_codec = sample_codec.get(self._res_space)
        self._check_req = check_req
        self._check_res = check_res
        self._req_noises = normalise_noise(req_noise)
        self._res_noises = normalise_noise(res_noise)
        self._qr = self._session.declare_querier(self._channel.full_name)

    def __call__(self, request: Any) -> StampedSample:
        if self._check_req and not self._req_space.contains(request):
            raise ValueError(
                f"Request {request} does not conform to the request space {self._req_space}"
            )
        for noise in self._req_noises:
            request = noise.apply(request)
        for reply in self._qr.get(payload=self._req_codec.encode(request)):
            if reply.ok:
                response = self._res_codec.decode(reply.ok)
                for noise in self._res_noises:
                    response = noise.apply(response)
                if self._check_res and not self._res_space.contains(response):
                    raise ValueError(
                        f"Response {response} does not conform to the response space {self._res_space}"
                    )
                return response
            elif reply.err:
                err = bytes(reply.err).decode("utf-8", errors="replace")
                raise RuntimeError(f"Query on '{self._channel}' failed: {err}")

    def close(self):
        self._qr.undeclare()
