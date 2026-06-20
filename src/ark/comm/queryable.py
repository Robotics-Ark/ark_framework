import zenoh
from typing import Callable
from gymnasium import Space
from .end_point import EndPoint
from .codec.registry import sample_codec
from .queryable_space import QueryableSpace
from .channel import Channel
from ark.noise import NOISE_TYPE, normalise_noise


class Queryable(EndPoint):

    def __init__(
        self,
        channel: Channel,
        req_space: Space,
        res_space: Space,
        callback: Callable,
        session: zenoh.Session,
        check_req: bool,
        check_res: bool,
        req_noise: NOISE_TYPE = None,
        res_noise: NOISE_TYPE = None,
    ):
        super().__init__(channel, session)
        self._req_space = req_space
        self._res_space = res_space
        self._callback = callback
        self._check_req = check_req
        self._check_res = check_res
        self._req_noises = normalise_noise(req_noise)
        self._res_noises = normalise_noise(res_noise)
        self._req_codec = sample_codec.get(self._req_space)
        self._res_codec = sample_codec.get(self._res_space)
        self._qr = self._session.declare_queryable(
            self._channel.full_name, self._on_query
        )
        self._qr_req = QueryableSpace(
            channel, "request", self._req_space, self._session
        )
        self._qr_res = QueryableSpace(
            channel, "response", self._res_space, self._session
        )

    def _on_query(self, query: zenoh.Query) -> None:
        with query:
            try:
                request = self._req_codec.decode(query.payload)
                for noise in self._req_noises:
                    request = noise.apply(request)
                if self._check_req and not self._req_space.contains(request):
                    raise ValueError(
                        f"Request {request} does not conform to the request space {self._req_space}"
                    )
                response = self._callback(request)
                for noise in self._res_noises:
                    response = noise.apply(response)
                if self._check_res and not self._res_space.contains(response):
                    raise ValueError(
                        f"Response {response} does not conform to the response space {self._res_space}"
                    )
                query.reply(query.key_expr, self._res_codec.encode(response))
            except Exception as e:
                err = f"Error processing query on '{self._channel}': {e}"
                query.reply_err(err.encode("utf-8"))

    def close(self):
        self._qr.undeclare()
        self._qr_req.undeclare()
        self._qr_res.undeclare()
