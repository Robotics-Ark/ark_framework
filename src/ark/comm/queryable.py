import zenoh
from typing import Callable
from gymnasium import Space
from .end_point import EndPoint
from .codec.registry import sample_codec
from .queryable_space import QueryableSpace
from .channel import Channel, ChannelNoise, NoNoise


class Queryable(EndPoint):

    def __init__(
        self,
        channel: Channel,
        req_space: Space,
        res_space: Space,
        callback: Callable,
        session: zenoh.Session,
        check_req: bool = False,
        check_res: bool = False,
        req_noise: ChannelNoise | None = None,
        res_noise: ChannelNoise | None = None,
    ):
        super().__init__(session)
        self._channel = channel
        self._req_space = req_space
        self._res_space = res_space
        self._callback = callback
        self._check_req = check_req
        self._check_res = check_res
        self._req_noise = req_noise or NoNoise()
        self._res_noise = res_noise or NoNoise()
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
                request = self._req_noise.apply(request)
                if self._check_req and not self._req_space.contains(request):
                    raise ValueError(
                        f"Request {request} does not conform to the request space {self._req_space}"
                    )
                response = self._callback(request)
                response = self._res_noise.apply(response)
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
