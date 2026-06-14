from __future__ import annotations

from dataclasses import dataclass


class ChannelName(str):
    _separator = "/"

    def __new__(cls, *parts: str | "ChannelName") -> "ChannelName":
        if not parts:
            raise ValueError("ChannelName requires at least one part")

        normalized_parts: list[str] = []
        for part in parts:
            normalized_parts.extend(cls._normalize_part(part))

        if not normalized_parts:
            raise ValueError("ChannelName cannot be empty")

        return str.__new__(cls, cls._separator.join(normalized_parts))

    @classmethod
    def _normalize_part(cls, part: str | "ChannelName") -> list[str]:
        if not isinstance(part, str):
            raise TypeError(
                "ChannelName parts must be strings or ChannelName instances, "
                f"got {type(part).__name__}"
            )

        text = part.strip(cls._separator)
        if not text:
            raise ValueError("ChannelName parts cannot be empty")

        return [segment for segment in text.split(cls._separator) if segment]

    def __truediv__(self, other: str | "ChannelName") -> "ChannelName":
        return type(self)(self, other)

    def joinpath(self, *others: str | "ChannelName") -> "ChannelName":
        return type(self)(self, *others)

    @property
    def parts(self) -> tuple[str, ...]:
        return tuple(self.split(self._separator))


@dataclass(frozen=True, slots=True)
class Channel:
    name: ChannelName
    env_name: str

    @property
    def full_name(self) -> ChannelName:
        return ChannelName(self.env_name) / self.name
