from __future__ import annotations


class Channel(str):
    """A channel name that joins segments with `/` and can build child channels with `/`."""

    _separator = "/"
    _internal_root = "_ark"

    def __new__(cls, *parts: str | "Channel"):

        if not parts:
            raise ValueError("ChannelName requires at least one part")

        normalized_parts: list[str] = []
        for part in parts:
            normalized_parts.extend(cls._normalize_part(part))

        if not normalized_parts:
            raise ValueError("ChannelName cannot be empty")

        return str.__new__(cls, cls._separator.join(normalized_parts))

    @classmethod
    def _normalize_part(cls, part: str | "Channel") -> list[str]:
        """Normalize a channel part by stripping separators and splitting into segments. Also validate that the part is a string or Channel."""
        if not isinstance(part, (str, cls)):
            raise TypeError(
                "Channel parts must be strings or Channel instances, "
                f"got {type(part).__name__}"
            )

        text = str(part).strip(cls._separator)
        if not text:
            raise ValueError("Channel parts cannot be empty")

        return [segment for segment in text.split(cls._separator) if segment]

    @classmethod
    def internal(cls, *parts: str | "Channel") -> "Channel":
        """Build a framework-internal channel under the reserved `_ark` root."""
        for part in parts:
            if cls._normalize_part(part)[0] == cls._internal_root:
                raise ValueError(
                    f"Internal channel parts cannot start with "
                    f"{cls._internal_root!r}: {part!r}"
                )
        return cls(cls._internal_root, *parts)

    @classmethod
    def public(cls, *parts: str | "Channel") -> "Channel":
        """Build a public channel that is not under the reserved `_ark` root."""
        for part in parts:
            if cls._normalize_part(part)[0] == cls._internal_root:
                raise ValueError(
                    f"Public channel parts cannot start with "
                    f"{cls._internal_root!r}: {part!r}"
                )
        return cls(*parts)

    def __truediv__(self, other: str | "Channel") -> "Channel":
        """Join this channel with another part using the separator."""
        return type(self)(self, other)

    def joinpath(self, *others: str | "Channel") -> "Channel":
        """Join this channel with multiple other parts using the separator."""
        return type(self)(self, *others)

    @property
    def parts(self) -> tuple[str, ...]:
        """The individual segments of this channel as a tuple."""
        return tuple(str(self).split(self._separator))

    @property
    def is_internal(self) -> bool:
        """Whether this channel lives under Ark's reserved internal root."""
        return self.parts[0] == self._internal_root

    @property
    def parent(self) -> "Channel":
        """The parent channel of this channel."""
        if len(self.parts) < 2:
            raise ValueError("Root channel names do not have a parent")
        return type(self)(*self.parts[:-1])
