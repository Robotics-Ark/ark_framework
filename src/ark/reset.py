from __future__ import annotations

import pickle
import threading

import zenoh
from ark.comm import Channel
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ark.time import Clock


def init_namespace(world_name: str) -> Channel:
    """Helper function to construct the base reset channel for a given world name."""
    return Channel.internal(world_name, "reset")


def init_register_channel(ns: Channel) -> Channel:
    """Helper function to construct the registration channel for a given reset namespace."""
    return ns / "register"


def init_initiate_channel(ns: Channel) -> Channel:
    """Helper function to construct the reset initiation channel for a given reset namespace."""
    return ns / "initiate"


def init_completed_channel(ns: Channel) -> Channel:
    """Helper function to construct the reset completion acknowledgement channel for a given reset namespace."""
    return ns / "completed"


class ResetableContainer:
    """A container for managing resetable members, allowing them to register themselves and receive reset initiation messages and provide completion acknowledgements."""

    def __init__(self, world_name: str, session: zenoh.Session, clock: Clock):
        self._session = session
        self._clock = clock
        self._world_name = world_name
        self._ns = init_namespace(self._world_name)
        self._members: set[str] = set()
        self._acks: set[str] = set()
        self._mutex = threading.Lock()

        # Register a queryable that allows resetable members to register themselves and receive a unique name for acknowledgements
        self._reg_channel = init_register_channel(self._ns)
        self._reg = self._session.declare_queryable(
            self._reg_channel, self._on_register
        )

        # Declare a publisher for sending reset initiation messages to members
        self._reset_channel = init_initiate_channel(self._ns)
        self._pub = self._session.declare_publisher(self._reset_channel)

        # Declare a subscriber for receiving reset completion acknowledgements from members
        self._reset_completed_channel = init_completed_channel(self._ns)
        self._sub = self._session.declare_subscriber(
            self._reset_completed_channel, self._on_ack
        )

    def _on_register(self, query: zenoh.Query):
        """Handle a registration query from a resetable member."""

        # Generate a unique name for this member and add it to the set of members
        name = f"resetable_{len(self._members)}"
        with self._mutex:
            self._members.add(name)

        # Reply to the query with the assigned name so the member can use it for acknowledgements
        query.reply(self._reg_channel, name.encode())

    def _initiate_reset(self, seed: dict | int | None = None) -> None:
        """Initiate a reset by clearing acknowledgements and publishing a reset message."""

        # Clear the set of acknowledgements so we can track which members have acknowledged the new reset
        with self._mutex:
            self._acks.clear()

        self._pub.put(pickle.dumps(seed, protocol=pickle.HIGHEST_PROTOCOL))

    def _on_ack(self, sample: zenoh.Sample):
        """Handle a reset completion acknowledgement from a member."""

        # Extract the member name from the sample and validate it
        name = sample.payload.to_string()
        if name not in self._members:
            raise ValueError(
                f"Reset completion acknowledgement from unknown member: {name!r}"
            )
        if name in self._acks:
            raise ValueError(
                f"Duplicate reset completion acknowledgement from member: {name!r}"
            )

        # Add the member name to the set of acknowledgements for the current reset
        with self._mutex:
            self._acks.add(name)

    def reset(self, seed: dict | int | None = None, timeout: float = None):
        """Send reset and block until all current members have acknowledged completion."""

        # Initiate the reset
        self._initiate_reset(seed)

        # Wait until all members have acknowledged the reset, or timeout if not all acks are received within the specified time limit
        if timeout:
            from ark.time import Time

            deadline = self._clock.wall_now() + Time.from_sec(timeout)
        else:
            deadline = None
        while missing := self._members - self._acks:
            if deadline and self._clock.wall_now() >= deadline:
                raise TimeoutError(
                    "Timeout waiting for reset completion "
                    f"acknowledgements from:\n{', '.join(sorted(missing))}"
                )

    def close(self):
        """Clean up zenoh resources used by this container."""
        self._reg.undeclare()
        self._pub.undeclare()
        self._sub.undeclare()


class ResetableObject(ABC):

    _REG_TIMEOUT = 5.0  # seconds

    def __init__(self, world_name: str, session: zenoh.Session):
        self._world_name = world_name
        self._session = session
        self._ns = init_namespace(self._world_name)

        # Register resetable object and get assigned name for acknowledgements
        self._reg_channel = init_register_channel(self._ns)
        self._name = self._register_resetable()
        self._name_enc = self._name.encode()

        # Subscribe to reset initiation messages
        self._reset_channel = init_initiate_channel(self._ns)
        self._initiate_reset_sub = self._session.declare_subscriber(
            self._reset_channel, self._on_reset_sample
        )

        # Declare a publisher for sending reset completion acknowledgements
        self._reset_completed_channel = init_completed_channel(self._ns)
        self._reset_completed_pub = self._session.declare_publisher(
            self._reset_completed_channel
        )

    def _register_resetable(self) -> str:
        """Register this resetable object and return the assigned name for acknowledgements."""
        query = self._session.declare_querier(
            self._reg_channel, timeout=self._REG_TIMEOUT
        )
        replies = query.get()
        if not replies:
            raise TimeoutError(
                f"No reply received from {self._reg_channel} within {self._REG_TIMEOUT}s"
            )
        for reply in replies:
            if reply.ok is None:
                continue
            name = reply.ok.payload.to_string()
            break
        query.undeclare()
        return name

    def _on_reset_sample(self, sample: zenoh.Sample):
        """Handle a reset initiation message by resetting internal state and sending an acknowledgement."""
        payload = bytes(sample.payload)
        seed = pickle.loads(payload) if payload else None
        self.reset(seed)
        self._reset_completed_pub.put(self._name_enc)

    @abstractmethod
    def reset(self, seed: dict | int | None = None):
        """Reset internal state."""

    def close(self):
        """Clean up zenoh resources used by this resetable object."""
        self._initiate_reset_sub.undeclare()
        self._reset_completed_pub.undeclare()


class TestResetableObject(ResetableObject):

    def __init__(self, world_name: str, session: zenoh.Session):
        super().__init__(world_name, session)
        self._reset_count = 0

    def reset(self, seed: dict | int | None = None):
        """Reset internal state by incrementing the reset count."""
        self._reset_count += 1
