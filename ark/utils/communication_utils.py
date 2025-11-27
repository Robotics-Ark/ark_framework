import importlib
from typing import Any, Callable

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from arktypes.utils import unpack, pack
from ark.decoders.registry import get_decoder
from ark.utils.data_utils import generate_flat_dict

from ark.decoders.builtin_decoders import OBS_SCHEMA


def get_ark_fn_type(ark_module: unpack, name: str):
    """
    Retrieve both an unpacking function and its corresponding type from Ark module.
    Args:
        ark_module: The module (e.g., ``arktypes.utils.unpack``) containing the
        unpack functions and optional type definitions.
        name: The base name of the function/type pair to retrieve.

    Returns:
        A tuple (fn, dtype) where:
          - fn is the unpacking function corresponding to ``name``.
          - dtype is the associated type object if defined, otherwise None.

    """
    fn = getattr(ark_module, name)
    dtype = getattr(ark_module, f"{name}_t")
    return fn, dtype


def _resolve_channel_types(mapping: dict[str, Any]) -> dict[str, type]:
    """
    Resolve a mapping of channel names to Ark types.
    Accepts either already-imported classes or string names present in the
    ``arktypes`` package. Returns a mapping of channel name to type.
    Args:
        A dictionary mapping channel names to resolved Ark type objects.

    Returns:

    """
    if not mapping:
        return {}
    resolved: dict[str, type] = {}
    arktypes_mod = importlib.import_module("arktypes")
    for ch_name, t in mapping.items():
        if isinstance(t, str):
            resolved[ch_name] = getattr(arktypes_mod, t)
        else:
            resolved[ch_name] = t
    return resolved


def get_channel_types(schema: dict, channel_type: str | None) -> dict[str, type]:
    """
    Generate a mapping of observation channel names to Python/Ark types
    based on the observation schema.

    Args:
        schema: Observation schema dictionary (from YAML or Python dict).
            Each channel entry can optionally include a 'type' key, which can be
            a string corresponding to a class in `arktypes` or a Python type.
        channel_type: channel type

    Returns:
        Dict[str, type]: Dictionary mapping channel name to resolved type.
    """
    channels: dict[str, Any] = {}

    if channel_type is not None:
        obs_schema = schema.get(channel_type, {})
    else:
        obs_schema = schema

    for key, entries in obs_schema.items():
        for item in entries:
            ch_name = item["from"]
            using = item["using"]
            _, ch_type = get_ark_fn_type(ark_module=unpack, name=using)
            if ch_name not in channels:
                channels[ch_name] = ch_type

    # Resolve type strings to actual type objects using _resolve_channel_types
    resolved_channels = _resolve_channel_types(channels)
    return resolved_channels


def _dynamic_observation_unpacker(schema: dict, namespace: str) -> Callable:
    """
    Create a dynamic observation unpacker based on a schema.

    The schema should be in the format:
    observation:
      state:
        - from: channel_name
          using: callable
      image_top:
        - from: channel_name
          using: callable
          wrap: True  # optional

    Returns a function:
        _unpack(observation_dict) -> dict
    """

    obs_schema = schema["observation_space"]
    obs_schema = namespace_channels(channels=obs_schema, namespace=namespace)

    def _unpack(observation_dict: dict[str, Any]) -> dict[str, Any]:
        if not observation_dict:
            return {}

        result: dict[str, Any] = {}

        for key, entries in obs_schema.items():
            parts = {}
            for item in entries:
                ch_name = item["from"]
                msg = observation_dict.get(ch_name)
                decoder = get_decoder(item["using"])
                decoded = decoder(msg)
                if "name" in item:
                    parts[item["name"]] = decoded
                else:
                    parts[item["using"]] = decoded

            result[key] = parts
        return result

    return _unpack


def _dynamic_action_packer(
    schema: dict, namespace: str
) -> Callable[..., dict[str, Any]]:
    """
    Create a dynamic action packer from schema.

    Returns a function:
        _pack(observation_dict) -> dict

    """

    act_schema = schema["action_space"]
    act_schema = namespace_channels(channels=act_schema, namespace=namespace)

    def _pack(action: list[float] | np.ndarray) -> dict[str, Any]:
        a = np.asarray(action).tolist()
        result: dict[str, Any] = {}

        for key, entries in act_schema.items():
            for item in entries:
                channel = item["from"]
                using = item["using"]
                select = item.get("select", {})

                # resolve packer dynamically
                fn, dtype = get_ark_fn_type(ark_module=pack, name=using)

                # build args from config
                args = []

                for field_name, idx in select.items():
                    if isinstance(idx, list):
                        args.append(np.array([a[i] for i in idx]))
                    elif isinstance(idx, str):
                        args.append(idx)
                    else:
                        args.append(a[idx])
                msg = fn(*args)
                result[channel] = msg
        return result

    return _pack


def build_action_space(schema):
    """
    Build a Gym-style action space from the action-space configuration.

    The expected config format is:

        action_space:
          action:
            - using: task_space_command
              dim: 8

    Which results in a single Box of shape (sum(dim),).
    """
    schema = schema["action_space"]
    proprio_dim = 0
    for items in schema.values():
        for item in items:
            proprio_dim += item["dim"]

    return spaces.Box(low=-1, high=1, shape=(proprio_dim,), dtype=np.float32)


def build_observation_space(schema: dict, flatten_obs_space: bool) -> gym.Space:
    """
    Convert observation_space schema into a Gym Dict space.
    """
    gym_dict = {}
    num_joints = schema["robot"]["num_joints"]
    schema = schema["observation_space"]

    for key, entries in schema.items():
        inner_dict = {}
        for item in entries:
            decoder = item["using"]
            if (
                decoder == "rgbd"
            ):  # check is there any other sensors/camera type is available
                component_dict = {}
                h = item.get("image_height")
                w = item.get("image_width")
                component_dict["rgb"] = gym.spaces.Box(
                    low=0, high=255, shape=(h, w, 3), dtype=np.float32
                )
                # component_dict["depth"] = gym.spaces.Box(
                #     low=0.0, high=5.0, shape=(h, w), dtype=np.float32
                # ) # TODO handle depth image in proper way
            else:
                components = OBS_SCHEMA[decoder]
                component_dict = {}
                for component in components:
                    if decoder == "pose":
                        dim = 3 if component == "position" else 4
                    elif decoder == "rigid_body_state":
                        dim = 4 if component == "orientation" else 3
                    else:
                        dim = num_joints
                    component_dict[component] = gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32
                    )
            if "name" in item:
                inner_dict[item["name"]] = gym.spaces.Dict(component_dict)
            else:
                inner_dict[decoder] = gym.spaces.Dict(component_dict)
        gym_dict[key] = gym.spaces.Dict(inner_dict)

    if flatten_obs_space:
        gym_dict = generate_flat_dict(gym_dict)
    return gym.spaces.Dict(gym_dict)


def namespace_channels(channels: dict, namespace: str):

    prefix = f"{namespace}/"

    # Flat mapping {channel_name: type}
    if all(isinstance(v, type) for v in channels.values()):
        out = {}
        for ch_name, ch_type in channels.items():
            out[f"{prefix}{ch_name}"] = ch_type
        return out

    # Structured schema with lists of dicts containing "from"
    out = {}

    for key, items in channels.items():
        new_items = []
        for entry in items:
            entry = entry.copy()  # avoid modifying original

            if "from" in entry:
                entry["from"] = f"{prefix}{entry['from']}"

            new_items.append(entry)

        out[key] = new_items

    return out
