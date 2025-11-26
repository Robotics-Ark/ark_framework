import gymnasium as gym
import torch
from collections.abc import Iterable

def generate_flat_dict(dic, prefix=None):
    """
    Helper function to recursively iterate through dictionary / gym.spaces.Dict @dic and flatten any nested elements,
    such that the result is a flat dictionary mapping keys to values

    Args:
        dic (dict or gym.spaces.Dict): (Potentially nested) dictionary to convert into a flattened dictionary
        prefix (None or str): Prefix to append to the beginning of all strings in the flattened dictionary. None results
            in no prefix being applied

    Returns:
        dict: Flattened version of @dic
    """
    out = dict()
    prefix = "" if prefix is None else f"{prefix}::"
    for k, v in dic.items():
        if isinstance(v, gym.spaces.Dict) or isinstance(v, dict):
            out.update(generate_flat_dict(dic=v, prefix=f"{prefix}{k}"))
        elif isinstance(v, gym.spaces.Tuple) or isinstance(v, tuple):
            for i, vv in enumerate(v):
                # Assume no dicts are nested within tuples
                out[f"{prefix}{k}::{i}"] = vv
        else:
            # Add to out dict
            out[f"{prefix}{k}"] = v

    return out


def generate_compatible_dict(dic):
    """
    Helper function to recursively iterate through dictionary and cast values to necessary types to be compatible with
    Gym spaces -- in particular, the Sequence and Tuple types for th.tensor values in @dic

    Args:
        dic (dict or gym.spaces.Dict): (Potentially nested) dictionary to convert into a flattened dictionary

    Returns:
        dict: Gym-compatible version of @dic
    """
    out = dict()
    for k, v in dic.items():
        if isinstance(v, dict):
            out[k] = generate_compatible_dict(dic=v)
        elif isinstance(v, torch.Tensor) and v.dim() > 1:
            # Map to list of tuples
            out[k] = tuple(tuple(row.tolist()) for row in v)
        elif isinstance(v, Iterable):
            # bounding box modalities give a list of tuples
            out[k] = tuple(v)
        else:
            # Preserve the key-value pair
            out[k] = v

    return out
