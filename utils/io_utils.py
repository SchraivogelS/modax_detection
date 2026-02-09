#!/usr/bin/env python3

"""
IO utility functions
Author: SCS
Date: 26.07.2021
"""

import json
import numpy as np

from utils.NumpyEncoder import NumpyEncoder


def make_json_safe(obj):
    """Recursively convert unknown objects to JSON-safe types."""
    from dataclasses import asdict, is_dataclass

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if is_dataclass(obj):
        return make_json_safe(asdict(obj))
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return make_json_safe(obj.to_dict())
    if hasattr(obj, "__dict__"):
        # last-resort: dump public attributes
        return make_json_safe({k: v for k, v in obj.__dict__.items() if not k.startswith("_")})
    # ultimate fallback
    return str(obj)


def save_np_values_to_json(values: dict, filepath_out: str, indent: int = 4):
    """
    Save dict containing numpy values to json file

    :param values: Values to save
    :param filepath_out: Save to filepath
    :param indent: json indent
    """
    import warnings

    try:
        with open(filepath_out, "w") as outfile:
            json.dump(values, outfile, cls=NumpyEncoder, indent=indent)
            print(f'Saved values to {filepath_out}')
    except TypeError as e:
        warnings.warn(f"JSON serialization failed: {e}. Saving fallback representation.")
        # Fallback: stringify non-serializable objects
        safe_values = make_json_safe(values)
        with open(filepath_out, "w", encoding="utf-8") as outfile:
            json.dump(safe_values, outfile, indent=indent)
        print(f"Saved fallback JSON to {filepath_out}")


def load_values_from_json(filepath_in, asarray=False):
    from json import JSONDecodeError

    try:
        with open(filepath_in, mode='r', encoding='utf8') as f:
            val = json.load(f)
    except JSONDecodeError as e:
        raise ValueError(f'{type(e).__name__} at {filepath_in}: {e}')
    if asarray:
        val = np.asarray(val)
    return val
