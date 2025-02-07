import struct
from typing import List

import json
import numpy as np
from dataclasses import asdict, is_dataclass


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle dataclasses (recursively convert them to dictionaries)
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        
        # Convert NumPy arrays to lists
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Let the base class default method raise the TypeError
        return super().default(obj)


def compress_8_bytes_to_double(byte_data: bytes) -> float:
    """
    Interpret 8 bytes as a double-precision floating point number.
    
    :param byte_data: A bytes object of length 8.
    :return: The double-precision floating point number represented by these bytes.
    :raises ValueError: If the input is not exactly 8 bytes long.
    """
    if len(byte_data) != 8:
        raise ValueError("Input must be exactly 8 bytes long.")
    return struct.unpack('<d', byte_data)[0]


def string_to_doubles(s: str) -> List[float]:
    """
    Convert a general string into a list of doubles, where each double
    is simply an 8-byte representation of a chunk of the string.

    Steps:
      1. Encode the string to bytes (UTF-8 by default).
      2. Split the bytes into 8-byte chunks.
      3. If the last chunk is shorter than 8 bytes, pad with zeros.
      4. Convert each chunk to a double.

    :param s: The input string.
    :return: A list of double-precision floats, each of which encodes 8 bytes
             from the string.
    """
    # Encode string to bytes
    s_bytes = s.encode('utf-8')

    doubles = []
    # Process in 8-byte chunks
    for i in range(0, len(s_bytes), 8):
        chunk = s_bytes[i:i+8]
        # Pad to 8 bytes if needed
        if len(chunk) < 8:
            chunk = chunk.ljust(8, b'\x00')  # pad with zero bytes
        # Convert the 8-byte chunk into a double
        val = struct.unpack('<d', chunk)[0]
        doubles.append(val)

    return doubles


def doubles_to_string(doubles: List[float]) -> str:
    raw_bytes = bytearray()
    for d in doubles:
        # pack the double back into 8 bytes
        chunk = struct.pack('<d', d)
        raw_bytes.extend(chunk)
    # Now raw_bytes contains the original (possibly zero-padded) data,
    # so we can decode it (strip zero padding if needed).
    return raw_bytes.rstrip(b'\x00').decode('utf-8', errors='ignore')
