from src.block import Block
from ctypes import Structure, c_int16, c_int32, c_float
import struct
from typing import Tuple, Optional, List
from enum import Enum

class SerializedColor(Enum):
    YELLOW = 1
    RED = 2
    BLUE = 4

def get_serialized_color(color_name: str) -> Optional[SerializedColor]:
    """Maps detected color names to their corresponding SerializedColor enum based on keyword matching."""
    color_name_upper = color_name.upper()

    for enum_member in SerializedColor:
        if enum_member.name in color_name_upper:
            return enum_member

    return None  # Return None if no match is found

class SerializedBlock(Structure):
    _pack_ = 1  # No padding
    _fields_ = [
        ("center_x", c_int16),
        ("center_y", c_int16),
        ("width", c_int16),
        ("height", c_int16),
        ("angle", c_float),
        ("color", c_int32)
    ]

    def __str__(self) -> str:
        return f"SerializedBlock(center=({self.center_x}, {self.center_y}), size=({self.width}, {self.height}), angle={self.angle}, color={self.color})"

    def pack_to_floats(self) -> Tuple[float, float, float, float]:
        """Packs the structure into four floats."""
        raw_bytes = bytes(self)
        assert len(raw_bytes) == 16, f"Expected 16 bytes, got {len(raw_bytes)}"
        f1, f2, f3, f4 = struct.unpack('<4f', raw_bytes)
        return f1, f2, f3, f4

    @staticmethod
    def from_block(block: 'Block') -> 'SerializedBlock':
        """Creates a SerializedBlock from a Block object."""
        color_enum = get_serialized_color(block.color.name)
        assert color_enum is not None, f"Unknown color: {block.color}"
        return SerializedBlock(
            center_x=int(block.center[0]),  # Ensure int conversion
            center_y=int(block.center[1]),
            width=int(block.size[0]),
            height=int(block.size[1]),
            angle=float(block.angle),  # Ensure float conversion
            color=color_enum.value  # Store enum as int
        )

    @classmethod
    def from_bytes(cls, raw_bytes: bytes) -> 'SerializedBlock':
        """Creates a SerializedBlock from raw bytes."""
        assert len(raw_bytes) == 16, f"Expected 16 bytes, got {len(raw_bytes)}"
        return cls.from_buffer_copy(raw_bytes)

    @classmethod
    def from_floats(cls, f1: float, f2: float, f3: float, f4: float) -> 'SerializedBlock':
        """Creates a SerializedBlock from four floats."""
        raw_bytes = struct.pack('<4f', f1, f2, f3, f4)
        return cls.from_bytes(raw_bytes)

def serialize_to_floats(blocks: List['Block']) -> List[float]:
    """Serializes a list of Blocks into a list of floats."""
    serialized_blocks = [SerializedBlock.from_block(block) for block in blocks]
    return [f for serialized_block in serialized_blocks for f in serialized_block.pack_to_floats()]

def deserialize_from_floats(floats: List[float]) -> List[SerializedBlock]:
    """Deserializes a list of floats into a list of SerializedBlocks."""
    assert len(floats) % 4 == 0, f"Expected a multiple of 4 floats, got {len(floats)}"
    serialized_blocks = [SerializedBlock.from_floats(floats[i], floats[i+1], floats[i+2], floats[i+3]) for i in range(0, len(floats), 4)]
    return serialized_blocks
