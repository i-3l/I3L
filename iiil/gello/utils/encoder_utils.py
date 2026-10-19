"""
Encoder conversion utilities for GELLO system.

For absolute encoders:
- Encoder value 2048 = 180 degrees = 0 radians (home position)
- Encoder range: [0, 4095]
- Conversion: radians = (encoder - 2048) * π / 2048
"""

import numpy as np
from typing import Union


# Constants
ENCODER_CENTER = 2048  # 180 degrees on absolute encoder = zero position
ENCODER_MIN = 0
ENCODER_MAX = 4095
ENCODER_TO_RAD = np.pi / ENCODER_CENTER


def absolute_encoder_to_radians(
    encoder_values: Union[np.ndarray, float, int],
    signs: Union[np.ndarray, float, int] = 1.0,
    offsets: Union[np.ndarray, float, int] = 0.0
) -> np.ndarray:
    """
    Convert absolute encoder readings to joint angles in radians.

    Encoder value 2048 (180°) represents zero position.

    Args:
        encoder_values: Raw encoder values (0-4095)
        signs: Joint direction signs (+1 or -1)
        offsets: Additional joint offsets in radians (default 0)

    Returns:
        Joint angles in radians

    Example:
        >>> absolute_encoder_to_radians(2048)  # Center position
        array([0.])
        >>> absolute_encoder_to_radians(3072)  # +90 degrees
        array([1.5708])
    """
    encoder_values = np.atleast_1d(encoder_values)
    signs = np.atleast_1d(signs)
    offsets = np.atleast_1d(offsets)

    # Convert: (encoder - 2048) * π / 2048 * sign + offset
    joint_angles = (encoder_values - ENCODER_CENTER) * ENCODER_TO_RAD * signs + offsets

    return joint_angles


def radians_to_absolute_encoder(
    joint_angles: Union[np.ndarray, float],
    signs: Union[np.ndarray, float, int] = 1.0,
    offsets: Union[np.ndarray, float, int] = 0.0
) -> np.ndarray:
    """
    Convert joint angles in radians to absolute encoder values.

    Args:
        joint_angles: Joint angles in radians
        signs: Joint direction signs (+1 or -1)
        offsets: Additional joint offsets in radians (default 0)

    Returns:
        Encoder values (integers in range [0, 4095])

    Example:
        >>> radians_to_absolute_encoder(0.0)  # Zero position
        array([2048])
        >>> radians_to_absolute_encoder(np.pi/2)  # +90 degrees
        array([3072])
    """
    joint_angles = np.atleast_1d(joint_angles)
    signs = np.atleast_1d(signs)
    offsets = np.atleast_1d(offsets)

    # Remove offset and convert to encoder
    # encoder = (angle - offset) / sign * 2048 / π + 2048
    encoder_values = ((joint_angles - offsets) / signs / ENCODER_TO_RAD + ENCODER_CENTER)

    # Wrap to valid encoder range [0, 4095] using modular arithmetic.
    # This correctly handles joints where the encoder crosses the 0/4095 boundary
    # during normal operation (e.g. Franka joints with large offsets).
    encoder_values = np.mod(encoder_values.round(), ENCODER_MAX + 1).astype(np.int32)

    return encoder_values


def validate_encoder_range(encoder_values: Union[np.ndarray, int]) -> bool:
    """
    Validate that encoder values are within valid range [0, 4095].

    Args:
        encoder_values: Encoder values to validate

    Returns:
        True if all values are valid, False otherwise
    """
    encoder_values = np.atleast_1d(encoder_values)
    return np.all((encoder_values >= ENCODER_MIN) & (encoder_values <= ENCODER_MAX))


def encoder_difference(
    encoder1: Union[np.ndarray, int],
    encoder2: Union[np.ndarray, int]
) -> np.ndarray:
    """
    Compute the angular difference between two encoder readings.

    Handles wraparound correctly for circular encoders.

    Args:
        encoder1: First encoder reading
        encoder2: Second encoder reading

    Returns:
        Difference in encoder units (can be negative)
    """
    encoder1 = np.atleast_1d(encoder1)
    encoder2 = np.atleast_1d(encoder2)

    diff = encoder2 - encoder1

    # Handle wraparound: if difference > 2048, subtract 4096
    diff = np.where(diff > ENCODER_CENTER, diff - (ENCODER_MAX + 1), diff)
    diff = np.where(diff < -ENCODER_CENTER, diff + (ENCODER_MAX + 1), diff)

    return diff


def get_encoder_home_position() -> int:
    """
    Get the home position encoder value (180 degrees = 2048).

    Returns:
        Home position encoder value
    """
    return ENCODER_CENTER
