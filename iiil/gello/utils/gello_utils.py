"""
Gello utilities.

Note: JoyconController has been moved to iiil/utils/joycon_utils.py.
This module provides a backward-compatible alias.
"""

# Backward compatibility - import from new location
from iiil.utils.joycon_utils import A1JoyConInterface as JoyconController

__all__ = ["JoyconController"]
