"""
Shim module to expose RETFoundArgs at RETFound.configure

Allows imports like:
    from RETFound.configure import RETFoundArgs
"""

from util.configure import RETFoundArgs

__all__ = ["RETFoundArgs"]
