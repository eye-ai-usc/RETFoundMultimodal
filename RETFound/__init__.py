
# RETFound/__init__.py
from importlib import import_module
import sys

# Public types
from .configure import RETFoundArgs

# Import top-level modules so we can expose them as RETFound submodules
import models_vit
import engine_finetune
from engine_finetune import train_one_epoch, evaluate
from main_finetune import main, get_args_parser

_modules = [
    "models_vit",
    "engine_finetune",
    "main_finetune",
    # add any other root-level .py you want to expose
]

for name in _modules:
    mod = import_module(name)                    # load the root-level module
    globals()[name] = mod                        # make it available as RETFound.<name>
    sys.modules[f"{__name__}.{name}"] = mod      # <-- critical: register as a submodule

# Friendly aliases for public API
finetune_main = main
get_finetune_args = get_args_parser

__all__ = [
    # Types
    "RETFoundArgs",
    # Entrypoints
    "finetune_main",
    "get_finetune_args",
    "main",
    "get_args_parser",
    # Training utilities
    "train_one_epoch",
    "evaluate",
    # Submodules (available as RETFound.models_vit, RETFound.engine_finetune, ...)
    "models_vit",
    "engine_finetune",
]
