# ------------------------------------------------------------------------
# GroundedDINO-VL
# Based on Grounding DINO by IDEA-Research
# Copyright (c) 2025 GhostCipher. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Original Copyright (c) 2023 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
from .grounding_dino import build_groundingdino


def build_model(args):
    """Build a model based on configuration.

    Args:
        args: Configuration object with modelname attribute

    Returns:
        Built model instance
    r"""
    # Import registry here to avoid circular imports
    from .registry import MODULE_BUILD_FUNCS

    assert (
        args.modelname in MODULE_BUILD_FUNCS._module_dict
    ), f"Model {args.modelname} not found in registry"

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model = build_func(args)
    return model


__all__ = ["build_model", "build_groundingdino"]
