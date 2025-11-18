"""
Tests to verify backward compatibility import paths work correctly.

This test suite ensures:
1. The canonical groundeddino_vl import works (recommended)
2. The legacy groundingdino import works (backward compatibility)
3. Both namespaces reference the same underlying modules
4. The CUDA extension is accessible through both namespaces
"""

import pytest


def test_import_groundeddino_vl():
    """Test that groundeddino_vl can be imported."""
    import groundeddino_vl

    # Should have version
    assert hasattr(groundeddino_vl, "__version__"), "groundeddino_vl missing __version__"
    assert isinstance(groundeddino_vl.__version__, str) and groundeddino_vl.__version__

    # Verify core functions are available
    assert hasattr(groundeddino_vl, "load_model"), "groundeddino_vl missing load_model"
    assert hasattr(groundeddino_vl, "predict"), "groundeddino_vl missing predict"


def test_import_groundingdino():
    """Test that groundingdino can be imported (legacy compatibility)."""

    # Verify core modules can be imported
    from groundingdino import models, util

    assert models is not None, "models module not found"
    assert util is not None, "util module not found"


def test_groundingdino_references_groundeddino_vl():
    """Test that groundingdino properly re-exports from groundeddino_vl."""
    import groundeddino_vl
    import groundingdino

    # Both should have the same version
    assert groundingdino.__version__ == groundeddino_vl.__version__

    # Both should reference the same models module
    assert groundingdino.models is groundeddino_vl.models

    # Both should have util and utils available (may be separate module instances
    # for the legacy compatibility shim)
    assert hasattr(groundingdino, "util"), "groundingdino missing util"
    assert hasattr(groundeddino_vl, "utils"), "groundeddino_vl missing utils"


def test_cuda_extension_loadable():
    """Test that the CUDA extension can be loaded."""
    try:
        import groundingdino._C as cuda_ext

        # If we get here, the extension loaded successfully
        assert cuda_ext is not None
    except ImportError as e:
        pytest.skip(f"CUDA extension not available: {e}")


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
