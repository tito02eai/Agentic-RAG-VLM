"""Shared environment probes for the regression suite.

The standalone repository does not distribute the school-provided Guanghua
asset bundle, and some machines cannot initialize a MuJoCo runtime at all
(for example headless Windows without working OpenGL drivers).  These probes
let ``pytest -q tests`` degrade to informative skips instead of collection
errors in such environments.  They verify environment capability only; they
do not change any frozen experiment protocol.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GUANGHUA_MJCF = (
    PROJECT_ROOT / "assets" / "guanghua_hand_env" / "mjcf" / "guanghua_hand_env.xml"
)
GUANGHUA_URDF = (
    PROJECT_ROOT
    / "assets"
    / "guanghua_hand_env"
    / "urdf"
    / "guanghua_robot_with_hands.urdf"
)


def guanghua_assets_present() -> bool:
    """True when the locally obtained school asset bundle is in place."""
    return GUANGHUA_MJCF.is_file() and GUANGHUA_URDF.is_file()


def mujoco_runtime_available() -> bool:
    """True when the MuJoCo Python bindings import and initialize.

    ``import mujoco`` can raise ``OSError`` (DLL initialization failure), not
    only ``ImportError``, so every exception is treated as unavailable.
    """
    try:
        import mujoco  # noqa: F401
    except Exception:
        return False
    return True
