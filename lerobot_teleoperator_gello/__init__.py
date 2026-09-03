# Flat-layout shim. Re-exports the inner package's own __all__ rather than a
# hand-kept copy of it: the copy drifted, and GelloEEConfig / BimanualGelloEE
# were missing from it, so `from lerobot_teleoperator_gello import GelloEEConfig`
# failed on import for anything outside the package.
from .lerobot_teleoperator_gello import *  # noqa: F401,F403
from .lerobot_teleoperator_gello import __all__  # noqa: F401
