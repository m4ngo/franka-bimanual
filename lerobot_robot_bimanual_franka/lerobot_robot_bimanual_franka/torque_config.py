"""config/control.yaml's ``torque`` block, resolved for BOTH sides of the RPyC link.

The torque modules (``osc_torque_controller``, ``pylibfranka_control``,
``pylibfranka_server``) run in two very different places:

- On the **workstation** they are ordinary package modules and ``franka_config``
  is installed, so the block is read straight out of ``config/control.yaml``.
- On the **NUC** they are copied flat into ``~/`` by
  ``scripts/deploy_nuc_server.sh`` and run in an env that is numpy + pylibfranka
  only. Installing ``franka_config`` there would put a YAML parse and a
  directory search inside the process that owns the 1 kHz loop, so the deploy
  script resolves the block on the workstation and writes it next to the server
  as ``nuc_control_config.py`` instead.

Either way ``config/control.yaml`` is the single source of truth; there is no
third copy of a gain anywhere. A missing source is a hard error rather than a
fallback to baked-in defaults -- a silently stale gain is exactly the failure
this indirection exists to prevent.

Redeploy after editing ``control.yaml``: the workstation copy is not what runs.
"""

from __future__ import annotations

from typing import Any

_MISSING = object()


def _load() -> dict[str, Any]:
    try:
        import franka_config as fc  # type: ignore
    except ImportError:
        pass
    else:
        return fc.control("torque")

    try:
        from nuc_control_config import TORQUE  # type: ignore
    except ImportError:
        raise RuntimeError(
            "Cannot resolve the `torque` block of config/control.yaml: neither "
            "franka_config nor a deploy-generated nuc_control_config.py is "
            "importable. On the NUC, re-run scripts/deploy_nuc_server.sh; on the "
            "workstation, `uv pip install --no-deps -e ~/franka_ws/franka_config "
            "-C editable_mode=compat` or set FRANKA_CONFIG_DIR."
        ) from None
    return TORQUE


TORQUE: dict[str, Any] = _load()


def torque(path: str, default: Any = _MISSING) -> Any:
    """Dotted lookup inside the torque block, e.g. ``torque("osc.default_kp")``."""
    node: Any = TORQUE
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            if default is not _MISSING:
                return default
            raise KeyError(f"config path not found: torque.{path} (missing {key!r})")
        node = node[key]
    return node
