"""OpenPI (pi05_libero) base-policy wrapper, drop-in compatible with
policy_wrapper.BasePolicy's interface so it can replace `base_policy` in
run_residual.py's `_run_episode` without touching that loop.

Architecture
------------
OpenPI models are served out-of-process via `scripts/serve_policy.py`
(websocket server) and queried with the `openpi_client` package -- this is
the documented/supported path for real-robot deployment (keeps the JAX/PyTorch
+ CUDA stack for the VLA isolated from the robot control process). Start the
server separately, e.g.:

    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_libero \
        --policy.dir=gs://openpi-assets/checkpoints/pi05_libero

Then point OpenPIBasePolicy at its host:port.

Action-space bridging
----------------------
pi05_libero outputs (action_horizon, 7) = [dq1..dq7] in JOINT space for this
Franka checkpoint (see env_wrapper_openpi_ext.py for the exact convention).
The checkpoint is arm-only: it emits 7 joint-delta channels and relies on the
robot's live gripper state being carried through unchanged.
Your residual pipeline (process_chunk / build_action / _ee_delta) expects (T,
10) EE-pose deltas. `infer()` here does the FK-based joint-delta -> EE-delta
conversion inline via `joint_deltas_to_ee_chunk`, so callers see the exact
same (T, 10) contract BasePolicy.infer() already provides -- no changes
needed in run_residual.py.

Because the conversion needs the joint configuration used as the seed for FK
integration, and OpenPI's `observation/state` is also joint-based, this
wrapper reads `r_joint_1..7` directly off the raw obs dict passed to infer()
(same dict run_residual.py already builds via strip_depth(obs)) rather than
taking a separately-threaded current_q argument -- matching the calling
convention `base_policy.infer(obs_no_depth)` uses in run_residual.py's loop.
"""

import logging
from typing import Optional

import numpy as np

from env_wrapper import _STATE_OBS_KEYS
from env_wrapper_openpi_ext import joint_deltas_to_ee_chunk

logger = logging.getLogger(__name__)


def _format_libero_image(img: np.ndarray) -> np.ndarray:
    """Ensure uint8, matching openpi's image_tools.convert_to_uint8.

    Camera rig here already outputs 224x224 RGB, so no resize/pad is needed
    (resize_with_pad would be a no-op at matching size anyway). Only a dtype
    cast is applied. If a camera obs key ever changes resolution, prefer
    `openpi_client.image_tools.resize_with_pad` over naive cv2.resize to
    match training-time preprocessing (it letterboxes rather than stretching).
    """
    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.floating):
        return (255 * img).astype(np.uint8)
    return img.astype(np.uint8)


class OpenPIBasePolicy:
    """BasePolicy-compatible wrapper around a remote pi05_libero server.

    Mirrors policy_wrapper.BasePolicy: reset() and infer(obs) -> (T, 10)
    np.ndarray in the EE-delta chunk format process_chunk/build_action expect.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        prompt: str = "",
        image_key: str = "cam_6_scene",
        wrist_image_key: str = "cam_3_wrist",
        default_kp: float = 0.3,
        default_kd: float = 0.05,
    ) -> None:
        """
        Args:
            host, port: address of a running `scripts/serve_policy.py`
                instance loaded with --policy.config=pi05_libero
                --policy.dir=gs://openpi-assets/checkpoints/pi05_libero
                (or a fine-tuned checkpoint using the same config).
            prompt: language instruction sent with every observation.
                Set via set_prompt() to change mid-run (e.g. per recorded
                episode task string).
            image_key, wrist_image_key: keys into the raw obs dict
                (obs_no_depth passed by run_residual.py) holding the two
                camera images pi05_libero expects as
                observation/image and observation/wrist_image respectively.
                pi05_libero only takes ONE third-person view + ONE wrist
                view, but this rig has three cameras (cam_3_wrist,
                cam_4_wrist, cam_6_scene). Default here maps the scene cam
                to observation/image and cam_3_wrist to
                observation/wrist_image; pass wrist_image_key="cam_4_wrist"
                to use the other wrist camera instead. The unused third
                camera is simply not read by this wrapper (it's still
                available in obs for recording/dataset purposes elsewhere
                in the pipeline).
            default_kp, default_kd: gains forwarded on every produced
                action, since OpenPI has no notion of impedance gains.
                These become build_action's kp/kd when --no-residual is
                used, or prev_kp/prev_kd context when the residual is
                active (the residual policy itself picks the executed
                gains from its own [damping, stiffness] output columns
                exactly as in run_residual.py; only the base policy's own
                per-step channel needs *a* value here).
        """
        from openpi_client import websocket_client_policy

        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.prompt = prompt
        self.image_key = image_key
        self.wrist_image_key = wrist_image_key
        self.default_kp = default_kp
        self.default_kd = default_kd
        logger.info("OpenPIBasePolicy connected to %s:%d (prompt=%r)", host, port, prompt)

    def set_prompt(self, prompt: str) -> None:
        self.prompt = prompt

    def reset(self) -> None:
        # The websocket server is stateless per infer() call for pi05 (no
        # internal action queue on the client side -- run_residual.py's
        # chunk_used/_CHUNK_EXEC bookkeeping already handles open-loop
        # chunk execution). Nothing to reset client-side.
        pass

    def infer(self, obs: dict) -> np.ndarray:
        """Run one inference pass against the remote pi05_libero server.

        Args:
            obs: raw obs dict as passed to base_policy.infer() in
                run_residual.py (i.e. obs_no_depth = strip_depth(obs));
                must contain r_joint_1..7, r_gripper, and the two camera
                images under self.image_key / self.wrist_image_key.

        Returns:
            (T, 10) numpy array -- [dx, dy, dz, dqx, dqy, dqz, dqw,
            gripper, kp, kd], the same EE-delta chunk format
            policy_wrapper.BasePolicy.infer() returns, ready for
            process_chunk()/build_action() unchanged.
        """
        current_q = np.array([obs[f"r_joint_{i}"] for i in range(1, 8)], dtype=np.float64)
        state = np.array([obs[k] for k in _STATE_OBS_KEYS], dtype=np.float32)  # (8,)

        observation = {
            "observation/image": _format_libero_image(obs[self.image_key]),
            "observation/wrist_image": _format_libero_image(obs[self.wrist_image_key]),
            "observation/state": state,
            "prompt": self.prompt,
        }

        result = self.client.infer(observation)
        openpi_chunk = np.asarray(result["actions"], dtype=np.float32)

        return joint_deltas_to_ee_chunk(
            openpi_chunk,
            current_q,
            current_gripper=float(obs["r_gripper"]),
            kp=self.default_kp,
            kd=self.default_kd,
        )
