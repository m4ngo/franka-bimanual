"""The bimanual FR3 follower.

Everything here runs at policy rate. The control law does not: goals go over RPyC
to `pylibfranka_control`'s 1 kHz loop on each NUC, which recomputes torque every
tick. That split is robosuite's own -- this module is `set_goal`, the NUC is
`run_controller` -- and the pieces of `set_goal` that are worth reading against
osc.py live in `ee_goals.py` rather than in the middle of the camera plumbing.

All three control modes land on torque; pylibfranka exposes no Cartesian-velocity
interface and there is no velocity domain left anywhere in this stack.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property

import franka_config as fc
import numpy as np

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig
from lerobot.robots import Robot
from lerobot.types import RobotAction, RobotObservation

from lerobot_camera_arv import ArvCamera, ArvCameraConfig
from lerobot_camera_framos import FramosCamera, FramosCameraConfig

from . import homing
from .bimanual_franka_config import BimanualFrankaConfig, ControlMode
from .ee_goals import OSCGoalBuilder, delta_rotvec
from .franka_gripper import FrankaGripper
from .franka_process import NUM_JOINTS, KinematicSnapshot, MultiRobotWrapper
from .osc_torque_controller import DAMPING_EXP_SCALE, KP_EXP_SCALE, resolve_gains
from .safety import ActionSafetyScreen
from .wsg import WSG

# Every constant below comes from config/control.yaml. Per-rig trims are NOT here:
# they reach the robot through its config dataclass, whose default_factory reads
# the same yaml, so there is exactly one place each is written down.
IMAGE_CHANNELS = fc.control("observation.image_channels")
_CAMERA_READ_TIMEOUT_MS: float = fc.control("observation.camera_read_timeout_ms")
_CONNECT_TIMEOUT_S = fc.control("franka.connect_timeout_s")
_DEPTH_POINT_COUNT = fc.control("observation.depth_point_count")
# Age past which get_observation()'s kin snapshot is re-read instead of reused.
# EE_DELTA anchors its goal on the measured pose, so a stale anchor silently
# subtracts whatever the arm travelled in between from the commanded delta.
_KIN_CACHE_MAX_AGE_S = fc.control("observation.kin_cache_max_age_s")

JOINT_FEATURE_KEYS: tuple[str, ...] = (*(f"joint_{i}" for i in range(1, NUM_JOINTS + 1)), "gripper")
EE_AXIS_KEYS: tuple[str, ...] = ("x", "y", "z", "qx", "qy", "qz", "qw")
EE_FEATURE_KEYS: tuple[str, ...] = (*EE_AXIS_KEYS, "gripper")

_CAMERA_CTORS: dict[type, type] = {FramosCameraConfig: FramosCamera, ArvCameraConfig: ArvCamera}

logger = logging.getLogger(__name__)


def _make_camera(cfg: CameraConfig) -> Camera:
    cls = _CAMERA_CTORS.get(type(cfg))
    if cls is None:
        raise TypeError(f"Unsupported camera config: {type(cfg).__name__}")
    return cls(cfg)


class BimanualFranka(Robot):
    config_class = BimanualFrankaConfig
    name = "bimanual_franka"

    def __init__(self, config: BimanualFrankaConfig):
        super().__init__(config)
        self.config = config
        self.control_mode = config.control_mode
        self.active_arms = config.active_arms

        self.cameras: dict[str, Camera] = {n: _make_camera(c) for n, c in config.cameras.items()}
        # A key in both dicts is the SAME camera object, so it is read once per
        # observation and its cloud is derived from that same frame.
        self._depth_cameras: dict[str, Camera] = {
            n: self.cameras[n] if n in self.cameras else _make_camera(c)
            for n, c in (config.depth_cam if config.depth else {}).items()
        }
        self._camera_pool = ThreadPoolExecutor(max_workers=max(len(self._all_cameras) + 1, 1))

        self.robot_manager = MultiRobotWrapper()
        self.grippers: dict[str, WSG | FrankaGripper] = {
            arm: self._make_gripper(arm) for arm in self.active_arms
        }

        # Robot base expressed in world (config/world.yaml): p_world = R @ p_base + t.
        # No inversion -- the pose is already base-in-world, which is the direction
        # every consumer (safety brake, depth crop, viz, sysid) needs.
        self._base_in_world_by_arm = {arm: config.base_in_world(arm) for arm in self.active_arms}
        # The worktable brake compares world-frame heights, so it needs each arm's
        # base pose rather than one shared base-frame threshold, plus each arm's EE
        # collision sphere (grippers differ in size between arms).
        self.safety = ActionSafetyScreen(
            self._base_in_world_by_arm,
            {arm: fc.ee_sphere(config.arm_name(arm)) for arm in self.active_arms},
        )

        # Fall back when the profile's depth-centre arm isn't among active_arms.
        self._depth_center_arm = (
            config.depth_center_arm if config.depth_center_arm in self.active_arms
            else self.active_arms[0]
        )
        self._base_in_world = self._base_in_world_by_arm[self._depth_center_arm]
        # Read directly by residual_wrapper for its world-frame proprio.
        self._r_robot_in_world = self._base_in_world.rotation
        self._t_robot_in_world = self._base_in_world.translation
        # Half-extent of the world-axis-aligned box crop (sim collect convention).
        self._depth_crop_radius_m = float(config.depth_crop_radius_m)
        self._last_full_point_cloud: np.ndarray | None = None

        # The OSC goal actually dispatched, per arm, and the measured pose it was
        # composed on. Read-only: nothing in the control path consumes them, so
        # they add no limit layer -- they exist so a rollout can record what the
        # arm was commanded instead of an estimate of it.
        self._last_osc_goal: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._last_osc_anchor: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        self._goals = OSCGoalBuilder(
            translation_fudge=config.ee_translation_fudge,
            rotation_fudge=config.ee_rotation_fudge,
            use_noise=config.use_noise,
            noise_pos_scale=config.noise_pos_scale,
            noise_rot_scale=config.noise_rot_scale,
        )
        # Per-rig gain trims, kept as plain attributes because sysid/tune.py sweeps
        # them by setattr on the robot and the parity harness reads them back off it.
        self._kp_ori_scale = np.asarray(config.kp_ori_scale, dtype=np.float64)
        self._kp_pos_scale = np.asarray(config.kp_pos_scale, dtype=np.float64)
        self._kd_ori_scale = np.asarray(config.kd_ori_scale, dtype=np.float64)
        self._kd_pos_scale = np.asarray(config.kd_pos_scale, dtype=np.float64)
        # robosuite's OSC nullspace reference (initial_joint); seeded in connect(),
        # re-anchored by home().
        self._home_q: dict[str, np.ndarray] = {}
        # Residual offsets added on top of action commands via cache_delta().
        self.delta_pos = np.zeros(3)
        self.delta_rot = np.zeros(3)

        # Populated by get_observation, consumed by the next send_action to skip a
        # redundant RPyC round-trip.
        self._cached_kin_state: dict[str, KinematicSnapshot] | None = None
        self._cached_kin_ts: float = 0.0
        self._kin_cache_stale = 0

    def _make_gripper(self, arm: str) -> WSG | FrankaGripper:
        gripper_ip = getattr(self.config, f"{arm}_gripper_ip")
        if gripper_ip == getattr(self.config, f"{arm}_robot_ip"):
            return FrankaGripper(
                name=arm,
                server_ip=getattr(self.config, f"{arm}_server_ip"),
                robot_ip=getattr(self.config, f"{arm}_robot_ip"),
                # No fallback to {arm}_port: both configs resolve this from
                # arms.yaml, and silently using the ARM's port instead sends
                # gripper commands to the torque server, which just refuses the
                # connection somewhere far from the cause.
                port=getattr(self.config, f"{arm}_gripper_port"),
                do_print=False,
            )
        return WSG(name=arm, TCP_IP=gripper_ip, do_print=False)

    @property
    def _all_cameras(self) -> dict[str, Camera]:
        """Every camera this robot owns. Same-key entries are one object."""
        return {**self.cameras, **self._depth_cameras}

    @property
    def _cloud_points(self) -> int:
        """Per-camera share of the fixed total cloud size, so the concatenated
        cloud is the same length however many depth cameras the rig has."""
        return _DEPTH_POINT_COUNT // max(len(self._depth_cameras), 1)

    # ---------------------------------------------------------------- features

    def _arm_features(self, keys: tuple[str, ...]) -> dict[str, type]:
        return {f"{arm}_{key}": float for arm in self.active_arms for key in keys}

    @cached_property
    def _camera_features(self) -> dict[str, tuple[int, int, int]]:
        out: dict[str, tuple[int, int, int]] = {}
        for n, cam in self.cameras.items():
            if cam.height is None or cam.width is None:
                raise RuntimeError(f"Camera '{n}' does not report height/width")
            out[n] = (int(cam.height), int(cam.width), IMAGE_CHANNELS)
        return out

    @property
    def observation_features(self) -> dict[str, type | tuple[int, int, int]]:
        # The depth cloud is not in here: it is an array, reached through
        # last_full_point_cloud, not 6144 scalar observation entries.
        return {**self._arm_features(JOINT_FEATURE_KEYS), **self._camera_features}

    @property
    def action_features(self) -> dict[str, type]:
        keys = JOINT_FEATURE_KEYS if self.control_mode == ControlMode.JOINT_POS else EE_FEATURE_KEYS
        return {**self._arm_features(keys), "kp": float, "kd": float}

    # --------------------------------------------------------------- lifecycle

    @property
    def is_connected(self) -> bool:
        return self.robot_manager.num_alive == len(self.active_arms)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        try:
            for name, cam in self._all_cameras.items():
                try:
                    cam.connect()
                except Exception as e:
                    logger.warning("Camera %s failed to connect: %s", name, e)
            for arm in self.active_arms:
                self.robot_manager.add_robot(
                    arm,
                    getattr(self.config, f"{arm}_server_ip"),
                    getattr(self.config, f"{arm}_robot_ip"),
                    getattr(self.config, f"{arm}_port"),
                    use_ee_delta=self.control_mode != ControlMode.JOINT_POS,
                )
                snap = self.robot_manager.current_kinematic_state(arm, timeout_s=_CONNECT_TIMEOUT_S)
                # Seed the nullspace reference and the latched goal orientation
                # before any goal is pushed, the way robosuite's Controller.__init__
                # captures initial_joint and reset_goal() parks goal_ori.
                self._home_q[arm] = np.asarray(snap[0], dtype=np.float64).copy()
                self._goals.reset(arm, snap[4])
            # ALWAYS push, even at the default. Server sessions are keyed by
            # robot_ip and outlive the client, so an assist set by an earlier script
            # (a probe run with --friction-kc, say) otherwise silently persists into
            # the next run, and a sysid sweep would measure a controller nobody
            # configured.
            self.robot_manager.set_tuning_all(friction_kc=self.friction_kc)
            for arm in self.active_arms:
                self.grippers[arm].home()
        except Exception:
            self.robot_manager.shutdown()
            raise

    def disconnect(self) -> None:
        self._camera_pool.shutdown(wait=False)
        self._cached_kin_state = None
        for cam in self._all_cameras.values():
            cam.disconnect()
        self.robot_manager.shutdown()
        for g in self.grippers.values():
            g.close()

    # -------------------------------------------------------------- observation

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        # Arm state FIRST: the depth crop needs ee_world at submit time, and that is
        # what lets each camera's cloud computation chain onto its own frame instead
        # of waiting for every camera in the rig to finish reading.
        kin = self.robot_manager.current_kinematic_state_batch(list(self.active_arms))
        self._cached_kin_state = kin
        self._cached_kin_ts = time.perf_counter()

        futs = self._submit_camera_reads(kin)

        obs: RobotObservation = {}
        for arm in self.active_arms:
            for i, qi in enumerate(kin[arm][0]):
                obs[f"{arm}_joint_{i + 1}"] = float(qi)
            gripper = self.grippers[arm]
            pos = gripper.position
            obs[f"{arm}_gripper"] = (0.0 if pos is None else pos) / gripper.GRIPPER_TRUE_MAX_MM

        clouds: dict[str, np.ndarray] = {}
        for name, fut in futs.items():
            try:
                img, cloud = fut.result()
            except Exception as e:
                logger.warning("Camera %s read failed: %s", name, e)
                img, cloud = None, None
            if name in self.cameras:
                obs[name] = img if img is not None else self._blank_frame(name)
            if name in self._depth_cameras:
                clouds[name] = (cloud if cloud is not None
                                else np.zeros((self._cloud_points, 3), dtype=np.float32))

        if self._depth_cameras:
            # Concatenated in _depth_cameras order so the cloud's per-camera layout
            # is stable across calls. Exposed as an array via last_full_point_cloud,
            # never as scalar obs entries -- flattening it to 6144 float keys and
            # rebuilding it cost ~1.1 ms/step for nothing.
            self._last_full_point_cloud = np.concatenate(
                [clouds[name] for name in self._depth_cameras], axis=0
            )
        return obs

    def _submit_camera_reads(self, kin: dict[str, KinematicSnapshot]):
        ee_world = self._base_in_world.apply(
            np.asarray(kin[self._depth_center_arm][3], dtype=np.float64)
        )

        def read(cam: Camera, with_cloud: bool):
            img = cam.async_read(_CAMERA_READ_TIMEOUT_MS)
            if not with_cloud:
                return img, None
            return img, cam.get_cropped_point_cloud(
                ee_world, self._depth_crop_radius_m, self._cloud_points)

        return {
            name: self._camera_pool.submit(read, cam, name in self._depth_cameras)
            for name, cam in self._all_cameras.items()
        }

    def _blank_frame(self, name: str) -> np.ndarray:
        """Last known image, or black. A camera failure degrades the observation
        rather than ending the episode."""
        blank = getattr(self.cameras[name], "blank_frame", None)
        if callable(blank):
            return blank()
        return np.zeros(self._camera_features[name], dtype=np.uint8)

    # ------------------------------------------------------------------ actions

    def send_action(self, action: RobotAction, ignore_action: bool = False) -> RobotAction:
        """Push this policy step's goal to the arms' 1 kHz torque loops.

        Mirrors robosuite's split: this is ``set_goal`` (once per policy step),
        while ``run_controller`` runs server-side every tick.
        """
        # Consumed at most once, whether or not this mode needs it: a snapshot held
        # into the next step would silently anchor that step's delta on this step's
        # pose.
        cached, self._cached_kin_state = self._cached_kin_state, None

        self._command_grippers(action)
        if self.control_mode == ControlMode.JOINT_POS:
            # No kinematic read at all: a joint setpoint is absolute, so there is
            # nothing to anchor and the RPyC round-trip would be pure latency.
            self._command_joints(action)
        else:
            self._command_osc(action, self._kinematic_state(cached), ignore_action)
        return action

    def _kinematic_state(self, cached: dict[str, KinematicSnapshot] | None):
        """The snapshot this step's goal is anchored on.

        Reuses get_observation's if it is fresh enough. In practice the bound
        always trips and this always re-reads -- the snapshot is taken before six
        camera reads, so it is tens of ms old by the time we see it. What actually
        keeps the anchor fresh is torque.loop.publish_decimation, since the anchor
        can never be newer than the last published state.
        """
        if cached is not None:
            if time.perf_counter() - self._cached_kin_ts <= _KIN_CACHE_MAX_AGE_S:
                return cached
            self._kin_cache_stale += 1
        return self.robot_manager.current_kinematic_state_batch(list(self.active_arms))

    def _command_grippers(self, action: RobotAction) -> None:
        """``{arm}_gripper`` is an ABSOLUTE normalized position in [0, 1].

        The same units the observation reports and every leader emits, so a
        recorded episode replays as itself. This used to integrate the value into
        a running accumulator, which only made sense for the SpaceMouse's momentary
        buttons and left GELLO -- which emits an absolute position -- pinned at
        whichever end the accumulator saturated against. The latch now lives in the
        SpaceMouse, where the momentary signal is.
        """
        for arm in self.active_arms:
            gripper = self.grippers[arm]
            target = float(np.clip(action[f"{arm}_gripper"], 0.0, 1.0))
            gripper.move(target * gripper.GRIPPER_TRUE_MAX_MM, blocking=False)

    def _command_joints(self, action: RobotAction) -> None:
        """Joint-impedance goal. Not screened: the worktable floor is a bound on an
        EE goal pose and a joint-position command has none; JOINT_POS is GELLO
        teleop with an operator in the loop.

        The joint law has no sim counterpart, so the gain channels are the bare
        exponential remap without osc.py's clip -- unlike the OSC path, which goes
        through resolve_gains.
        """
        kp_scale = KP_EXP_SCALE ** float(np.clip(action["kp"], -1.0, 1.0))
        kd_scale = DAMPING_EXP_SCALE ** float(np.clip(action["kd"], -1.0, 1.0))
        goals = {
            arm: (
                np.fromiter((action[f"{arm}_joint_{i}"] for i in range(1, NUM_JOINTS + 1)),
                            dtype=np.float64, count=NUM_JOINTS),
                kp_scale,
                kd_scale,
            )
            for arm in self.active_arms
        }
        self.robot_manager.move_joint_goal_batch(goals)

    def _command_osc(
        self, action: RobotAction, kin: dict[str, KinematicSnapshot], ignore_action: bool
    ) -> None:
        kp, kd = resolve_gains(
            action["kp"], action["kd"], self._kp_ori_scale, self._kd_ori_scale,
            kp_pos_scale=self._kp_pos_scale, kd_pos_scale=self._kd_pos_scale,
        )
        goals = {arm: self._osc_goal(arm, action, kin[arm], ignore_action) for arm in self.active_arms}
        goals = self.safety.shape_goal(goals)
        # After clip, fudge, the latched goal_ori and the worktable screen: the pose
        # the arm was told to hold, which is what a sim replay has to reproduce.
        # Copied -- move_osc_goal_batch ships these over RPyC.
        self._last_osc_goal = {a: (np.array(p, dtype=np.float64), np.array(q, dtype=np.float64))
                               for a, (p, q) in goals.items()}
        self._last_osc_anchor = {a: (np.array(kin[a][3], dtype=np.float64),
                                     np.array(kin[a][4], dtype=np.float64))
                                 for a in self.active_arms}
        self.robot_manager.move_osc_goal_batch(
            {a: (pos, quat, kp, kd, self._home_q.get(a)) for a, (pos, quat) in goals.items()}
        )

    def _osc_goal(self, arm: str, action: RobotAction, snap: KinematicSnapshot, ignore_action: bool):
        _, _, _, ee_pos, ee_quat_xyzw, _ = snap
        if self.control_mode == ControlMode.EE_DELTA:
            # The residual is summed into the delta BEFORE the envelope, so it is
            # clipped and counts toward "was a rotation commanded" like any other
            # delta. See OSCGoalBuilder.from_delta.
            dpos = self._action_vec(action, arm, ("x", "y", "z")) + self.delta_pos
            drot = delta_rotvec(self._action_vec(action, arm, ("qx", "qy", "qz", "qw"))) + self.delta_rot
            return self._goals.from_delta(arm, dpos, drot, ee_pos, ee_quat_xyzw)

        if ignore_action:
            # Park the goal on the current pose, so only the cache_delta() residual
            # moves the arm.
            goal = self._goals.absolute(ee_pos, ee_quat_xyzw)
        else:
            target = self._action_vec(action, arm, EE_AXIS_KEYS)
            goal = self._goals.absolute(target[:3], target[3:])
        goal = self._goals.offset(goal, self.delta_pos, self.delta_rot)
        # Last, so it lands on the goal the arm is told to hold -- the residual
        # included -- as from_delta's noise does on the delta path.
        return self._goals.perturb(goal)

    @staticmethod
    def _action_vec(action: RobotAction, arm: str, keys: tuple[str, ...]) -> np.ndarray:
        return np.fromiter((action[f"{arm}_{k}"] for k in keys), dtype=np.float64, count=len(keys))

    # ------------------------------------------------------------------- homing

    def home(
        self,
        home_q_left: np.ndarray | None,
        home_q_right: np.ndarray | None,
        gripper_norm: float = fc.control("homing.gripper_norm"),
        max_time_s: float = fc.control("homing.max_time_s"),
        tol_rad: float = fc.control("homing.tol_rad"),
        fps: int = fc.control_fps(),
        *,
        home_fps: int | None = None,
        **_unused,
    ) -> bool:
        """Drive both arms to a saved home configuration.

        ``home_q_*`` is the desired ``q`` in every control mode: homing always runs
        server-side joint impedance, the only law that reaches a joint configuration
        directly. The commanded goal LEADS the measured ``q`` rather than jumping to
        the target, which bounds the approach speed and lets it taper to zero on
        arrival instead of overshooting -- see `homing.HomingRamp`.

        On success the target also becomes the OSC nullspace reference. Convergence
        is judged in joint space against ``tol_rad``; ``**_unused`` swallows the
        EE-space tolerances older call sites still pass.
        """
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        candidates = {"l": home_q_left, "r": home_q_right}
        targets_q = {
            arm: np.asarray(q, dtype=np.float64)
            for arm, q in candidates.items()
            if q is not None and arm in self.active_arms
        }
        if not targets_q:
            return True

        for arm in targets_q:
            self.grippers[arm].move(
                gripper_norm * self.grippers[arm].GRIPPER_TRUE_MAX_MM, blocking=False
            )

        names = list(targets_q)
        rate_hz = float(home_fps if home_fps is not None else fc.home_fps())
        period_s = 1.0 / rate_hz
        deadline = time.perf_counter() + max_time_s
        start = self.robot_manager.current_kinematic_state_batch(names)
        ramp = homing.HomingRamp({arm: snap[0] for arm, snap in start.items()}, targets_q, rate_hz)

        while True:
            tick_start = time.perf_counter()
            kin = self.robot_manager.current_kinematic_state_batch(names)

            # Not screened: a home configuration is a saved, known-safe q, and the
            # worktable floor only bounds an EE goal pose.
            self.robot_manager.move_joint_goal_batch(
                {arm: (goal, 1.0, 1.0)
                 for arm, goal in ramp.advance({arm: kin[arm][0] for arm in names}).items()}
            )

            max_err = max(float(np.max(np.abs(targets_q[arm] - kin[arm][0]))) for arm in names)
            # Exit at rest, not merely in position: returning mid-motion leaves the
            # arm coasting into whatever the caller does next.
            max_qdot = max(float(np.max(np.abs(kin[arm][1]))) for arm in names)
            converged = max_err < tol_rad and max_qdot < homing.SETTLE_QDOT

            if converged or tick_start >= deadline:
                self._cached_kin_state = None
                if converged:
                    for arm in names:
                        self._home_q[arm] = targets_q[arm].copy()
                else:
                    logger.warning(
                        "home(): timeout after %.2fs, max joint error %.4f rad", max_time_s, max_err
                    )
                # robosuite reset_goal(): park the held orientation on the current
                # pose, so post-home EE_DELTA offsets are relative to where the arm
                # actually is.
                for arm, snap in kin.items():
                    self._goals.reset(arm, snap[4])
                return converged

            elapsed = time.perf_counter() - tick_start
            if elapsed < period_s:
                time.sleep(period_s - elapsed)

    # ------------------------------------------------------- tuning and readouts

    @property
    def friction_kc(self) -> np.ndarray:
        """Coulomb assist scale as (2, 7): the scalar times the per-joint trim,
        split by rotation direction. Row 0 applies where the commanded torque on
        that joint is positive, row 1 where it is negative -- breakaway on this arm
        is directional, so one symmetric gain either under-assists the hard
        direction or over-assists the easy one."""
        return float(self.config.friction_kc) * np.stack([
            np.asarray(self.config.friction_kc_joint_pos, dtype=np.float64),
            np.asarray(self.config.friction_kc_joint_neg, dtype=np.float64),
        ])

    def set_friction_kc(self, kc) -> None:
        """Push a new assist scale to the running control loops.

        Scalar or (7,) sets both directions; (2, 7) sets them independently.
        """
        kc = np.asarray(kc, dtype=np.float64)
        if kc.size != 2 * NUM_JOINTS:
            kc = np.broadcast_to(kc, (NUM_JOINTS,))
            kc = np.stack([kc, kc])
        self.robot_manager.set_tuning_all(friction_kc=kc.reshape(2, NUM_JOINTS))

    # The delta fudges and the latched goal orientation live in OSCGoalBuilder, but
    # sysid/tune.py sweeps them by setattr on the robot and the parity harness reads
    # them back off it. Forwarding keeps one definition without breaking either.
    @property
    def _trans_fudge(self) -> float:
        return self._goals._trans_fudge

    @_trans_fudge.setter
    def _trans_fudge(self, v: float) -> None:
        self._goals._trans_fudge = float(v)

    @property
    def _rot_fudge(self) -> float:
        return self._goals._rot_fudge

    @_rot_fudge.setter
    def _rot_fudge(self, v: float) -> None:
        self._goals._rot_fudge = float(v)

    @property
    def _osc_goal_ori(self) -> dict:
        """osc.py's latched goal_ori, per arm. Mutable in place; the parity harness
        seeds it before the first step."""
        return self._goals._goal_ori

    @property
    def kin(self) -> dict[str, KinematicSnapshot] | None:
        return self._cached_kin_state

    @property
    def last_ee_wrench(self) -> dict[str, np.ndarray | None]:
        """Per arm, libfranka's estimated external wrench at the EE from the latest
        state read (get_observation's, or send_action's re-read): force (N) then
        torque (Nm), base frame. None for an arm whose NUC server predates it.
        Read-only diagnostics, like _last_osc_goal."""
        return {arm: self.robot_manager.ee_wrench(arm) for arm in self.active_arms}

    @property
    def base_in_world(self):
        """`franka_config.Pose` mapping robot-base coordinates into world."""
        return self._base_in_world

    def cache_delta(self, dpos: np.ndarray, drot: np.ndarray) -> None:
        self.delta_pos = dpos
        self.delta_rot = drot

    @property
    def last_full_point_cloud(self) -> np.ndarray | None:
        """Cropped and subsampled world-space point cloud from the depth cameras.

        Updated every get_observation() call when depth is enabled.
        Shape: (N, 3) float32 in world-frame metres, or None before the first
        observation.
        """
        return self._last_full_point_cloud
