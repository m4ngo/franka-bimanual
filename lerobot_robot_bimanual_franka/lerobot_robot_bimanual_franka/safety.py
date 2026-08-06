"""Action safety screens for the bimanual Franka.

All checks are pure (action, kin_state) → action transforms applied before dispatch.
Currently implements a worktable brake; bimanual arm-repel is not yet implemented.

Every threshold here comes from config/control.yaml and config/world.yaml.

The worktable brake reasons in WORLD frame: the table is one plane shared by
both arms, so each arm's EE height and vertical velocity are lifted through
that arm's `robot_base_in_world` before being compared against it. Arms whose
bases sit at different heights or orientations therefore get the right
envelope without any per-arm threshold.

The EE is treated as a SPHERE, not a point. Its centre is given in the TOOL
frame and rotates with the gripper, and clearance is measured from the
sphere's lowest point — so a tilted gripper cannot graze the table with its
side while the TCP still reads as clear.
"""

import franka_config as fc  # type: ignore
import numpy as np

from .franka_process import KinematicSnapshot

WORKTABLE_HEIGHT = fc.worktable_height_m()                                   # m, WORLD-frame Z of the worktable surface
EE_SPHERE = fc.default_ee_sphere()                                           # shared EE collision sphere; arms may override
WORKTABLE_DISTANCE_MIN = fc.control("worktable_brake.distance_min_m")        # m, minimum clearance; downward velocity zeroed at/past this
WORKTABLE_MAX_DECEL = fc.control("worktable_brake.max_decel_mps2")           # m/s², assumed deceleration for the braking envelope
WORKTABLE_VELOCITY_EPS = fc.control("worktable_brake.velocity_eps_mps")      # m/s, ignore commands smaller than this (float noise)

JOINT_VELOCITY_MAX = fc.control("limits.joint_velocity_max")          # rad/s, L2-norm ceiling on joint velocity commands
EE_LINEAR_VELOCITY_MAX = fc.control("limits.ee_linear_velocity_max")  # m/s
EE_ANGULAR_VELOCITY_MAX = fc.control("limits.ee_angular_velocity_max")  # rad/s


def _quat_xyzw_to_matrix(q_xyzw) -> np.ndarray:
    """Rotation matrix from an xyzw quaternion (the FR3 state's convention)."""
    return fc.quat_wxyz_to_matrix(fc.quat_xyzw_to_wxyz(tuple(np.asarray(q_xyzw, dtype=np.float64))))


def _point_velocity(twist: np.ndarray, lever: np.ndarray) -> np.ndarray:
    """Linear velocity of a point offset `lever` from the twist's reference point.

    Rigid-body carry: v_point = v + w x r. With a zero lever this is just the
    twist's linear part, which is the case for a sphere centred on the TCP.
    """
    twist = np.asarray(twist, dtype=np.float64)
    if not lever.any():
        return twist[:3]
    return twist[:3] + np.cross(twist[3:6], lever)


def _clamp_joint_velocity(velocity: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(velocity))
    return velocity * (JOINT_VELOCITY_MAX / norm) if norm > JOINT_VELOCITY_MAX else velocity


def _clamp_ee_twist(twist: np.ndarray) -> np.ndarray:
    twist = np.asarray(twist, dtype=np.float64).copy()
    linear, angular = twist[:3], twist[3:]
    lin_norm = float(np.linalg.norm(linear))
    if lin_norm > EE_LINEAR_VELOCITY_MAX:
        linear *= EE_LINEAR_VELOCITY_MAX / lin_norm
    ang_norm = float(np.linalg.norm(angular))
    if ang_norm > EE_ANGULAR_VELOCITY_MAX:
        angular *= EE_ANGULAR_VELOCITY_MAX / ang_norm
    twist[:3] = linear
    twist[3:] = angular
    return twist


class ActionSafetyScreen:
    """Screens arm actions before dispatch, braking toward-table motion.

    Args:
        base_in_world: arm key -> `franka_config.Pose` of that arm's base in the
            world frame. Required: the table height is world-frame, so an arm
            with no pose cannot be screened and is rejected rather than silently
            compared against the wrong frame.
        ee_spheres: arm key -> `franka_config.EESphere` collision volume. Any arm
            missing an entry falls back to the shared `EE_SPHERE` default.
    """

    def __init__(
        self,
        base_in_world: dict[str, "fc.Pose"],
        ee_spheres: dict[str, "fc.EESphere"] | None = None,
    ) -> None:
        if not base_in_world:
            raise ValueError("base_in_world must contain at least one arm pose.")
        self._base_in_world = dict(base_in_world)
        self._ee_spheres = {
            arm: (ee_spheres or {}).get(arm, EE_SPHERE) for arm in self._base_in_world
        }
        # World +Z expressed in each arm's base frame, so a base-frame vector's
        # world vertical component is just a dot product. Unit length because
        # the rotation is orthonormal.
        self._world_up_in_base = {
            arm: np.asarray(pose.rotation, dtype=np.float64)[2, :].copy()
            for arm, pose in self._base_in_world.items()
        }
        self._sphere_center_tool = {
            arm: np.asarray(sphere.center_tool_m, dtype=np.float64)
            for arm, sphere in self._ee_spheres.items()
        }

    def _pose_for(self, arm: str):
        try:
            return self._base_in_world[arm]
        except KeyError:
            raise KeyError(
                f"no robot_base_in_world pose for arm {arm!r}; known: "
                f"{sorted(self._base_in_world)}. The worktable brake is world-frame "
                "and cannot screen an arm whose base pose is unknown."
            ) from None

    def _sphere_geometry(self, arm: str, ee_translation, ee_quat_xyzw) -> tuple[float, np.ndarray]:
        """(world Z of the sphere's lowest point, lever arm from EE to centre).

        The centre is defined in the TOOL frame, so it is rotated by the current
        EE orientation before being lifted into world — that is what keeps the
        clearance honest when the gripper is tilted. The sphere is rotationally
        symmetric, so its lowest point is always centre_z - radius.

        The returned lever arm (base frame, EE origin -> centre) is what the
        caller needs to carry the EE twist over to the centre: v_c = v + w x r.
        """
        sphere = self._ee_spheres[arm]
        center_tool = self._sphere_center_tool[arm]
        lever_base = np.zeros(3)
        if center_tool.any():
            lever_base = _quat_xyzw_to_matrix(ee_quat_xyzw) @ center_tool
        center_base = np.asarray(ee_translation, dtype=np.float64) + lever_base
        center_world_z = float(self._pose_for(arm).apply(center_base)[2])
        return center_world_z - sphere.radius_m, lever_base

    def shape_ee(
        self,
        ee_by_arm: dict[str, np.ndarray],
        kin_state: dict[str, KinematicSnapshot],
    ) -> dict[str, np.ndarray]:
        ee_by_arm = self._apply_worktable_brake(ee_by_arm, kin_state, is_ee=True)
        return {arm: _clamp_ee_twist(twist) for arm, twist in ee_by_arm.items()}

    def shape_joint(
        self,
        joint_velocities_by_arm: dict[str, np.ndarray],
        kin_state: dict[str, KinematicSnapshot],
    ) -> dict[str, np.ndarray]:
        joint_velocities_by_arm = self._apply_worktable_brake(
            joint_velocities_by_arm, kin_state, is_ee=False
        )
        return {arm: _clamp_joint_velocity(vel) for arm, vel in joint_velocities_by_arm.items()}

    def _apply_worktable_brake(
        self,
        action_by_arm: dict[str, np.ndarray],
        kin_state: dict[str, KinematicSnapshot],
        is_ee: bool,
    ) -> dict[str, np.ndarray]:
        """Limit downward EE velocity to prevent table contact.

        Heights and vertical velocities are evaluated in WORLD frame against
        WORKTABLE_HEIGHT; the arm's own base pose supplies the conversion. The
        clearance is measured from the bottom of the EE collision sphere, not
        from the TCP, so tilting the gripper cannot hide its lower edge.

        Layer 1: cap downward speed at sqrt(2 * MAX_DECEL * clearance) so the arm can decelerate
        before reaching WORKTABLE_DISTANCE_MIN.
        Layer 2: if measured downward speed already exceeds the envelope, zero commanded downward vel.

        EE mode: the twist's linear part is corrected along world-up only, so
        horizontal motion is untouched.
        Joint mode: the whole vector is uniformly scaled to preserve joint-space direction.
        """
        out: dict[str, np.ndarray] = {}
        for arm, action in action_by_arm.items():
            action = np.asarray(action, dtype=np.float64)
            _, dq, jacobian, ee_translation, ee_quat_xyzw, _ = kin_state[arm]
            jacobian = np.asarray(jacobian, dtype=np.float64)
            dq = np.asarray(dq, dtype=np.float64)

            world_up = self._world_up_in_base[arm]
            contact_z, lever = self._sphere_geometry(arm, ee_translation, ee_quat_xyzw)

            safe_dist = contact_z - WORKTABLE_HEIGHT - WORKTABLE_DISTANCE_MIN
            v_envelope = 0.0 if safe_dist <= 0.0 else float(np.sqrt(2.0 * WORKTABLE_MAX_DECEL * safe_dist))
            # Velocity of the SPHERE CENTRE, not the EE origin: an offset sphere
            # on a rotating wrist descends faster than the TCP does.
            twist_measured = jacobian @ dq
            v_actual_z = float(world_up @ _point_velocity(twist_measured, lever))
            twist_commanded = action if is_ee else jacobian @ action
            v_commanded_z = float(world_up @ _point_velocity(twist_commanded, lever))

            if v_commanded_z >= -WORKTABLE_VELOCITY_EPS:
                out[arm] = action
                continue

            v_target_z = max(v_commanded_z, -v_envelope)
            if -v_actual_z > v_envelope:
                v_target_z = max(v_target_z, 0.0)

            if v_target_z == v_commanded_z:
                out[arm] = action
                continue

            if is_ee:
                # Minimum-norm correction along world-up (unit), so only the
                # world-vertical component of the twist changes.
                action = action.copy()
                action[:3] += (v_target_z - v_commanded_z) * world_up
            else:
                # Uniform scale preserves joint-space direction while braking descent.
                scale = max(0.0, v_target_z / v_commanded_z)
                action = action * scale

            out[arm] = action
        return out
