"""The one workstation-side screen: a hard floor under the commanded OSC goal.

`shape_goal` raises a goal position along world-up until the EE collision
sphere's lowest point sits at or above `worktable.height_m + distance_min_m`.
That is the whole layer. It is a pure (goal, pose) -> goal transform with no
dependence on gains, measured velocity, or the previous step, so it cannot
interact with a tuning sweep: changing kp/kd cannot move the floor.

It reasons in WORLD frame. The table is one plane shared by both arms, so each
arm's goal is lifted through that arm's `robot_base_in_world` before being
compared against it; arms whose bases sit at different heights or orientations
get the right floor with no per-arm threshold.

The EE is treated as a SPHERE, not a point. Its centre is given in the TOOL
frame and rotates with the gripper, and clearance is measured from the sphere's
lowest point, so a tilted gripper cannot graze the table with its side while the
TCP still reads as clear.

Everything that reaches the joints is bounded on the NUC instead, by the torque
rate limit and clamp in `pylibfranka_control`. There is no velocity envelope and
no joint-space form here; JOINT_POS and `home()` drive saved configurations and
are not screened.

Bimanual arm-repel is not implemented.
"""

import franka_config as fc  # type: ignore
import numpy as np

WORKTABLE_HEIGHT = fc.worktable_height_m()                             # m, WORLD-frame Z of the worktable surface
EE_SPHERE = fc.default_ee_sphere()                                     # shared EE collision sphere; arms may override
WORKTABLE_DISTANCE_MIN = fc.control("worktable_brake.distance_min_m")  # m, minimum clearance above the surface


def _quat_xyzw_to_matrix(q_xyzw) -> np.ndarray:
    """Rotation matrix from an xyzw quaternion (the FR3 state's convention)."""
    return fc.quat_wxyz_to_matrix(fc.quat_xyzw_to_wxyz(tuple(np.asarray(q_xyzw, dtype=np.float64))))


class ActionSafetyScreen:
    """Clamps EE goal poses so the gripper cannot be commanded into the table.

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

    @property
    def goal_z_floor(self) -> float:
        """Lowest WORLD-frame Z the EE collision sphere's bottom may be commanded to.

        A plane, so it is arm-independent — the per-arm part (base pose, sphere
        radius) lives in `sphere_bottom_world_z`, which is what this is compared
        against. It is NOT a base-frame goal-position bound.
        """
        return WORKTABLE_HEIGHT + WORKTABLE_DISTANCE_MIN

    def sphere_bottom_world_z(self, arm: str, ee_translation, ee_quat_xyzw) -> float:
        """World Z of the EE collision sphere's lowest point for a base-frame pose.

        The centre is defined in the TOOL frame, so it is rotated by the given EE
        orientation before being lifted into world — that is what keeps the
        clearance honest when the gripper is tilted. The sphere is rotationally
        symmetric, so its lowest point is always centre_z - radius.
        """
        try:
            pose = self._base_in_world[arm]
        except KeyError:
            raise KeyError(
                f"no robot_base_in_world pose for arm {arm!r}; known: "
                f"{sorted(self._base_in_world)}. The worktable floor is world-frame "
                "and cannot screen an arm whose base pose is unknown."
            ) from None
        center_tool = self._sphere_center_tool[arm]
        center_base = np.asarray(ee_translation, dtype=np.float64)
        if center_tool.any():
            center_base = center_base + _quat_xyzw_to_matrix(ee_quat_xyzw) @ center_tool
        return float(pose.apply(center_base)[2]) - self._ee_spheres[arm].radius_m

    def shape_goal(
        self,
        goals_by_arm: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Raise each EE goal along world-up until its collision sphere clears the
        floor. Orientation goals pass through untouched.

        The sphere is evaluated at the COMMANDED orientation, since that is where
        it will be, and the correction is applied along world-up only, so
        horizontal motion is unaffected and an arm whose base is tilted or raised
        gets the right floor without a per-arm threshold.
        """
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for arm, (goal_pos, goal_quat) in goals_by_arm.items():
            goal_pos = np.asarray(goal_pos, dtype=np.float64)
            rise = self.goal_z_floor - self.sphere_bottom_world_z(arm, goal_pos, goal_quat)
            if rise > 0.0:
                # world_up is unit, so this raises the world Z by exactly `rise`.
                goal_pos = goal_pos + rise * self._world_up_in_base[arm]
            out[arm] = (goal_pos, goal_quat)
        return out
