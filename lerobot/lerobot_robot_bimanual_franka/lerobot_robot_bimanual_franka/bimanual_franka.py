from lerobot.robots import Robot
from lerobot.types import RobotAction, RobotObservation

from .bimanual_franka_config import BimanualFrankaConfig
from .franka_process import MultiRobotWrapper
from .wsg import WSG

import numpy as np

class BimanualFranka(Robot):
    config_class = BimanualFrankaConfig
    name = "my_cool_robot"

    def __init__(self, config: BimanualFrankaConfig):
        super().__init__(config)
        self.config = config
        self.use_ee_delta = self.config.use_ee_delta
        self.active_arms = self.config.active_arms

        self.robot_manager = MultiRobotWrapper()
        self.grippers: dict[str, WSG] = {}
        if "l" in self.active_arms:
            self.grippers["l"] = WSG(TCP_IP=self.config.l_gripper_ip)
        if "r" in self.active_arms:
            self.grippers["r"] = WSG(TCP_IP=self.config.r_gripper_ip)
        
        # self.bus = FeetechMotorsBus(
        #     port=self.config.port,
        #     motors={
        #         "joint_1": Motor(1, "sts3250", MotorNormMode.RANGE_M100_100),
        #         "joint_2": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
        #         "joint_3": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
        #         "joint_4": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
        #         "joint_5": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
        #     },
        #     calibration=self.calibration,
        # )
        # self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        if self.use_ee_delta:
            features: dict[str, type] = {}
            for s in self.active_arms:
                features[f"{s}_x"] = float
                features[f"{s}_y"] = float
                features[f"{s}_z"] = float
                features[f"{s}_roll"] = float
                features[f"{s}_pitch"] = float
                features[f"{s}_yaw"] = float
                features[f"{s}_gripper"] = float
            return features

        return self._motors_ft_joints
    
    @property
    def _motors_ft_joints(self) -> dict[str, type]:
        features: dict[str, type] = {}
        for s in self.active_arms:
            features[f"{s}_joint_1"] = float
            features[f"{s}_joint_2"] = float
            features[f"{s}_joint_3"] = float
            features[f"{s}_joint_4"] = float
            features[f"{s}_joint_5"] = float
            features[f"{s}_joint_6"] = float
            features[f"{s}_joint_7"] = float
            features[f"{s}_gripper"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            # cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict:
        return {**self._motors_ft_joints, **self._cameras_ft}
    
    @property
    def action_features(self) -> dict:
        return self._motors_ft
    
    @property
    def is_connected(self) -> bool:
        # return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())
        return self.robot_manager.num_processes == len(self.active_arms)
    
    def connect(self, calibrate: bool = True) -> None:
        if "l" in self.active_arms:
            self.robot_manager.add_robot("l", self.config.l_server_ip, self.config.l_robot_ip, self.config.l_port)
        if "r" in self.active_arms:
            self.robot_manager.add_robot("r", self.config.r_server_ip, self.config.r_robot_ip, self.config.r_port)

        # Try to confirm at least one robot is alive by querying state early
        import time
        time.sleep(1.0)  # Give processes time to initialize
        for arm in self.active_arms:
            try:
                self.robot_manager.current_joint_positions(arm)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to communicate with robot '{arm}' at {getattr(self.config, f'{arm}_robot_ip')}: {e}"
                )

        if not self.is_calibrated and calibrate:
            self.calibrate()

        # for cam in self.cameras.values():
        #     cam.connect()

        self.configure()
        for s in self.active_arms:
            self.grippers[s].home()

    def disconnect(self) -> None:
        # self.bus.disconnect()
        # for cam in self.cameras.values():
        #     cam.disconnect()
        self.robot_manager.shutdown()
        for gripper in self.grippers.values():
            gripper.close()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        # with self.bus.torque_disabled():
        #     self.bus.configure_motors()
        #     for motor in self.bus.motors:
        #         self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        #         self.bus.write("P_Coefficient", motor, 16)
        #         self.bus.write("I_Coefficient", motor, 0)
        #         self.bus.write("D_Coefficient", motor, 32)
        # self.robot.relative_dynamics_factor = 0.05
        pass

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        # Read arm position
        # obs_dict: RobotObservation = self.bus.sync_read("Present_Position")
        obs: RobotObservation = {}
        for s in self.active_arms:
            joints: np.ndarray = self.robot_manager.current_joint_positions(s)
            obs_dict: RobotObservation = {f"{i}": joints[i - 1] for i in range(1, len(joints) + 1)}
            obs_dict = {f"{s}_joint_{motor}": val for motor, val in obs_dict.items()}
            obs_dict[f"{s}_gripper"] = self.grippers[s].position
            obs = {**obs, **obs_dict}

        # Capture images from cameras
        # for cam_key, cam in self.cameras.items():
        #     obs_dict[cam_key] = cam.async_read()

        return obs
    
    def send_action(self, action: RobotAction) -> RobotAction:
        # goal_pos: RobotAction = {key.removesuffix(".pos"): val for key, val in action.items()}

        # Send goal position to the arm
        # self.bus.sync_write("Goal_Position", goal_pos)    
        # 
        for s in self.active_arms:
            self.grippers[s].move(action[f"{s}_gripper"], blocking=False)

            if self.use_ee_delta:
                ee: list = [
                    action[f"{s}_x"],
                    action[f"{s}_y"],
                    action[f"{s}_z"],
                    action[f"{s}_roll"],
                    action[f"{s}_pitch"],
                    action[f"{s}_yaw"],
                    ]
                self.robot_manager.move_ee_delta(s, ee, True)
                continue

            joints: list = [action[f"{s}_joint_{i}"] for i in range(1,8)]
            self.robot_manager.move_joints(s, joints, True)

        return action
