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

        self.robot_manager = MultiRobotWrapper()
        self.grippers: dict[str, WSG] = {}
        self.grippers["l"] = WSG(TCP_IP=self.config.l_gripper_ip)
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
            return {
            "l_x": float,
            "l_y": float,
            "l_z": float,
            "l_roll": float,
            "l_pitch": float,
            "l_yaw": float,
            "l_gripper": float,

            "r_x": float,
            "r_y": float,
            "r_z": float,
            "r_roll": float,
            "r_pitch": float,
            "r_yaw": float,
            "r_gripper": float,
            }

        return self._motors_ft_joints
    
    @property
    def _motors_ft_joints(self) -> dict[str, type]:
        return {
            "l_joint_1": float,
            "l_joint_2": float,
            "l_joint_3": float,
            "l_joint_4": float,
            "l_joint_5": float,
            "l_joint_6": float,
            "l_joint_7": float,
            "l_gripper": float,
            
            "r_joint_1": float,
            "r_joint_2": float,
            "r_joint_3": float,
            "r_joint_4": float,
            "r_joint_5": float,
            "r_joint_6": float,
            "r_joint_7": float,
            "r_gripper": float,
        }

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
        return self.robot_manager.num_processes == 2
    
    def connect(self, calibrate: bool = True) -> None:
        self.robot_manager.add_robot("l", self.config.l_server_ip, self.config.l_robot_ip, self.config.l_port)
        self.robot_manager.add_robot("r", self.config.r_server_ip, self.config.r_robot_ip, self.config.r_port)

        if not self.is_calibrated and calibrate:
            self.calibrate()

        # for cam in self.cameras.values():
        #     cam.connect()

        self.configure()

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
        for s in ["l", "r"]:
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
        for s in ["l", "r"]:    
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
