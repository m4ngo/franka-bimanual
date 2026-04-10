from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots import Robot
from lerobot.types import RobotAction, RobotObservation
from .my_cool_robot_config import MyCoolRobotConfig

class MyCoolRobot(Robot):
    config_class = MyCoolRobotConfig
    name = "my_cool_robot"

    def __init__(self, config: MyCoolRobotConfig):
        super().__init__(config)
        self.config = config
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "joint_1": Motor(1, "sts3250", MotorNormMode.RANGE_M100_100),
                "joint_2": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "joint_3": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "joint_4": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "joint_5": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            "joint_1.pos": float,
            "joint_2.pos": float,
            "joint_3.pos": float,
            "joint_4.pos": float,
            "joint_5.pos": float,
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}
    
    @property
    def action_features(self) -> dict:
        return self._motors_ft
    
    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())
    
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

    def disconnect(self) -> None:
        self.bus.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                self.bus.write("P_Coefficient", motor, 16)
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected.")

        # Read arm position
        obs_dict: RobotObservation = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read()

        return obs_dict
    
    def send_action(self, action: RobotAction) -> RobotAction:
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items()}

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)

        return action
    