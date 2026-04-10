from typing import Optional
import numpy as np
import gymnasium as gym


class FrankaBimanualEnv(gym.Env):

    def __init__(self):
        # observation space: 2x7 ? each vector is delta EE position + 1 claw pos
        # action space: 2x7 ? same as above?
        pass

    def _get_obs(self):
        """Convert internal state to observation format
        """
        pass

    def _get_info(self):
        """ Compute auxiliary information for debugging
        """
        pass

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        pass

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """

        # this is where you would control the two arms and implement
        # bimanual safety
        pass

    
