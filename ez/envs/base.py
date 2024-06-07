import gym
import numpy as np

from ez.utils.format import arr_to_str


class BaseWrapper(gym.Wrapper):
    def __init__(self, env, obs_to_string, clip_reward):
        """Cosine Consistency loss function: similarity loss
        Parameters
        ----------
        obs_to_string: bool. Convert the observation to jpeg string if True, in order to save memory usage.
        """
        super().__init__(env)
        self.obs_to_string = obs_to_string
        self.clip_reward = clip_reward

    def format_obs(self, obs):
        if self.obs_to_string:
            # convert obs to jpeg string for lower memory usage
            obs = obs.astype(np.uint8)
            obs = arr_to_str(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # format observation
        obs = self.format_obs(obs)

        info['raw_reward'] = reward
        if self.clip_reward:
            reward = np.sign(reward)

        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # format observation
        obs = self.format_obs(obs)

        return obs

    def close(self):
        return self.env.close()
