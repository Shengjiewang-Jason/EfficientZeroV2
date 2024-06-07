from ..base import BaseWrapper


class GymWrapper(BaseWrapper):
    """
    Make your own wrapper: Atari Wrapper
    """
    def __init__(self, env, obs_to_string=False):
        super().__init__(env, obs_to_string, False)

    def step(self, action):
        obs, reward, _, done, info = self.env.step(action)
        info['raw_reward'] = reward
        return obs, reward, done, info

    def reset(self,):
        obs, info = self.env.reset()

        return obs