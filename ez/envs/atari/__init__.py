from ..base import BaseWrapper


class AtariWrapper(BaseWrapper):
    """
    Make your own wrapper: Atari Wrapper
    """
    def __init__(self, env, obs_to_string=False, clip_reward=False):
        super().__init__(env, obs_to_string, clip_reward)
