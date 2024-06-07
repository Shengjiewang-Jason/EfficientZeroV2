from ..base import BaseWrapper


class DMCWrapper(BaseWrapper):
    """
    Make your own wrapper: DMC Wrapper
    """
    def __init__(self, env, obs_to_string=False, clip_reward=False):
        super().__init__(env, obs_to_string, clip_reward)

