import gymnasium as gym



class RewardWrapper(gym.Wrapper):
    """
    Wrapper specific pour la reproduction des environnements de l'article de Peng and Williams
    """

    def __init__(self, env):
        super(RewardWrapper, self).__init__(env)

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        # reward shaping
        if reward:
            reward = reward * 100
        return next_state, reward, terminated, truncated, info