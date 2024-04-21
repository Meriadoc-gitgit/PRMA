import gymnasium as gym
from RewardWrapper import RewardWrapper
from omegaconf import OmegaConf
# Load YAML config file as DictConfig
config = OmegaConf.load("config.yaml")

def setup_env_9x6():
    env_9x6 = gym.make("MazeMDP-v0", kwargs={"width": 9, "height": 6,"start_states": [2],"walls": [13, 14, 15, 34, 42, 43, 44],
                        "terminal_states": [41]}, render_mode="rgb_array", gamma= config.main.gamma)
    env_9x6.metadata['render_fps'] = 1
    env_9x6 = RewardWrapper(env_9x6)

    env_9x6.reset()

    env_9x6.set_no_agent()

    return env_9x6

def setup_env_18x12() :
    env_18x12 = gym.make("MazeMDP-v0", kwargs={"width": 18, "height": 12,
    "start_states": [4], "walls": [50,51,52,53,54,62,63,64,65,66, 128,129,140,141,168,169,170,171,172,173,180,181,182,183,184,185],
    "terminal_states": [166,167,178,179]}, render_mode="rgb_array", gamma= config.main.gamma)

    env_18x12.metadata['render_fps'] = 1
    env_18x12 = RewardWrapper(env_18x12)

    env_18x12.reset()

    env_18x12.set_no_agent()
    return env_18x12
