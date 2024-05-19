import gymnasium as gym
from assets.RewardWrapper import RewardWrapper
from omegaconf import OmegaConf

import sys
sys.path.append('../')   

# Load YAML config file as DictConfig
config = OmegaConf.load("setup/config.yaml")

def setup_env_9x6():
    env_9x6 = gym.make("MazeMDP-v0", kwargs={"width": 9, "height": 6,"start_states": [2],"walls": [13, 14, 15, 34, 42, 43, 44],
                        "terminal_states": [41]}, render_mode="rgb_array", gamma= config.main.gamma)
    env_9x6.unwrapped.terminal_states = [41]
    env_9x6.metadata['render_fps'] = 1
    env_9x6 = RewardWrapper(env_9x6)

    env_9x6.reset()

    env_9x6.set_no_agent()

    return env_9x6

def setup_env_18x12() :
    env_18x12 = gym.make("MazeMDP-v0", kwargs={"width": 18, "height": 12,
    "start_states": [4], "walls": [50,51,52,53,54,62,63,64,65,66, 128,129,140,141,168,169,170,171,172,173,180,181,182,183,184,185],
    "terminal_states": [166,167,178,179]}, render_mode="rgb_array", gamma= config.main.gamma)
    env_9x6.unwrapped.terminal_states = [166,167,178,179]

    env_18x12.metadata['render_fps'] = 1
    env_18x12 = RewardWrapper(env_18x12)

    env_18x12.reset()

    env_18x12.unwrapped.set_no_agent()
    return env_18x12

def setup_env_36x24() :
    env_36x24 = gym.make("MazeMDP-v0", kwargs={"width": 36, "height": 24,
    "start_states": [8], "walls": [196,197,198,199,200,201,202,203,204,205,206,207
                                    ,220,221,222,223,224,225,226,227,228,229,230,231
                                    ,244,245,246,247,248,249,250,251,252,253,254,255
                                    ,268,269,270,271,272,273,274,275,276,277,278,279
                                    ,496,497,498,499
                                    ,520,521,522,523
                                    ,544,545,546,547
                                    ,568,569,570,571
                                    ,672,673,674,675,676,677,678,679,680,681,682,683
                                    ,696,697,698,699,700,701,702,703,704,705,706,707
                                    ,720,721,722,723,724,725,726,727,728,729,730,731
                                    ,744,745,746,747,748,749,750,751,752,753,754,755
                                    ],
    "terminal_states": [656,657,658,659
                        ,680,681,682,683
                        ,704,705,706,707
                        ,728,729,730,731]},
    render_mode="rgb_array", gamma=config.main.gamma)
    env_9x6.unwrapped.terminal_states = [656,657,658,659
                                        ,680,681,682,683
                                        ,704,705,706,707
                                        ,728,729,730,731]

    env_36x24.metadata['render_fps'] = 1
    env_36x24 = RewardWrapper(env_36x24)
    env_36x24.reset()

    env_36x24.unwrapped.set_no_agent()
    return env_36x24
