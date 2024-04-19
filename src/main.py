import gymnasium as gym
import bbrl_gymnasium
from moviepy.editor import ipython_display as video_display
from RewardWrapper import RewardWrapper
from PrioritizedReplayAgent import PrioritizedReplayAgent
import matplotlib.pyplot as plt

from LargestFirst import LargestFirst
from RandomDyna import RandomDyna
from FocusedDyna import FocusedDyna
from SuccessorRepresentation import FocusedDynaSR

import pandas as pd
import numpy as np

from omegaconf import OmegaConf
# Load YAML config file as DictConfig
config = OmegaConf.load("config.yaml")


def main() :

    env = gym.make("MazeMDP-v0", kwargs={"width": 18, "height": 12, "start_states": [4], "walls": [50,51,52,53,54,62,63,64,65,66, 128,129,140,141,168,169,170,171,172,173,180,181,182,183,184,185],"terminal_states": [166,167,178,179]}, render_mode="rgb_array", gamma=0.9)

    QueueDyna = LargestFirst(env, config.main.alpha, config.main.delta, config.main.epsilon,config.main.max_step, config.main.render, config.main.nb_episode)
    QueueDyna.execute()

    data = pd.read_csv("executionInformation.csv")

    nb_steps = data.iloc[:, 1].tolist()
    nb_backup = data.iloc[:,0].tolist()

    plt.plot(nb_backup,nb_steps)


    plt.title('Courbe du nombre de step to goal en fonction du nombre de backup')
    plt.xlabel('nb_backup')
    plt.ylabel('nb_step')
    plt.xscale('log')
    plt.grid(True)
    plt.show()