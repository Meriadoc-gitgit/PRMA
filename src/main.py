import gymnasium as gym
import bbrl_gymnasium
from moviepy.editor import ipython_display as video_display
from RewardWrapper import RewardWrapper
from PrioritizedReplayAgent import PrioritizedReplayAgent
import matplotlib.pyplot as plt

from LargestFirst import LargestFirst
from RandomDyna import RandomDyna
from DjikstraFD import DjikstraFD
from SuccessorRepresentationFD import SuccessorRepresentationFD

from maze import setup_env_9x6

import pandas as pd
import numpy as np

from omegaconf import OmegaConf
# Load YAML config file as DictConfig
config = OmegaConf.load("config.yaml")


def main() :

    env = setup_env_9x6()

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