import gymnasium as gym
import bbrl_gymnasium
# from moviepy.editor import ipython_display as video_display
from RewardWrapper import RewardWrapper
from PrioritizedReplayAgent import PrioritizedReplayAgent
import matplotlib.pyplot as plt

from LargestFirst import LargestFirst
from RandomDyna import RandomDyna
from FocusedDyna import FocusedDyna
from SuccessorRepresentation import FocusedDynaSR

import pandas as pd
import numpy as np

from scipy.interpolate import interp1d

from omegaconf import OmegaConf
# Load YAML config file as DictConfig
config = OmegaConf.load("config.yaml")

epsilon = config.main.epsilon  # parametres pour gerer l'exploration dans epsilongreedy
delta = config.main.delta  #treshold
gamma = 0.95  #discount factor
alpha = config.main.alpha   #learning rate
max_step = config.main.max_step #nombre de pas maximum pour un episode
nb_episode = config.main.nb_episode
render  = config.main.render

# environnement 9x6 
env_9x6 = gym.make("MazeMDP-v0", kwargs={"width": 9, "height": 6,
"start_states": [2], "walls": [13, 14, 15, 34, 42, 43, 44],
"terminal_states": [41]}, render_mode="rgb_array", gamma=gamma)

env_9x6.metadata['render_fps'] = 1
env_9x6 = RewardWrapper(env_9x6)
env_9x6.reset()

env_9x6.set_no_agent()
#env_9x6.init_draw("The maze 9x6")

# environnement 18x12
env_18x12 = gym.make("MazeMDP-v0", kwargs={"width": 18, "height": 12,
"start_states": [4], "walls": [50,51,52,53,54,62,63,64,65,66, 128,129,140,141,168,169,170,171,172,173,180,181,182,183,184,185],
"terminal_states": [166,167,178,179]}, render_mode="rgb_array", gamma=gamma)

env_18x12.metadata['render_fps'] = 1
env_18x12 = RewardWrapper(env_18x12)
env_18x12.reset()

env_18x12.set_no_agent()
#env_18x12.init_draw("The maze 18x12")


print("start")

fdsr = FocusedDynaSR(env_18x12, alpha,delta, epsilon, config.sr.nb_episode,config.main.max_step, config.sr.small.train_episode_length, config.sr.small.test_episode_length)

def moyenne_par_indice(liste):
    tableau = np.array(liste)
    moyennes = np.mean(tableau, axis=0)
    moyennes = np.where(np.isnan(moyennes), None, moyennes)
    return moyennes.tolist()

all_steps_lg = []
all_backups_lg = []

all_steps_rd = []
all_backups_rd = []

all_steps_fc = []
all_backups_fc = []

all_steps_sr = []
all_backups_sr = []

nb_exec = 2

print("loop")


for i in range(nb_exec):
    QueueDyna = LargestFirst(env_18x12, alpha, delta, epsilon,max_step, render, nb_episode)
    QueueDyna.execute()
    print(i)
    data = pd.read_csv("executionInformation.csv")

    nb_steps = data.iloc[:, 1].tolist()
    nb_backup = data.iloc[:,0].tolist()
    all_steps_lg.append(nb_steps[:-2])
    all_backups_lg.append(nb_backup[:-2])

    RDyna = RandomDyna(env_18x12, alpha, delta, epsilon,max_step, render, nb_episode)
    print(i)
    data = pd.read_csv("executionInformation.csv")
    nb_steps =data.iloc[:, 1].tolist()
    nb_backup =data.iloc[:,0].tolist()
    all_steps_rd.append(nb_steps)
    all_backups_rd.append(nb_backup)

    FDyna = FocusedDyna(env_18x12, alpha, delta, epsilon,max_step, render, nb_episode)
    FDyna.execute()
    data = pd.read_csv("executionInformation.csv")
    print(i)
    nb_steps = data.iloc[:, 1].tolist()
    nb_backup = data.iloc[:,0].tolist()
    all_steps_fc.append(nb_steps)
    all_backups_fc.append(nb_backup)


    
    fdsr.execute()
    data = pd.read_csv("executionInformation.csv")
    print(i)
    nb_steps = data.iloc[:, 1].tolist()
    nb_backup = data.iloc[:,0].tolist()
    all_steps_sr.append(nb_steps)
    all_backups_sr.append(nb_backup)

    
plt.figure(figsize=(10,5))
plt.plot(moyenne_par_indice(all_backups_lg), moyenne_par_indice(all_steps_lg), color='red', linewidth=2, label = f"Largest First nb_episode/execution = {QueueDyna.nb_episode}")

plt.plot(moyenne_par_indice(all_backups_rd), moyenne_par_indice(all_steps_rd) ,color='blue', linewidth=2, label = f"Random Dyna nb_episode/execution = {RDyna.nb_episode}")

plt.plot(moyenne_par_indice(all_backups_fc), moyenne_par_indice(all_steps_fc), color='green', linewidth=2, label = f"FocusedDyna nb_episode/execution = {FDyna.nb_episode}")


plt.plot(moyenne_par_indice(all_backups_sr), moyenne_par_indice(all_steps_sr), color='yellow', linewidth=2, label = f"FocusedDyna Successor Representation nb_episode/execution = {fdsr.nb_episode}")


print("end")
plt.title(f'Courbe du nombre de step to goal en fonction du nombre de backup moyenne sur {nb_exec} executions ')
plt.xlabel('nb_backup')
plt.ylabel('nb_step')
plt.xscale('log')
plt.legend(loc='best')
plt.grid(True)
#plt.savefig("img/fig-8.png")
plt.show()