
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from LargestFirst import LargestFirst
from RandomDyna import RandomDyna
from DjikstraFD import DjikstraFD
from SuccessorRepresentationFD import SuccessorRepresentationFD

from maze import setup_env_9x6     
from utils import moyenne_par_indice        
from scipy.interpolate import interp1d     

import os       
from omegaconf import OmegaConf

# Load YAML config file as DictConfig
config = OmegaConf.load("src/config.yaml")

#Create result directory
if not os.path.exists('src/res/test'):
    os.makedirs('src/res/test')
output_path = 'src/res/test/figuretest.png'

env = setup_env_9x6()

#largest first
all_steps_lg = []
all_backups_lg = []
#random dyna
all_steps_rd = []
all_backups_rd = []
#djikstra focused dyna
all_steps_dfd = []
all_backups_dfd = []
#successor representation focused dyna
all_steps_srfd = []
all_backups_srfd = []

nb_exec = config.test.nb_execution

for i in range(nb_exec):
    QueueDyna = LargestFirst(env, config.test.alpha, config.test.delta, 
                             config.test.epsilon,config.test.max_step, 
                             config.test.render, config.test.nb_episode)
    QueueDyna.execute()
    print(i)
    data = pd.read_csv("executionInformation.csv")

    nb_steps = data.iloc[:, 1].tolist()
    nb_backup = data.iloc[:,0].tolist()
    all_steps_lg.append(nb_steps[:-2])
    all_backups_lg.append(nb_backup[:-2])

    RDyna = RandomDyna(env, config.test.alpha, config.test.delta, 
                       config.test.epsilon,2, config.test.render, 
                       config.test.nb_episode)
    RDyna.execute()
    print(i)
    data = pd.read_csv("executionInformation.csv")
    nb_steps =data.iloc[:, 1].tolist()
    nb_backup =data.iloc[:,0].tolist()
    all_steps_rd.append(nb_steps)
    all_backups_rd.append(nb_backup)

    Djikstra = DjikstraFD(env, config.test.alpha, config.test.delta, 
                          config.test.epsilon,2, config.test.render, 
                          config.test.nb_episode)
    Djikstra.execute()
    data = pd.read_csv("executionInformation.csv")
    print(i)
    nb_steps = data.iloc[:, 1].tolist()
    nb_backup = data.iloc[:,0].tolist()
    all_steps_dfd.append(nb_steps)
    all_backups_dfd.append(nb_backup)


    SR = SuccessorRepresentationFD(env, config.test.alpha,config.test.delta, 
                                                          config.test.epsilon, config.test.nb_episode,
                                                          config.test.max_step, config.test.train_episode_length, 
                                                          config.test.test_episode_length)
    SR.execute()
    data = pd.read_csv("executionInformation.csv")
    print(i)
    nb_steps = data.iloc[:, 1].tolist()
    nb_backup = data.iloc[:,0].tolist()
    all_steps_srfd.append(nb_steps)
    all_backups_srfd.append(nb_backup)

    
plt.plot(moyenne_par_indice(all_backups_lg), moyenne_par_indice(all_steps_lg), color='red', linewidth=2, label = f"Largest First nb_episode/execution = {QueueDyna.episode}")

plt.plot(moyenne_par_indice(all_backups_rd), moyenne_par_indice(all_steps_rd) ,color='blue', linewidth=2, label = f"Random Dyna nb_episode/execution = {RDyna.episode}")

plt.plot(moyenne_par_indice(all_backups_dfd), moyenne_par_indice(all_steps_dfd), color='green', linewidth=2, label = f"FocusedDyna avec Djikstra nb_episode/execution = {Djikstra.episode}")


plt.plot(moyenne_par_indice(all_backups_srfd), moyenne_par_indice(all_steps_srfd), color='orange', linewidth=2, label = f"FocusedDyna avec Successor Representation nb_episode/execution = {SR.episode}")



plt.title(f'Courbe du nombre de step to goal en fonction du nombre de backup moyenne sur {nb_exec} executions ')
plt.xlabel('nb_backup')
plt.ylabel('nb_step')
plt.xscale('log')
plt.legend(loc='best')
plt.grid(True)
plt.savefig(output_path)

