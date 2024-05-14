import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from assets.LargestFirst import LargestFirst
from assets.RandomDyna import RandomDyna
from assets.DjikstraFD import DjikstraFD
from assets.SuccessorRepresentationFD import SuccessorRepresentationFD

from assets.maze import setup_env_36x24, setup_env_9x6, setup_env_18x12 
from assets.utils import moyenne_par_indice        
from scipy.interpolate import interp1d       

import os       
from omegaconf import OmegaConf

import seaborn as sns


# Load YAML config file as DictConfig
config = OmegaConf.load("setup/config.yaml")

#Create result directory
if not os.path.exists('res'):
    os.makedirs('res')
output_path = 'res/figure-7.png'

env = [setup_env_9x6(), setup_env_18x12(), setup_env_36x24()]

terminal_states = [max(e.terminal_states) for e in env]

laby = "all"

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

nb_exec = config.main.nb_execution


# LOOP
all_backups_lg = []
all_backups_rd = []
all_backups_dfd = []
all_backups_srfd = []
for i in range(len(env)) :   
    e = env[i]
    print(all_backups_lg)
    print(all_backups_rd)
    print(all_backups_dfd)
    print(all_backups_srfd)

    QueueDyna = LargestFirst(e, config.main.alpha, config.main.delta, 
                             config.main.epsilon,config.main.max_step, 
                             config.main.render, config.main.nb_episode)
    QueueDyna.execute()
    data = pd.read_csv("executionInformation.csv")

    nb_backup = data.iloc[:,0].tolist() 
    nb_steps = data.iloc[:, 1].tolist()
    all_backups_lg.append(nb_backup[np.argmin(nb_steps)])
    print(i,nb_backup[np.argmin(nb_steps)],np.argmin(nb_steps))
    

    RDyna = RandomDyna(e, config.main.alpha, config.main.delta, 
                       config.main.epsilon,config.main.max_step, config.main.render, 
                       config.main.nb_episode)
    RDyna.execute()
    data = pd.read_csv("executionInformation.csv")
    nb_backup =data.iloc[:,0].tolist()    
    all_backups_rd.append(nb_backup[np.argmin(nb_steps)])
    print(i,nb_backup[np.argmin(nb_steps)],np.argmin(nb_steps))

    Djikstra = DjikstraFD(e, config.main.alpha, config.main.delta, 
                            config.main.epsilon,config.main.max_step, 
                            config.main.render, config.main.nb_episode)
    Djikstra.execute()
    data = pd.read_csv("executionInformation.csv")
    nb_backup =data.iloc[:,0].tolist()    
    all_backups_dfd.append(nb_backup[np.argmin(nb_steps)])
    print(i,nb_backup[np.argmin(nb_steps)],np.argmin(nb_steps))

    SR = SuccessorRepresentationFD(e, config.main.alpha,config.main.delta, 
                                    config.main.epsilon,config.main.nb_episode,
                                    config.main.max_step, config.sr.env18x12.train_episode_length, 
                                    config.sr.env18x12.test_episode_length)
    SR.execute()
    data = pd.read_csv("executionInformation.csv")
    nb_backup =data.iloc[:,0].tolist()    
    all_backups_srfd.append(nb_backup[np.argmin(nb_steps)])
    print(i,nb_backup[np.argmin(nb_steps)],np.argmin(nb_steps))




# FIGURES
plt.figure(figsize=(15,10))

sns.set_theme(style="whitegrid")

sns.lineplot(x=terminal_states, y=all_backups_lg,label = f"LF",errorbar='sd',err_style='band')

sns.lineplot(x=terminal_states, y=all_backups_rd,label = f"RD",errorbar='sd',err_style='band')

sns.lineplot(x=terminal_states, y=all_backups_dfd,label = f"DFD",errorbar='sd',err_style='band')

sns.lineplot(x=terminal_states, y=all_backups_srfd,label = f"SRFD",errorbar='sd',linestyle="--",err_style='band')


plt.title(f'Courbe du nombre de step to goal en fonction du nombre de backup moyenne sur {nb_exec} executions ')
plt.xlabel('No. States')
plt.ylabel('No. Backups Until Optimal Solution')
plt.xscale('log')
plt.legend(loc='center')
plt.grid(True)
plt.savefig(output_path)