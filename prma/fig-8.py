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
output_path = 'res/figure-8-2.png'

env = setup_env_36x24()
laby = "36x24"

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
for i in range(nb_exec):
    QueueDyna = LargestFirst(env, config.main.alpha, config.main.delta, 
                             config.main.epsilon,config.main.max_step, 
                             config.main.render, config.main.nb_episode)
    QueueDyna.execute()
    print(i)
    data = pd.read_csv("executionInformation.csv")

    nb_steps = data.iloc[:, 1].tolist()
    nb_backup = data.iloc[:,0].tolist()
    all_steps_lg.append(nb_steps)
    all_backups_lg.append(nb_backup)

    RDyna = RandomDyna(env, config.main.alpha, config.main.delta, 
                       config.main.epsilon,config.main.max_step, config.main.render, 
                       config.main.nb_episode)
    RDyna.execute()
    print(i)
    data = pd.read_csv("executionInformation.csv")
    nb_steps =data.iloc[:, 1].tolist()
    nb_backup =data.iloc[:,0].tolist()
    all_steps_rd.append(nb_steps)
    all_backups_rd.append(nb_backup)

    Djikstra = DjikstraFD(env, config.main.alpha, config.main.delta, 
                            config.main.epsilon,config.main.max_step, 
                            config.main.render, config.main.nb_episode)
    Djikstra.execute()
    data = pd.read_csv("executionInformation.csv")
    print(i)
    nb_steps = data.iloc[:, 1].tolist()
    nb_backup = data.iloc[:,0].tolist()
    all_steps_dfd.append(nb_steps)
    all_backups_dfd.append(nb_backup)

    SR = SuccessorRepresentationFD(env, config.main.alpha,config.main.delta, 
                                    config.main.epsilon,config.main.nb_episode,
                                    config.main.max_step, config.sr.env18x12.train_episode_length, 
                                    config.sr.env18x12.test_episode_length)
    SR.execute()
    data = pd.read_csv("executionInformation.csv")
    print(i)
    nb_steps = data.iloc[:, 1].tolist()
    nb_backup = data.iloc[:,0].tolist()
    all_steps_srfd.append(nb_steps)
    all_backups_srfd.append(nb_backup)




# FIGURES
plt.figure(figsize=(15,10))

sns.set_theme(style="whitegrid")

sns.lineplot(x=moyenne_par_indice(all_backups_lg), y=moyenne_par_indice(all_steps_lg),label = f"LF {QueueDyna.episode} episodes",errorbar='sd',err_style='band')

sns.lineplot(x=moyenne_par_indice(all_backups_rd), y=moyenne_par_indice(all_steps_rd),label = f"RD {RDyna.episode} episodes",errorbar='sd',err_style='band')

sns.lineplot(x=moyenne_par_indice(all_backups_dfd), y=moyenne_par_indice(all_steps_dfd),label = f"DFD {Djikstra.episode} episodes",errorbar='sd',err_style='band')

sns.lineplot(x=moyenne_par_indice(all_backups_srfd), y=moyenne_par_indice(all_steps_srfd),label = f"SRFD {SR.episode} episodes",errorbar='sd',linestyle="--",err_style='band')

#plt.text(0.2,0.5, f" $\epsilon$ : {config.main.epsilon}\n $\delta$ : {config.main.delta}\n α : {config.main.alpha}\n $\gamma$ : {config.main.gamma}\n max_step : {config.main.max_step}\n nb_episode : {config.main.nb_episode}\n labyrinthe : {laby}", fontsize =11)

plt.title(f'Courbe du nombre de step to goal en fonction du nombre de backup moyenne sur {nb_exec} executions ')
plt.xlabel('nb_backup')
plt.ylabel('nb_step')
plt.xscale('log')
plt.legend(loc='center')
plt.grid(True)
plt.savefig(output_path)



print("Variance LG :",np.var(moyenne_par_indice(all_steps_lg)))
print("Variance RD :",np.var(moyenne_par_indice(all_steps_rd)))
print("Variance DFD :",np.var(moyenne_par_indice(all_steps_dfd)))
print("Variance SRFD :",np.var(moyenne_par_indice(all_steps_srfd)))
