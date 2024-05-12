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
sns.set_theme(style="whitegrid")


# Load YAML config file as DictConfig
config = OmegaConf.load("setup/config.yaml")

#Create result directory
if not os.path.exists('res'):
    os.makedirs('res')
output_path = 'res/figure-8.png'

env = setup_env_18x12()
laby = "18x12"

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
                                    config.main.epsilon, config.sr.nb_episode,
                                    config.main.max_step, config.sr.env9x6.train_episode_length, 
                                    config.sr.env18x12.test_episode_length)
    SR.execute()
    data = pd.read_csv("executionInformation.csv")
    print(i)
    nb_steps = data.iloc[:, 1].tolist()
    nb_backup = data.iloc[:,0].tolist()
    all_steps_srfd.append(nb_steps)
    all_backups_srfd.append(nb_backup) 



# Calcul de la variance
var_lg = np.var(moyenne_par_indice(all_steps_lg))
var_rd = np.var(moyenne_par_indice(all_steps_rd))
var_dfd = np.var(moyenne_par_indice(all_steps_dfd))
var_srfd = np.var(moyenne_par_indice(all_steps_srfd))


# FIGURES
plt.figure(figsize=(15,10))

sns.lineplot(x=moyenne_par_indice(all_backups_lg), y=moyenne_par_indice(all_steps_lg), color='red', linewidth=2, label=f"LF {QueueDyna.episode} episodes")

sns.lineplot(x=moyenne_par_indice(all_backups_rd), y=moyenne_par_indice(all_backups_rd), color='blue', linewidth=2, label=f"LF {RDyna.episode} episodes")

sns.lineplot(x=moyenne_par_indice(all_backups_dfd), y=moyenne_par_indice(all_backups_dfd), color='green', linewidth=2, label=f"LF {Djikstra.episode} episodes")

sns.lineplot(x=moyenne_par_indice(all_backups_srfd), y=moyenne_par_indice(all_backups_srfd), color='orange', linewidth=2, label=f"LF {SR.episode} episodes",linestyle="--")

#plt.plot(moyenne_par_indice(all_backups_lg), moyenne_par_indice(all_steps_lg), color='red', linewidth=2, label = f"LF {QueueDyna.episode} episodes")

#plt.plot(moyenne_par_indice(all_backups_rd), moyenne_par_indice(all_steps_rd) ,color='blue', linewidth=2, label = f"RD {RDyna.episode} episodes")

#plt.plot(moyenne_par_indice(all_backups_dfd), moyenne_par_indice(all_steps_dfd), color='green', linewidth=2, label = f"DFD {Djikstra.episode} episodes")

#plt.plot(moyenne_par_indice(all_backups_srfd), moyenne_par_indice(all_steps_srfd), color='orange', linewidth=2, label = f"SRFC {SR.episode} episodes",linestyle="--")



plt.text(0.2,0.5, f" $\epsilon$ : {config.main.epsilon}\n $\delta$ : {config.main.delta}\n α : {config.main.alpha}\n $\gamma$ : {config.main.gamma}\n max_step : {config.main.max_step}\n nb_episode : {config.main.nb_episode}\n labyrinthe : {laby}\n var_lg = {var_lg}\n var_rd = {var_rd}\n var_dfd = {var_dfd}\n var_srfd = {var_srfd}", fontsize =11)

plt.title(f'Courbe du nombre de step to goal en fonction du nombre de backup moyenne sur {nb_exec} executions ')
plt.xlabel('nb_backup')
plt.ylabel('nb_step')
plt.xscale('log')
plt.legend(loc='center')
plt.grid(True)
plt.savefig(output_path)


