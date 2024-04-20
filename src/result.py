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

from maze import setup_env_9x6, setup_env_18x12

import pandas as pd
import numpy as np

from omegaconf import OmegaConf
# Load YAML config file as DictConfig
config = OmegaConf.load("config.yaml")

"""
=========================== README ===========================

Afin de recevoir la visualisation des algorithmes, veuillez suivre les instructions affichées dans le Terminal. 

Pour assurer l'uniformité des programmes, veuillez vérifier que vous avez une version de Python convenable, dont Python 3.x. Veuillez exécuter la commande suivante dans le Terminal avant de commencer : `python3 -V`

==============================================================
"""

def main() : 
  """
  ==============================================================

  3 programmes à visualiser : 
    1. Largest First 
    2. Random Dyna
    3. Focused Dyna
      1. Avec la Successor Representation
      2. Avec Dijsktra

    Veuillez entrer le code indiqué ci-dessus afin d'attribuer l'algorithme à visualiser. 

  ==============================================================

  2 maze pour tester : 
    1. Maze de taille 9x6
    2. Maze de taille 18x12

  Veuillez entrer soit 1, soit 2 pour attribuer le maze voulu. 

  ==============================================================

  À noter : Chaque algo recoit son propre discount factor, afin de visualiser la différence de comportement des agents envers différent discount factor.

  ==============================================================
  """
  print("==============================================================\n\n3 programmes à visualiser : \n1. Largest First \n2. Random Dyna\n3. Focused Dyna\n\t1. Avec la Successor Representation\n\t2. Avec Dijsktra\nVeuillez entrer le code indiqué ci-dessus afin d'attribuer l'algorithme à visualiser. \n\n==============================================================\n\n2 maze pour tester : \n\t1. Maze de taille 9x6\n\t2. Maze de taille 18x12\nVeuillez entrer soit 1, soit 2 pour attribuer le maze voulu. \n\n==============================================================\n\nÀ noter : Chaque algo recoit son propre discount factor, afin de visualiser la différence de comportement des agents envers différent discount factor.\n\n==============================================================")

  program = input("Enter the name of the algorithm that you want to visualize : ")
  discount_factor = int(input("Enter the discount factor wanted (in %) : ")) / 100
  maze = input("Enter the maze size wanted : ")


  print("==============================================================\n")
  print("You have entered :\nProgramme :",program,"\nDiscount factor :",discount_factor,"\nMaze :",maze)
  print("\n==============================================================")

  # Attribuer le maze indiqué
  if int(maze) == 1 : 
    env = setup_env_9x6()
  elif int(maze) == 2 : 
    env = setup_env_18x12()

  # Algorithme
  if int(program) == 1 : 
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

  elif int(program) == 2 : 
    RDyna = RandomDyna(env, config.main.alpha, config.main.delta, config.main.epsilon,config.main.max_step, config.main.render, config.main.episode)
    RDyna.execute()

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
  else : 
    print("==============================================================\n")
    type_of_FD = input("Please enter the type of Focused Dyna wanted : ")
    print("You entered :",type_of_FD)
    print("\n==============================================================")

    if int(type_of_FD) == 1 : 
      fdsr = FocusedDynaSR(env, config.main.alpha, config.main.epsilon, config.sr.episode, config.sr.small.train_episode_length, config.sr.small.test_episode_length)
      fdsr.execute()
      print("Minimal length of path to goal :",fdsr.optimal_path_length())
      
      fig, (ax1, ax2) = plt.subplots(nrows=2)
      ax1.plot(fdsr.test_lengths)
      ax1.set_title("Successor Representation performance")

      ax2.plot(fdsr.lifetime_td_errors)
      ax2.set_title("Lifetime TD-error")
      
      plt.show()

  print("==============================================================\n")
  print("Thank you")
  print("\n==============================================================")

main()