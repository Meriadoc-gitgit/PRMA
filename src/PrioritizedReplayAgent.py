"""
-------------------
La classe PrioritizedReplayAgent est une classe abstraite contenant les méthodes partagées par les
des sous classes : LargestFirst, FocusedDyna et RandomDyna
-------------------
"""
# Import necessary libraries

from abc import abstractmethod
# from moviepy.editor import ipython_display as video_display
import numpy as np
import matplotlib.pyplot as plt
from bbrl_gymnasium.envs.maze_mdp import MazeMDPEnv
import gymnasium as gym
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from IPython.display import Video
from mazemdp.toolbox import egreedy
from mazemdp.mdp import Mdp as mdp
from collections import defaultdict
from heapq import heappop, heappush
import csv
import os

"""==============================================================================================================="""

class PrioritizedReplayAgent:
  def __init__(self, mdp, alpha, delta, epsilon, max_step ,render, episode) :
    """ Initialisation de la classe PrioritizedReplayAgent
        Arguments
        ----------
            mdp -- Mdp de mazemdp.mdp 
            alpha -- float : taux d'apprentissage
            delta -- float : seuil pour tester la convergence
            epsilon -- float : taux d'exploration pour e-greedy
            max_step -- int : nombre de step pour l'apprentissage
            render -- bool : paramètre pour l'affichage en .avi
            episode -- int : nombre d'épisode pour l'apprentissage
        ----------
        
    """
    self.mdp = mdp
    self.render = render
    self.episode = episode
    self.q_table = np.zeros((mdp.unwrapped.nb_states, mdp.action_space.n))   # Q-Table nombre de state x nombre d'action
    self.memory = []  #memoire contient un tri des experiences vecues
    self.nb_backup = 0
    self.start = self.mdp.reset()[0]
    self.experienced = defaultdict(list)


    self.alpha = alpha
    self.delta = delta
    self.epsilon = epsilon
    self.max_step = max_step

    if os.path.exists('executionInformation.csv'):
      os.remove('executionInformation.csv')

  """================== Excecution =================="""  
  def execute(self) : 
    """ Excécution de PrioritizedReplayAgent
        Arguments
        ----------
            model_name -- str : nom du modèle
        ----------
    """
 
    for i in range(self.episode): 
      state, _ = self.mdp.reset()                
      if self.render:
        self.mdp.draw_v_pi(self.q_table, self.q_table.argmax(axis=1), recorder=None)
      
      for j in range(self.max_step):
        action, next_state, reward, terminated = self.step_in_world(state)

        for k in range(5):
          if not self.memory:
            break
          self.update_memory()
   
        if state in self.mdp.unwrapped.terminal_states:
          # print("end")
          break 
        
        state=next_state                     #l'agent est maintenant à l'etat qui succède x
      self.get_nb_step()
      

   

  """================== POUR EFFECTUER UN PAS DANS LE MONDE ==================""" 
  def step_in_world(self, state) : 
    """ Effectuer un pas dans le monde - instructions correspondantes à la premières parties de la boucle

        Arguments
        ----------
            state -- int : état courant
        
        Returns
        ----------      
            action -- int : l'action déterminée par e-greedy
            next_state -- int : l'état suivant
            reward -- float : récompense accordée
            terminated -- bool : déterminé si l'état est terminal
    """
    action = egreedy(self.q_table, state, self.epsilon)       # choix d'une action avec la méthode epsilon
    next_state, reward, terminated, truncated, _ = self.mdp.step(action)    # effectue l'action à dans l'environnement

    experience = [state,action, next_state, reward]
    priority = self.compute_priority(experience)

    if priority :
      self.q_table[state,action] = self.q_table[state,action] + self.alpha*(self.TD_error(state,action,next_state,reward))   #backup qu'à partir du moment où on a atteint le goal
      self.add_predecessors(state)
      self.nb_backup+=1  
    
    self.fill_memory(experience, priority = priority)  
    self.experienced[next_state].append(experience)

    return action, next_state, reward, terminated

  """==============================================================================================================="""

  def TD_error(self,state,action,next_state,reward):
    """ Mets à jour le modele #explication insuffisante

        Arguments
        ----------
            state -- int : etat d'origine
            action -- int : action effectue
            next_state -- int : etat d'arrivee
            reward -- float : recompense recue
        
        Returns
        ----------      
            reward + mdp.unwrapped.gamma* max_reward_next_state- q_table[state,action]
    """
    if state not in self.mdp.unwrapped.terminal_states:
      max_reward_next_state=np.max(self.q_table[next_state])
    else:
      max_reward_next_state = 0
    
    return reward + self.mdp.unwrapped.gamma * max_reward_next_state - self.q_table[state,action]
 
  """==============================================================================================================="""

  def add_predecessors(self, state_for_pred):
    """ Ajoute les experiences qui ont comme next_state state_for_pred à la mémoire

        Arguments
        ---------
            state_for_pred -- int : 
        ---------
        Returns
        ---------- 
    """
    if state_for_pred in self.experienced.keys() :
      pred = self.experienced[state_for_pred]
      for experience in pred:          #après avoir trouvé les predecesseurs repondant au critere on peut les ajouter a PQueue
        self.fill_memory(experience)

  """==============================================================================================================="""
  def fill_memory(self, experience, priority = None):
    """ Ajoute experience à la mémoire si son erreur de différence temporelle est significative

      Arguments
      ---------
          experience -- list : experience à ajouter à la mémoire
          priority -- float : priorité de l'experience, si ajout de l'experience suite à un pas dans le monde elle est connu si ajout lors du traitement
                              des predecessors elle est inconnu il faut donc la calculer
      ---------
      Returns
      ---------- 
    """
    if priority is None:
      priority = self.compute_priority(experience)

    if priority>= self.delta :
      heappush(self.memory, (-priority, experience))

  """==============================================================================================================="""
  def update_memory(self): 
    """ Traite l'experience la plus prioritaire de la mémoire: mise à jour des q-valeurs, 
        réinsertion en mémoire et ajout des prédecesseurs de l'experience)

      Arguments
      ---------
      Returns
      ---------- 
    """
    (priority, [state, action, next_state, reward]) = heappop(self.memory)

    #ici on multiplie en fait self.TD_error par alpha=1
    self.q_table[state,action] = self.q_table[state,action] + self.TD_error(state,action,next_state,reward) 
    self.fill_memory([state,action,next_state,reward], priority=priority)

    if priority >= self.delta:
      self.add_predecessors(state) 

  """==============================================================================================================="""

  def get_nb_step(self):
    state = self.start
    for nb_step in range(self.max_step):
      action = np.argmax(self.q_table[state])
      next_state = np.argmax(self.mdp.unwrapped.P[state,action])
      if next_state in self.mdp.unwrapped.terminal_states:
        break
      state = next_state

    with open('executionInformation.csv', mode ='a', newline='') as file:
      writer = csv.writer(file)
      writer.writerow([self.nb_backup, nb_step])



      

        
  