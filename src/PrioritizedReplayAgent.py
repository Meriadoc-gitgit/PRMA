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

        if self.memory:                                      
          k=5
          while(k>0 and self.memory):
            self.update_memory()
            k-=1

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

    self.handle_step(state,action,next_state,reward)
    
    return action, next_state, reward, terminated

  """==============================================================================================================="""
  def update_q_value(self, state, action, next_state, reward, alpha) :
    """ Mets à jour le modele 

        Arguments
        ---------
            x -- int : etat d'origine
            a -- int : action effectue
            y -- int : etat d'arrivee
            r -- float : recompense recue
        
        Returns
        ----------      
            q_table[x,a] + alpha*(r+mdp.unwrapped.gamma*v_y-q_table[x,a])
    """
    #v_y correspond à la valeur maximal estimee pour l'etat y, multiplication par 1-terminated pour s'assurer de
    #ne prendre en compte ce resultat que si l'etat y n'est pas successeur d'un etat terminal
    v_y = 0
    if state not in self.mdp.unwrapped.terminal_states:
      v_y =np.max(self.q_table[next_state])

    self.q_table[state,action] = self.q_table[state,action] + alpha*(reward + self.mdp.unwrapped.gamma * v_y - self.q_table[state,action])
   

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

    

      
    
      
  
 
    


      

        
  