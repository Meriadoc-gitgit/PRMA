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
from mazemdp.toolbox import egreedy, egreedy_loc
from mazemdp.mdp import Mdp as mdp
import csv
import os

"""==============================================================================================================="""

class PrioritizedReplayAgent:
  def __init__(self, mdp, ALPHA, DELTA, EPSILON, MAX_STEP ,render, episode) :
    """ Initialisation de la classe PrioritizedReplayAgent
        Arguments
        ----------
            mdp -- Mdp de mazemdp.mdp 
            ALPHA -- float : taux d'apprentissage
            DELTA -- float : seuil pour tester la convergence
            EPSILON -- float : taux d'exploration pour e-greedy
            MAX_STEP -- int : nombre de step pour l'apprentissage
            render -- bool : paramètre pour l'affichage en .avi
            episode -- int : nombre d'épisode pour l'apprentissage
        ----------
        
    """
    self.mdp = mdp
    self.render = render
    self.episode = episode
    self.q_table = np.zeros((mdp.nb_states, mdp.action_space.n))   # Q-Table nombre de state x nombre d'action
    self.memory = []
    self.experienced=[]
    self.nb_backup = 0

    self.ALPHA = ALPHA
    self.DELTA = DELTA
    self.EPSILON = EPSILON
    self.MAX_STEP = MAX_STEP

    if os.path.exists('executionInformation.csv'):
      os.remove('executionInformation.csv')

  """================== Excecution =================="""  
  def execute(self) : 
    """ Excécution de 
        Arguments
        ----------
            model_name -- str : nom du modèle
        ----------
    """
    

 
    for i in range(self.episode): 
      state, _ = self.mdp.reset()                
      if self.render:
        self.mdp.draw_v_pi(self.q_table, self.q_table.argmax(axis=1), recorder=None)
      for j in range(self.MAX_STEP):
        action, next_state, reward, terminated = self.step_in_world(state)

        if self.memory:                                      #update_model
          if len(self.memory)>5:                             #pour un pas de temps 5 update 
            for k in range(4):
              self.update_memory()
          self.update_memory()

        if state in self.mdp.terminal_states:
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
    action = egreedy(self.q_table, state, self.EPSILON)       # choix d'une action avec la méthode EPSILON
    next_state, reward, terminated, truncated, _ = self.mdp.step(action)    # effectue l'action à dans l'environnement

    TD = abs(self.TD_error(state,action,next_state,reward))    # calcul de la différence de magnitude  utilise comme priorite dans la Queue
    if TD:
      self.update_q_value(state, action, next_state, reward, self.ALPHA)   #backup qu'à partir du moment où on a atteint le goal
      self.nb_backup+=1  
    
    experience = [state,action,next_state,reward]
    self.experienced.append(experience) 
    self.fill_memory(experience)
    
    return action, next_state, reward, terminated

  """==============================================================================================================="""
  def update_q_value(self, state, action, next_state, reward, ALPHA) :
    """ Mets à jour le modele 

        Arguments
        ---------
            x -- int : etat d'origine
            a -- int : action effectue
            y -- int : etat d'arrivee
            r -- float : recompense recue
        
        Returns
        ----------      
            q_table[x,a] + ALPHA*(r+mdp.gamma*v_y-q_table[x,a])
    """
    #v_y correspond à la valeur maximal estimee pour l'etat y, multiplication par 1-terminated pour s'assurer de
    #ne prendre en compte ce resultat que si l'etat y n'est pas successeur d'un etat terminal
    v_y = 0
    if state not in self.mdp.terminal_states:
      v_y =np.max(self.q_table[next_state])

    self.q_table[state,action] = self.q_table[state,action] + ALPHA*(reward + self.mdp.gamma * v_y - self.q_table[state,action])
   

  """==============================================================================================================="""

  def get_policy_from_q(q: np.ndarray) -> np.ndarray:
      """ 
          Arguments
          ----------
              q -- np.ndarray
          
          Returns
          ----------      
              policy -- np.ndarray
      """
      # Outputs a policy given the action values
      policy = np.zeros(len(q), dtype=int)
      for x in range(len(q)) : 
          policy[x] = np.argmax(q[x,:])                  
      return policy   

  """==============================================================================================================="""
  # calcule l'equivalence de DELTA 


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
              reward + mdp.gamma* max_reward_next_state- q_table[state,action]
      """
      if state not in self.mdp.terminal_states:
        max_reward_next_state=np.max(self.q_table[next_state])
      else:
        max_reward_next_state = 0
      
      return reward + self.mdp.gamma * max_reward_next_state - self.q_table[state,action]


  """==============================================================================================================="""

  def get_nb_step(self):
    """ renvoie le nombre de pas de temps que fait l'agent jusqu'au goal
    """
    state,_ = self.mdp.reset()  
    for i in range(self.MAX_STEP):
      action = np.argmax(self.q_table[state, :])
      state,_,terminated, _, _ = self.mdp.step(action)
      if state in self.mdp.terminal_states:
        break
      
    with open('executionInformation.csv', mode ='a', newline='') as file:
      writer = csv.writer(file)
      writer.writerow([self.nb_backup, i])
     
  # def get_nb_step(self):
  #   state = 2
  #   for i in range(self.MAX_STEP):
  #     action = np.argmax(self.q_table[state])
  #     next_state = np.argmax(self.mdp.P[state,action])
  #     if next_state in self.mdp.terminal_states:
  #       return i
  #     state= next_state

  #   return i
    

      
    
      
  
 
    


      

        
  