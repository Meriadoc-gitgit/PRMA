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
    self.QTable = np.zeros((mdp.nb_states, mdp.action_space.n))   # Q-Table nombre de state x nombre d'action
    self.PQueue = dict()                                               # Dictionnaire PQueue for planning
    self.memory = []
    self.experienced=[]
    self.nb_backup = 0

    self.alpha = alpha
    self.delta = delta
    self.epsilon = epsilon
    self.max_step = max_step


  """================== Excecution =================="""  
  def execute(self) : 
    """ Excécution de 
        Arguments
        ----------
            model_name -- str : nom du modèle
        ----------
    """
    infos = [
      ['nb_backup', 'nb_step']
    ]

 
    for i in range(self.episode): 
      state, _ = self.mdp.reset()                
      if self.render:
        self.mdp.draw_v_pi(self.QTable, self.QTable.argmax(axis=1), recorder=None)
      for j in range(self.max_step):
        action, next_state, reward, terminated = self.step_in_world(state)


        if self.memory:                                      #update_model
          if len(self.memory)>5:                             #pour un pas de temps 5 update 
            for k in range(4):
              self.updateMemory()
          self.updateMemory()

        if state in self.mdp.terminal_states:
          break 

        state=next_state                     #l'agent est maintenant à l'etat qui succède x
          
      nb_step_res = self.get_nb_step()
      infos.append([self.nb_backup, nb_step_res])

    with open('executionInformation.csv', 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerows(infos)

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
    action = egreedy(self.QTable, state, self.epsilon)       # choix d'une action avec la méthode epsilon
    next_state, reward, terminated, truncated, _ = self.mdp.step(action)    # effectue l'action à dans l'environnement

    TD = abs(self.TD_error(state,action,next_state,reward))    # calcul de la différence de magnitude  utilise comme priorite dans la Queue
    if TD:
      self.updateQValue(state, action, next_state, reward, self.alpha)   #backup qu'à partir du moment où on a atteint le goal
      self.nb_backup+=1  
    
    experience = [state,action,next_state,reward]
    self.experienced.append(experience) 
    if TD >= self.delta:
      self.fillMemory(TD,experience)
    
    return action, next_state, reward, terminated

  """==============================================================================================================="""
  def updateQValue(self, state, action, next_state, reward, alpha) :
    """ Mets à jour le modele 

        Arguments
        ---------
            x -- int : etat d'origine
            a -- int : action effectue
            y -- int : etat d'arrivee
            r -- float : recompense recue
        
        Returns
        ----------      
            q_table[x,a] + alpha*(r+mdp.gamma*v_y-q_table[x,a])
    """
    #v_y correspond à la valeur maximal estimee pour l'etat y, multiplication par 1-terminated pour s'assurer de
    #ne prendre en compte ce resultat que si l'etat y n'est pas successeur d'un etat terminal
    v_y = 0
    if state not in self.mdp.terminal_states:
      v_y =np.max(self.QTable[next_state])

    self.QTable[state,action] = self.QTable[state,action] + alpha*(reward + self.mdp.gamma * v_y - self.QTable[state,action])
   

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
  # calcule l'equivalence de delta 


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
        max_reward_next_state=np.max(self.QTable[next_state])
      else:
        max_reward_next_state = 0
      
      return reward + self.mdp.gamma * max_reward_next_state - self.QTable[state,action]


  """==============================================================================================================="""

  def get_nb_step(self):
    """ renvoie le nombre de pas de temps que fait l'agent jusqu'au goal
    """
    state,_ = self.mdp.reset()  
    for i in range(self.max_step):
      action = np.argmax(self.QTable[state, :])
      state,_,terminated, _, _ = self.mdp.step(action)
      if state in self.mdp.terminal_states:
        return i
    return i
     
  # def get_nb_step(self):
  #   state = 2
  #   for i in range(self.max_step):
  #     action = np.argmax(self.QTable[state])
  #     next_state = np.argmax(self.mdp.P[state,action])
  #     if next_state in self.mdp.terminal_states:
  #       return i
  #     state= next_state

  #   return i
    

      
    
      
  
 
    


      

        
  