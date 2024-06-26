"""
-------------------
La classe PrioritizedReplayAgent est une classe abstraite contenant les méthodes partagées par les
des sous classes : LargestFirst, FocusedDyna et RandomDyna
-------------------
"""
# Import necessary libraries

import numpy as np
from mazemdp.toolbox import egreedy
from collections import defaultdict
from heapq import heappop, heappush
import csv
import os
from bbrl_gymnasium.envs.maze_mdp import MazeMDPEnv
import gymnasium as gym
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
import matplotlib.pyplot as plt

"""==============================================================================================================="""

class PrioritizedReplayAgent:
  def __init__(self, mdp, alpha, delta, epsilon, max_step ,render, episode, video_name) :
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
    self.video_name = video_name
    self.episode = episode
    self.q_table = np.zeros((mdp.unwrapped.nb_states, mdp.action_space.n))   # Q-Table nombre de state x nombre d'action
    self.memory = []  #memoire contient un tri des experiences vecues
    self.nb_backup = 0
    self.start = self.mdp.reset()[0]
    self.experienced = defaultdict(set)


    self.alpha = alpha
    self.delta = delta
    self.epsilon = epsilon
    self.max_step = max_step

    # if os.path.exists('executionInformation.csv'):
    #   os.remove('executionInformation.csv')
    if os.path.exists("executionInformation.csv"):
    # Ouvrir le fichier en mode écriture ('w') pour vider son contenu
      with open("executionInformation.csv", 'w') as file:
        writer = csv.writer(file)
        writer.writerow([0,self.max_step])

    

  """================== Excecution =================="""  
  def execute(self) : 
    """ Excécution de PrioritizedReplayAgent
        Arguments
        ----------
            model_name -- str : nom du modèle
        ----------
    """
    self.get_nb_step()

    if self.render:
      video_recorder = VideoRecorder(self.mdp.unwrapped, "videos/"+self.video_name+".mp4", enabled=self.render)

    for i in range(self.episode): 
      print(i)        # to track the process
      state, _ = self.mdp.reset()            

      self.mdp.unwrapped.draw_v_pi(self.q_table, self.q_table.argmax(axis=1), recorder=video_recorder)
      self.mdp.unwrapped.render()

      for j in range(self.max_step):
        action, next_state, reward, terminated = self.step_in_world(state)
        for k in range(5):
          if not self.memory:
            break
          self.update_memory()
   
        if state in self.mdp.unwrapped.terminal_states:
          print(self.mdp.terminal_states)
          print(self.mdp.unwrapped.terminal_states)
          break 
        
        state = next_state                     #l'agent est maintenant à l'etat qui succède x
    if self.render:
      print("finish")
      self.mdp.current_state = 0
      self.mdp.unwrapped.draw_v_pi(self.q_table, self.q_table.argmax(axis=1), recorder=video_recorder)
      video_recorder.close()

      

   

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

    experience = (state,action, next_state, reward)
    priority = self.compute_priority(experience)

    if priority :
      self.q_table[state,action] = self.q_table[state,action] + self.alpha*(self.TD_error(state,action,next_state,reward))   #backup qu'à partir du moment où on a atteint le goal
      self.add_predecessors(state)
      self.nb_backup+=1
      self.get_nb_step()

    
    self.fill_memory(experience, priority = priority) 
    self.experienced[next_state].add(experience)

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
            reward + gamma* max_reward_next_state- q_table[state,action]
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

    if state_for_pred in self.experienced :
      pred = self.experienced[state_for_pred]
      # del self.experienced[state_for_pred]
      self.experienced[state_for_pred] = set()
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
    
    if priority>= self.delta:
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

    (priority, (state, action, next_state, reward)) = heappop(self.memory)
    #ici on multiplie en fait self.TD_error par alpha=1
    self.q_table[state,action] = self.q_table[state,action] + self.TD_error(state,action,next_state,reward) 
    self.fill_memory((state,action,next_state,reward), priority=priority)

    if priority <= self.delta:
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



      

        
  