"""
-------------------

Contient l'algorithme de Largest First
#test
-------------------
"""
from PrioritizedReplayAgent import PrioritizedReplayAgent
from heapq import heappop, heappush
from mazemdp.toolbox import egreedy
from collections import defaultdict
import numpy as np



# Queue Dyna Priority Based on Prediction Difference Magnitude

class LargestFirst(PrioritizedReplayAgent) : 

  def __init__(self, mdp, alpha, delta, epsilon, max_step ,render, episode) :
    super().__init__(mdp, alpha, delta, epsilon, max_step ,render, episode)
    
    #stock l'entiereté des experiences vecus contrairement à mémorie qui ne contient
    #que les experiences avec une TD significative
    self.experienced = defaultdict(list)


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
  def fill_memory(self, experience):
    """ Ajoute experience à la mémoire si son erreur de différence temporelle est significative

      Arguments
      ---------
          experience -- list : experience à ajouter à la mémoire
      ---------
      Returns
      ---------- 
    """
    [state,action,next_state,reward] = experience
    TD = abs(self.TD_error(state,action,next_state,reward))
    if TD >= self.delta:
      heappush(self.memory, (-TD, experience))


  def update_memory(self): 
    """ Traite l'experience la plus prioritaire de la mémoire: mise à jour des q-valeurs, 
        réinsertion en mémoire et ajout des prédecesseurs de l'experience)

      Arguments
      ---------
      Returns
      ---------- 
    """
    (TD, [state, action, next_state, reward]) = heappop(self.memory)

    self.update_q_value(state,action,next_state,reward, 1)   
    self.fill_memory([state,action,next_state,reward])

    TD = abs(self.TD_error(state,action,next_state,reward))
    if TD >= self.delta:
      self.add_predecessors(state) 

    
  def handle_step(self, state,action,next_state,reward):

    """Gestion d'un pas dans le monde, traitement spécifique à LargestFirst:
    Arguments
    ----------
        state -- int : état courant
        action -- int : action effectuée
        next_state -- int : état suivant
        reward -- float : récompense accordée
    ----------  
    Returns
    ----------  
    """

    TD = abs(self.TD_error( state,action,next_state,reward))    # calcul de la différence de magnitude  utilise comme priorite dans la Queue
    if TD:
      self.update_q_value(state, action, next_state, reward, self.alpha)   #backup qu'à partir du moment où on a atteint le goal
      self.nb_backup+=1  
    
    experience = [state,action,next_state,reward]
    self.fill_memory(experience)  
    self.add_experience(experience)
    
  def add_experience(self, experience):

    """Ajout d'une experience à experienced
    Arguments
    ----------
        experience -- list : experience à stocker
    ----------  
    Returns
    ----------  
    """
    state, action , next_state, reward = experience
    self.experienced[next_state].append(experience)
