"""
-------------------

Contient l'algorithme de Focused Dyna

-------------------
"""

import heapq
from heapq import heappop, heappush
from Algorithms import PrioritizedReplayAgent
import numpy as np

class FocusedDyna(PrioritizedReplayAgent) : 
  def __init__(self, mdp, ALPHA, DELTA, EPSILON, MAX_STEP ,render, episode) :
    super().__init__(mdp, ALPHA, DELTA, EPSILON, MAX_STEP ,render, episode)
    
    #creation du dictionnaire etat:distance_du_debut
    self.start = self.mdp.reset()[0]
    self.stepsFromStart = {self.start : 0}
    self.djikstra()

    # self.stepsFromStart = self.dijkstra() 
    # self.dijkstra()

  def fill_memory(self,experience):
      """
      ajoute dans la priority queue la combinaison action états utilisé pour focused dyna 
      clé : priorité calculé avec la formule proposée par Peng & Williams section 6.1.2

      Arguments 
      -----------
          mdp -- Mdp from mazemdp.mdp
          experience : [state,actio,next_state,reward]
          stepsFromStart -- dictionnary (state : distance_from_start)

      """
      [state,action,next_state,reward] = experience
      priority = pow(self.mdp.gamma,self.stepsFromStart[state]) * self.TD_error(state,action,next_state,reward)
      heappush(self.memory, (-priority, experience))



  """==============================================================================================================="""
  def djikstra(self):
    frontier = []
    visited = set()
    heappush(frontier,(0,self.start))
    while frontier :
       distance, state = heappop(frontier)

       if state in visited:
          continue
       visited.add(state)

       for action in range(self.mdp.action_space.n):
          next_state = np.argmax(self.mdp.P[state,action])
          if next_state not in self.stepsFromStart or self.stepsFromStart[next_state] > distance+1:
             self.stepsFromStart[next_state] = distance +1
             heappush(frontier,(self.stepsFromStart[next_state], next_state))
    

  """================== METTRE À JOUR LE MODÈLE =================="""  
  def update_memory(self) : 
    """ Mettre à jour le modèle - instructions correspondantes à la deuxième partie de la boucle

        Arguments
        ----------
            algo -- str : nom de l'algorithme choisie 
            state -- int : état courant
            action -- int : action à effectuer
            next_state -- int : état suivant
            reward -- float : récompense accordée

        
        Returns
        ----------      
    """
    (priority, [state, action, next_state, reward]) = heappop(self.memory)

    self.update_q_value(state,action,next_state,reward, 1)   #mise à jour du modele ici on prend le ALPHA de l'agent qu'on s'attend a etre 1

    

  """==============================================================================================================="""
  
