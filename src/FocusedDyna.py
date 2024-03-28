"""
-------------------

Contient l'algorithme de Focused Dyna

-------------------
"""

import heapq
from heapq import heappop, heappush
from Algorithms import PrioritizedReplayAgent

class FocusedDyna(PrioritizedReplayAgent) : 
  def __init__(self, mdp, alpha, delta, epsilon, max_step ,render, episode) :
    super().__init__(mdp, alpha, delta, epsilon, max_step ,render, episode)
    
    #creation du dictionnaire etat:distance_du_debut
    self.start = self.mdp.reset()[0]
    self.stepsFromStart = self.dijkstra() 
    self.stepsFromStart = {self.start : 0}
    self.dijkstra()

  def updatePQueue(self,current_state, action, next_state, reward):
      """
      ajoute dans la priority queue la combinaison action états utilisé pour focused dyna 
      clé : priorité calculé avec la formule proposée par Peng & Williams section 6.1.2

      Arguments 
      -----------
          mdp -- Mdp from mazemdp.mdp
          PQueue -- priority queue of the agent- dictionary (state: priority)
          stepsFromStart -- dictionnary (state : distance_from_start)

      """
      priority = pow(self.mdp.gamma,self.stepsFromStart[current_state]) * self.TD_error(current_state,action,next_state,reward)
      self.PQueue[priority] = (current_state, action, next_state, reward)


  """==============================================================================================================="""
  def dijkstra(self):
      """
      Dijkstra's algorithm used to find the shortest distance from the start state to all other states in a graph. 
      Used for Focused Dyna

      Returns:
          A dictionary where each state is the key and the value is the distance from the start state.
      """
      frontier = []  
      visited = set()   
    
      heappush(frontier, (0, self.start))  

      while frontier:
          current_distance, current_state = heappop(frontier)

          if current_state in visited:
              continue
          visited.add(current_state)

          for i in range(4):
              next_state, _, _, _, _ = self.mdp.step(i)
              self.mdp.current_state = current_state #not sure if works yet
              if next_state not in self.stepsFromStart or self.stepsFromStart[next_state] > current_distance + 1:
                  self.stepsFromStart[next_state] = current_distance + 1
                  heappush(frontier, (self.stepsFromStart[next_state], next_state))
  
  
  

  """================== METTRE À JOUR LE MODÈLE =================="""  
  def update_model(self, state, action, next_state, reward) : 
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
    key = self.key_choice()
    state, action, next_state, reward = self.PQueue[key]                # on récupère les valeurs de s_t, action, s_t+1, reward (ceux-là ne sont pas ceux en dehors de la boucle)
    self.PQueue.pop(key)
    self.QTable[state,action] = self.updateQValue(state,action,next_state,reward)
    self.updatePQueue(state,action, next_state, reward)

  """==============================================================================================================="""
  

  """================== CHOIX DE CLÉ =================="""  
  def key_choice(self) : 
    """ Choix de clé selon l'algorithme pris en entrée

        Arguments
        ----------
        
        Returns   
        ----------      
            key -- float : clé choisie selon l'algorithme pris en entrée
    """ 
    return max(self.PQueue.keys())  