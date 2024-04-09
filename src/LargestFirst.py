"""
-------------------

Contient l'algorithme de Largest First

-------------------
"""
from PrioritizedReplayAgent import PrioritizedReplayAgent
from heapq import heappop, heappush
from mazemdp.toolbox import egreedy


# Queue Dyna Priority Based on Prediction Difference Magnitude

class LargestFirst(PrioritizedReplayAgent) : 

  def __init__(self, mdp, alpha, delta, epsilon, max_step ,render, episode) :
    super().__init__(mdp, alpha, delta, epsilon, max_step ,render, episode)
    
    #creation du dictionnaire etat:distance_du_debut
    self.experienced = []


  def add_predecessors(self, stateForPred):
    """ Ajoute les experiences qui on comme next_state stateForPred

        Arguments
        ---------
        Returns
        ---------- 
        #unclear naming scheme for variables : is stateForPred the state at t-1 and state the current state and next_state the state two steps after stateForPred ? 
        #the argument for this method in update_model should have the same name as the parameter in here 
    """
    pred = self.find_predecessors(stateForPred)
    for experience in pred:          #après avoir trouvé les predecesseurs repondant au critere on peut les ajouter a PQueue
      self.fill_memory(experience)



  """==============================================================================================================="""
  def fill_memory(self, experience):
     [state,action,next_state,reward] = experience
     TD = abs(self.TD_error(state,action,next_state,reward))
     if TD >= self.delta:
      heappush(self.memory, (-TD, experience))


  def update_memory(self): 
    (TD, [state, action, next_state, reward]) = heappop(self.memory)

    self.update_q_value(state,action,next_state,reward, 1)   #mise à jour du modele ici on prend le alpha de l'agent qu'on s'attend a etre 1
    self.fill_memory([state,action,next_state,reward])
    # self.add_predecessors(state) 

  def find_predecessors(self,stateForPred ):
    pred =[]
    for [state, action, next_state, reward] in self.experienced:
      if next_state == stateForPred:
        pred.append([state,action,stateForPred, reward])
    return pred
    
  def handle_step(self, state,action,next_state,reward):

    TD = abs(self.TD_error(state,action,next_state,reward))    # calcul de la différence de magnitude  utilise comme priorite dans la Queue
    if TD:
      self.update_q_value(state, action, next_state, reward, self.alpha)   #backup qu'à partir du moment où on a atteint le goal
      self.nb_backup+=1  
    
    experience = [state,action,next_state,reward]
    self.experienced.append(experience) 
    self.fill_memory(experience)