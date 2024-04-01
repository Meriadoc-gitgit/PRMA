"""
-------------------

Contient l'algorithme de Largest First

-------------------
"""
from Algorithms import PrioritizedReplayAgent
from heapq import heappop, heappush


# Queue Dyna Priority Based on Prediction Difference Magnitude

class LargestFirst(PrioritizedReplayAgent) : 

  def addPredecessors(self, stateForPred):
    """ Ajoute les experiences qui on comme next_state stateForPred

        Arguments
        ---------
        Returns
        ---------- 
        #unclear naming scheme for variables : is stateForPred the state at t-1 and state the current state and next_state the state two steps after stateForPred ? 
        #the argument for this method in update_model should have the same name as the parameter in here 
    """
    pred = self.findPredecessors(stateForPred)
    for (key_e,experience) in pred:          #après avoir trouvé les predecesseurs repondant au critere on peut les ajouter a PQueue
      self.fillMemory(key_e,experience)



  """==============================================================================================================="""
  def fillMemory(self, key, experience):
     heappush(self.memory, (-key, experience))

  def updateMemory(self):
    (TD, [state, action, next_state, reward]) = heappop(self.memory)

    self.updateQValue(state,action,next_state,reward, 1)   #mise à jour du modele ici on prend le alpha de l'agent qu'on s'attend a etre 1
    self.fillMemory(self.TD_error(state,action,next_state,reward), [state,action,next_state,reward])
    self.addPredecessors(state) 

  def findPredecessors(self,stateForPred ):
    pred =[]
    for [state, action, next_state, reward] in self.experienced:
      if next_state == stateForPred:
        TD = abs(self.TD_error(state,action,next_state, reward))       
        if TD >= self.delta:
          pred.append((TD, [state,action,stateForPred, reward]))
    return pred
    