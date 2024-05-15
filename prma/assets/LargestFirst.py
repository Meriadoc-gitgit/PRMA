"""
-------------------

Contient l'algorithme de Largest First
#test
-------------------
"""
from assets.PrioritizedReplayAgent import PrioritizedReplayAgent




# Queue Dyna Priority Based on Prediction Difference Magnitude

class LargestFirst(PrioritizedReplayAgent) : 

  """==============================================================================================================="""
  def compute_priority(self, experience):
    [state,action, next_state, reward] = experience
    return abs(self.TD_error( state,action,next_state,reward)) 