"""
-------------------

Contient l'algorithme de Largest First
#test
-------------------
"""
from assets.PrioritizedReplayAgent import PrioritizedReplayAgent
from heapq import heappop, heappush
from mazemdp.toolbox import egreedy
from collections import defaultdict
import numpy as np



# Queue Dyna Priority Based on Prediction Difference Magnitude

class LargestFirst(PrioritizedReplayAgent) : 

  """==============================================================================================================="""
  def compute_priority(self, experience):
    [state,action, next_state, reward] = experience
    return abs(self.TD_error( state,action,next_state,reward)) 