"""
-------------------

Contient l'algorithme de Focused Dyna

-------------------
"""

from heapq import heappop, heappush
from assets.PrioritizedReplayAgent import PrioritizedReplayAgent
import numpy as np
from mazemdp.toolbox import egreedy

class FocusedDyna(PrioritizedReplayAgent) : 
    def __init__(self, mdp, alpha, delta, epsilon, max_step ,render, episode) :
       super().__init__(mdp, alpha, delta, epsilon, max_step ,render, episode)
       
       #creation du dictionnaire etat:distance_du_debut
       self.stepsFromStart = {self.start : 0}

    def compute_priority(self,experience):
        [state,_, next_state, reward] =experience
        if state not in self.mdp.unwrapped.terminal_states:
            max_reward_next_state=np.max(self.q_table[next_state])
        else:
            max_reward_next_state = 0
        return (pow(self.mdp.unwrapped.gamma,self.stepsFromStart[state]))* (reward + self.mdp.unwrapped.gamma*max_reward_next_state)
    
