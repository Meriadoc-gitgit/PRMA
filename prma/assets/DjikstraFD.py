"""
-------------------

Contient la classe DjikstraFD classe fille de Focused Dyna qui utilise Djikstra 
pour calculer les distances entre états dans le maze

-------------------
"""

from assets.FocusedDyna import FocusedDyna
from heapq import heappop, heappush
import numpy as np


class DjikstraFD(FocusedDyna) :
    def __init__(self, mdp, alpha, delta, epsilon, max_step, render, episode):
        super().__init__(mdp, alpha, delta, epsilon, max_step, render, episode)
        self.djikstra()

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
                next_state = np.argmax(self.mdp.unwrapped.P[state,action])
                if next_state not in self.stepsFromStart or self.stepsFromStart[next_state] > distance+1:
                    self.stepsFromStart[next_state] = distance +1
                    heappush(frontier,(self.stepsFromStart[next_state], next_state))
    