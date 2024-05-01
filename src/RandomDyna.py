"""
-------------------

Contient l'algorithme de Random Dyna

-------------------
"""
import random
from PrioritizedReplayAgent import PrioritizedReplayAgent
from mazemdp.toolbox import egreedy

class RandomDyna(PrioritizedReplayAgent) : 
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
    key = random.randint(0, len(self.memory)-1)
    [state, action, next_state, reward] = self.memory[key]                # on récupère les valeurs de s_t, action, s_t+1, reward (ceux-là ne sont pas ceux en dehors de la boucle)
    del(self.memory[key])
    self.q_table[state,action] = self.q_table[state,action] + self.TD_error(state,action,next_state,reward)  #mise à jour du modele
  

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

    self.q_table[state,action] = self.q_table[state,action] + self.alpha*(self.TD_error(state,action,next_state,reward)) #backup qu'à partir du moment où on a atteint le goal
    self.nb_backup+=1  
    
    self.memory.append([state,action,next_state,reward])
    
    return action, next_state, reward, terminated
