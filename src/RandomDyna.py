"""
-------------------

Contient l'algorithme de Random Dyna

-------------------
"""
import random
from Algorithms import PrioritizedReplayAgent

class RandomDyna(PrioritizedReplayAgent) : 

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
    self.QTable[state,action] = self.updateQValue(state,action,next_state,reward)   #mise à jour du modele
  
  """================== CHOIX DE CLÉ =================="""  
  def key_choice(self) : 
    """ Choix de clé selon l'algorithme pris en entrée

        Arguments
        ----------
        
        Returns   
        ----------      
            key -- float : clé choisie selon l'algorithme pris en entrée
    """
    return random.choice(list(self.PQueue.keys()))
