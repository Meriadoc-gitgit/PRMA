"""
-------------------

Contient l'algorithme de Random Dyna

-------------------
"""
import random
from Algorithms import PrioritizedReplayAgent
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
    key = self.key_choice()
    [state, action, next_state, reward] = self.memory[key]                # on récupère les valeurs de s_t, action, s_t+1, reward (ceux-là ne sont pas ceux en dehors de la boucle)
    del(self.memory[key])
    self.update_q_value(state,action,next_state,reward, 1)   #mise à jour du modele
  
  """================== CHOIX DE CLÉ =================="""  
  def key_choice(self) : 
    """ Choix de clé selon l'algorithme pris en entrée

        Arguments
        ----------
        
        Returns   
        ----------      
            key -- float : clé choisie selon l'algorithme pris en entrée
    """
    return random.randint(0, len(self.memory)-1)


  def fill_memory(self, experience):
    self.memory.append(experience)

  def handle_step(self, state,action,next_state,reward):

    self.update_q_value(state, action, next_state, reward, self.ALPHA)   #backup qu'à partir du moment où on a atteint le goal
    self.nb_backup+=1  
    
    experience = [state,action,next_state,reward]
    self.fill_memory(experience)
    