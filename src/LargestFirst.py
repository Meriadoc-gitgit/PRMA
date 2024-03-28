"""
-------------------

Contient l'algorithme de Largest First

-------------------
"""
from Algorithms import PrioritizedReplayAgent

# Queue Dyna Priority Based on Prediction Difference Magnitude

class LargestFirst(PrioritizedReplayAgent) : 

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
    self.QTable[state,action] = self.updateQValue(state,action,next_state,reward)   #mise à jour du modele ici on prend le alpha de l'agent qu'on s'attend a etre 1
    self.updatePQueue(state)             #mise à jour de la queue

  """==============================================================================================================="""

  def updatePQueue(self, stateForPred):
    """ Ajoute les prédécesseurs de à la Queue

        Arguments
        ---------
        Returns
        ---------- 
        #unclear naming scheme for variables : is stateForPred the state at t-1 and state the current state and next_state the state two steps after stateForPred ? 
        #the argument for this method in update_model should have the same name as the parameter in here 
    """
    pred =[]
  
    for key , (state, action, next_state, reward) in self.PQueue.items():
        if next_state == stateForPred:
            e = abs(self.TD_error(state,action,next_state, reward))       
            if e >= self.delta:
              pred.append((e, [state,action,stateForPred, reward]))   #on peut pas directement les ajouter au dico sinon ca taille varie
    
    for (key_e,value_tab) in pred:          #après avoir trouvé les predecesseurs repondant au critere on peut les ajouter a PQueue
      self.PQueue[key_e] = value_tab



  """==============================================================================================================="""
