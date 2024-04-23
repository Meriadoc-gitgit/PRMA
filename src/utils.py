"""

-------------------

Ce fichier contient des fonctions classiques de RL

-------------------
"""

import numpy as np




"""==============================================================================================================="""
#in utilise pas cette fonction soit l'integrer dans le code soit la supprimer si inutile
def get_policy_from_q(q: np.ndarray) -> np.ndarray:
    """ 
        Arguments
        ----------
            q -- np.ndarray
        
        Returns
        ----------      
            policy -- np.ndarray
    """
    # Outputs a policy given the action values
    policy = np.zeros(len(q), dtype=int)
    for x in range(len(q)) : 
        policy[x] = np.argmax(q[x,:])                  
    return policy   


"""==============================================================================================================="""

def moyenne_par_indice(liste):
    """Prends une liste de listes et retourne une liste contenant à chaque indice la valeur moyenne des valeurs de cet indice de chaque liste
    """
    tableau = np.array(liste)
    moyennes = np.mean(tableau, axis=0)
    moyennes = np.where(np.isnan(moyennes), None, moyennes)
    return moyennes.tolist()

"""==============================================================================================================="""

def onehot(value, max_value) :
  vec = np.zeros(max_value)
  vec[value] = 1
  return vec

