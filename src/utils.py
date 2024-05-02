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

def moyenne_par_indice(liste_de_listes):
    """Prends une liste de listes et retourne une liste contenant à chaque indice la valeur moyenne des valeurs de cet indice de chaque liste
    """

    somme_par_indice = {}

    for liste in liste_de_listes:
        for indice, valeur in enumerate(liste):
            if indice not in somme_par_indice:
                somme_par_indice[indice] = []  
            somme_par_indice[indice].append(valeur)  

    resultat = []
    for indice in sorted(somme_par_indice.keys()): 
        valeurs = somme_par_indice[indice]
        moyenne = sum(valeurs) / len(valeurs) 
        resultat.append(moyenne)

    return resultat



"""==============================================================================================================="""

def onehot(value, max_value) :
  vec = np.zeros(max_value)
  vec[value] = 1
  return vec

