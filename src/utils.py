"""

-------------------

Ce fichier contient des fonctions classiques de RL

-------------------
"""

import numpy as np




"""==============================================================================================================="""

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
# calcule l'equivalence de delta 


def TD_error(mdp, q_table,state,action,next_state,reward):
    """ Mets à jour le modele 

        Arguments
        ----------
            mdp -- Mdp from mazemdp.mdp : environnement
            q_table : modele
            state -- int : etat d'origine
            action -- int : action effectue
            next_state -- int : etat d'arrivee
            reward -- float : recompense recue
        
        Returns
        ----------      
            reward + mdp.gamma* max_reward_next_state- q_table[state,action]
    """
    terminated = state in mdp.terminal_states
    max_reward_next_state = np.max(q_table[int(next_state),:])*(1-terminated)                # retourne la valeur max estimee de next state
    return reward + mdp.gamma* max_reward_next_state- q_table[state,action]


"""==============================================================================================================="""

def moyenne_par_indice(liste):
    """Prends une liste de listes et retourne une liste contenant à chaque indice la valeur moyenne des valeurs de cet indice de chaque liste
    """
    tableau = np.array(liste)
    moyennes = np.mean(tableau, axis=0)
    moyennes = np.where(np.isnan(moyennes), None, moyennes)
    return moyennes.tolist()



