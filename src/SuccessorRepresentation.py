"""
-------------------

Contient l'intégral du code de la Successor Representation

-------------------
"""
# Import necessary libraries
import copy
import numpy as np

from Algorithms import PrioritizedReplayAgent

class SuccessorRepresentationAgent(PrioritizedReplayAgent) : 
  def __init__(self, env) : 
    self.mdp = env

    self.list_state_reward_SR = dict()
    for st in range(self.mdp.nb_states) : 
      self.list_state_reward_SR[st] = 0

    self.T = np.zeros((env.nb_states, env.nb_states))                  # Matrice de transition pour le SR  nb_state x nb_state
    self.list_exp = self.exploration([])


  def execute_action_chain(self,action_list) : 
    """ Exécution d'une chaine d'action

        Arguments
        ----------
            env -- Mdp from mazemdp.mdp : environnement sur lequel travailler
            action_list -- List[int] : chaine d'action à effectuer

        
        Returns
        ----------      
        state -- int : état atteint à la fin de la chaine d'action
    """
    self.mdp.reset()
    if len(action_list) == 0 : return

    list_state = []

    for action in action_list : 
      state,_,_,_,_ = self.mdp.step(action)
      list_state.append(state)
    return state, list_state


  def exploration(self, list_exp) : 
    """ Phase d'exploitation de la Successor Representation afin de construire la matrice de transition T

        Arguments
        ----------
            env -- Mdp from mazemdp.mdp : environnement sur lequel travailler
            list_exp -- List[List[int]] : liste principale d'expérience

        
        Returns
        ----------      
            list_exp -- List[List[int]] : liste principale d'expérience développée par cette fonction après 5000 tours de boucle
    """
    if len(list_exp) == 0 : 
      list_exp.append([0])
    
    i = 0
    for trail in list_exp : 
      i += 1
      for action in range(self.mdp.action_space.n) : 
        tmp_trail = copy.deepcopy(trail)

        self.mdp.reset()
        current_state, list_state = self.execute_action_chain(tmp_trail)
        
        next_state, reward, terminated, truncated,_ = self.mdp.step(action)
        list_state.append(next_state)
        

        if current_state != next_state :
          tmp_trail.append(action)
          #print(tmp_trail, action, next_state, current_state, terminated)
          self.T[current_state][next_state] += 1

          
          if tmp_trail not in list_exp : 
            list_exp.append(tmp_trail)

          if next_state in self.mdp.terminal_states : 
            for st in list_state : 
              self.list_state_reward_SR[st] += 1

      list_exp.remove(trail)
      if i == 5000 : 
        return list_exp

  def transition_rate_from_state(self, state) : 
    """ Déterminer le taux de transition aux autres états à partir de l'état pris en entrée

        Arguments
        ----------
            env -- Mdp from mazemdp.mdp : environnement sur lequel travailler
            state -- int : état demandé

        
        Returns
        ----------      
            transition_rate -- List[float] : taux de transition aux autres états à partir de l'état pris en entrée
    """
    #print(self.mdp.gamma * self.T)
    SR = np.linalg.inv(np.eye(self.mdp.nb_states) - self.mdp.gamma * self.T) # (I - gamma * T)^-1
    transition_rate = SR[state]

    return transition_rate

  def transition_recommandation(self, state) : 
    """ Retourne le dictionnaire de clé (action, next_state) et de valeur taux de transition de state à next_state

        Arguments
        ----------
            env -- Mdp from mazemdp.mdp : environnement sur lequel travailler
            state -- int : état demandé

        
        Returns
        ----------      
            possible_state_to_transit -- Dict[(action, next_state), transition_rate] : dictionnaire de clé (action, next_state) et de valeur taux de transition de state à next_state
    """
    if state in self.mdp.terminal_states : 
      return dict()
    
    action_list = []

    for exp in self.list_exp : 
      current_state, _ = self.execute_action_chain(exp)
      if current_state == state : 
        action_list = exp 
    
    """if state != 0 and len(action_list) == 0 :
      return np.argmax(PrioritizedReplayAgent.transition_rate_from_state(env,state))"""

      
    possible_state_to_transit = dict()
    for action in range(self.mdp.action_space.n) : 
      self.mdp.reset()
      _, _ = self.execute_action_chain(action_list)
      next_state, _, _, _, _ = self.mdp.step(action)
      if next_state != state : 
        possible_state_to_transit[(action,next_state)] = self.transition_rate_from_state(state)[next_state]

    return possible_state_to_transit


  """ ===================== SR USING V FUNCTION ===================== """
  def SR_v_function(self,state) : 
    """ Retourne la value function for Successor Representation

        Arguments
        ----------
            env -- Mdp from mazemdp.mdp : environnement sur lequel travailler
            state -- int : état demandé
        
        Returns
        ----------      
        V(s) = Sum{ M(s,s') * R(s') }
    """
    possible_state_to_transit = self.transition_recommandation(state)
    V_s = 0

    for (action, next_state), SR in possible_state_to_transit.items() : 
      V_s += SR * self.list_state_reward_SR[next_state]

    return V_s

  def updateQValue(self, x, a, y, r) : 
    """ Mets à jour le modele 

        Arguments
        ---------
            x -- int : etat d'origine
            a -- int : action effectue
            y -- int : etat d'arrivee
            r -- float : recompense recue
        
        Returns
        ----------      
            q_table[x,a] + alpha*(r+mdp.gamma*v_y-q_table[x,a])
    """
    #v_y correspond à la valeur maximal estimee pour l'etat y, multiplication par 1-terminated pour s'assurer de
    #ne prendre en compte ce resultat que si l'etat y n'est pas successeur d'un etat terminal
    terminated = x in self.mdp.terminal_states
    v_y = self.SR_v_function(x)                 # difference avec celui de PrioritizedReplayAgent
    return  self.QTable[x,a] + self.alpha*(r + self.mdp.gamma * v_y - self.QTable[x,a])

  """ ===================== SR IN TD DIFFERENCE LEARNING - SARSA TD ===================== """
  def onehot(self,state) : 
    vec = np.zeros(self.mdp.nb_states)
    vec[state] = 1
    return vec

  def SR_TD_error(self, current_state, next_state, terminated) : 
    """ Retourne la différence temporelle utilisée pour la Successor Representation

        Arguments
        ----------
            env -- Mdp from mazemdp.mdp : environnement sur lequel travailler
            state -- int : état demandé
            next_state -- état suivant
            terminated -- bool : pour déterminer si on atteint l'état final
        
        Returns
        ----------      
        TD_error = I[s_t = s'] + gamma * M_hat(s_t+1, s') - M_hat(s_t,s')
    """
    I = self.onehot(current_state)
    if terminated : 
      SR_current_st = self.transition_rate_from_state(current_state)
      TD_error = (I + self.mdp.gamma * self.onehot(next_state) - SR_current_st)
    else : 
      SR_current_st = self.transition_rate_from_state(current_state)
      SR_next_st = self.transition_rate_from_state(next_state)
      TD_error = (I + self.mdp.gamma * SR_next_st - SR_current_st)
    return np.mean(np.abs(TD_error))



