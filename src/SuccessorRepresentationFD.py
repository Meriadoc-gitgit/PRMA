"""
-------------------

Contient l'intégralité du code de la Successor Representation

-------------------
"""
import numpy as np
from heapq import heappop, heappush
from utils import onehot
from FocusedDyna import FocusedDyna

class SuccessorRepresentationFD(FocusedDyna) : 
  def __init__(self, mdp, alpha, delta, epsilon, episode, max_step, train_episode_length, test_episode_length) : 
    """ Initialisation de la classe FocusedDynaSR

        Arguments
        ---------
            mdp -- Mdp de mazemdp.mdp
            alpha -- float : taux d'apprentissage
            epsilon -- float : taux d'exploration pour e-greedy
            episode -- int : nombre d'épisode pour l'apprentissage
            train_episode_length -- int : nombre d'épisode pour la phase d'apprentissage
            test_episode_length -- int : nombre d'épisode pour la phase test
        
        Returns
        ----------      
    """
    super().__init__(mdp, alpha, delta, epsilon, max_step,render=False, episode=episode)
    # self.alpha = alpha
    self.train_episode_length = train_episode_length
    self.test_episode_length = test_episode_length

    # La Successor Representation
    self.M = np.stack([np.identity(mdp.unwrapped.nb_states) for i in range(mdp.action_space.n)])

    self.experiences = []
    self.lifetime_td_errors = []  #a quoi ca sert?
    self.test_lengths = []

    self.path_length_from_start()


  def Q_estimates(self, state) : 
    """ Generer Q values pour toutes actions.

        Arguments
        ---------
            state -- int : état courant
        
        Returns
        ----------   
        Liste de taille action_space.n correspondant à state
    """
    goal = onehot(self.mdp.unwrapped.terminal_states, self.mdp.unwrapped.nb_states)
    
    return np.matmul(self.M[:,state,:],goal)

  def sample_action(self, state) : 
    """ Choisir une action suivant l'approche epsilon-greedy

        Arguments
        ---------
            state -- int : état courant
        
        Returns
        ----------   
        action choisie
    """
    
    if np.random.uniform(0, 1) < self.epsilon:
      action = np.random.randint(self.mdp.action_space.n)
    else:
      Qs = self.Q_estimates(state)
      action = np.argmax(Qs)
    return action
  
  def update_sr(self, current_exp, next_exp) : 
    """ SARSA TD learning rule

        Arguments
        ---------
            current_exp -- (state, action, next_state,reward,terminated) : expérience courant
            next_exp -- (state, action, next_state,reward,terminated) : expérience suivant
        
        Returns
        ----------   
        1 array de td_error
    """
    state, action, next_state,reward,terminated = current_exp
    _,next_action,_,_,_ = next_exp

    I = onehot(state, self.mdp.unwrapped.nb_states)
    if terminated : 
      td_error = I + self.mdp.unwrapped.gamma * onehot(next_state, self.mdp.unwrapped.nb_states) - self.M[action,state]

    else : 
      td_error = I + self.mdp.unwrapped.gamma * self.M[next_action,next_state] - self.M[action,state]
    self.M[action,state] += self.alpha * td_error 
    return td_error


  def train_phase(self) : 
    """ Phase d'apprentissage

        Arguments
        ---------
        
        Returns
        ----------      
    """
    state, _ = self.mdp.reset()
    episodic_error = []

    for i in range(self.train_episode_length) : 
      action = self.sample_action(state) 
      next_state, reward, terminated, truncated,_ = self.mdp.step(action)
      self.experiences.append([state, action, next_state, reward, terminated])
      state = next_state
      if i > 1 : 
        td_sr = self.update_sr(self.experiences[-2],self.experiences[-1])
        episodic_error.append(np.mean(np.abs(td_sr)))

      if terminated : 
        #print("end")
        td_sr = self.update_sr(self.experiences[-1], self.experiences[-1])
        episodic_error.append(np.mean(np.abs(td_sr)))
        break
      
    self.lifetime_td_errors.append(np.mean(episodic_error))

  def test_phase(self) : 
    """ Phase de test

        Arguments
        ---------
        
        Returns
        ----------      
    """
    state, _ = self.mdp.reset()
    for i in range(self.test_episode_length) : 
      action = self.sample_action(state) 
      next_state, reward, terminated, truncated,_ = self.mdp.step(action)
      state = next_state
      #if terminated : 
      if next_state in self.mdp.unwrapped.terminal_states :
        self.test_lengths.append(i+1)
        break

  def trial(self) : 
    """ Excécution de FocusedDynaSR

        Arguments
        ---------
        
        Returns
        ----------      
    """
    for i in range(self.episode) : 
      self.train_phase()
      self.test_phase()

  def optimal_path_length(self) : 
    """ Calculate the optimal path length

        Arguments
        ---------
        
        Returns
        ----------  
        Optimal path length
    """
    if len(self.test_lengths) == 0 : 
      print(self.mdp.unwrapped.terminal_states)
      return self.test_episode_length
    return np.min(self.test_lengths)

  
  def path_length_from_start(self) : 
    true_terminal_states = self.mdp.unwrapped.terminal_states
    for state_goal in range(1, self.mdp.unwrapped.nb_states) : 
      state, _ = self.mdp.reset()
      self.lifetime_td_errors = []
      self.test_lengths = []
      self.mdp.unwrapped.terminal_states = [state_goal]
      self.trial()
      self.stepsFromStart[state_goal] = self.optimal_path_length()
      #print(state_goal, self.stepsFromStart[state_goal])
    self.mdp.unwrapped.terminal_states = true_terminal_states 
    #derniere ligne : on a besoin de ca sinon le terminal states corresponds
    #au dernier état et non pas au réel dernier état ce qui pose problème quand 
    #on fait execute de Prioritized replay agent
    #solution pas très propre/temporaire il vaudrait mieux ne pas toucher au terminal_states
    


