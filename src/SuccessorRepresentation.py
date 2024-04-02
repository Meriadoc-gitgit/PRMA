"""
-------------------

Contient l'intégralité du code de la Successor Representation

-------------------
"""
import numpy as np


def onehot(value, max_value) :
  vec = np.zeros(max_value)
  vec[value] = 1
  return vec

class TabularSuccessorAgent : 
  def __init__(self, mdp, learning_rate, epsilon) : 
    self.mdp = mdp 
    self.learning_rate = learning_rate 
    self.epsilon = epsilon

    # La Successor Representation
    self.M = np.stack([np.identity(mdp.nb_states) for i in range(mdp.action_space.n)])
    
  def Q_estimates(self, state) : 
    # Generate Q values for all actions.
    goal = onehot(self.mdp.terminal_states, self.mdp.nb_states)
    
    return np.matmul(self.M[:,state,:],goal)

  def sample_action(self, state) : 
    # Samples action using epsilon-greedy approach
    if np.random.uniform(0, 1) < self.epsilon:
      action = np.random.randint(self.mdp.action_space.n)
    else:
      Qs = self.Q_estimates(state)
      action = np.argmax(Qs)
    return action
  
  def update_sr(self, current_exp, next_exp) : 
    # SARSA TD learning rule
    state, action, next_state,reward,terminated = current_exp
    _,next_action,_,_,_ = next_exp

    I = onehot(state, self.mdp.nb_states)
    if terminated : 
      td_error = I + self.mdp.gamma * onehot(next_state, self.mdp.nb_states) - self.M[action,state]

    else : 
      td_error = I + self.mdp.gamma * self.M[next_action,next_state] - self.M[action,state]
    self.M[action,state] += self.learning_rate * td_error 
    return td_error


class FocusedDynaSR : 
  def __init__(self, mdp, learning_rate, epsilon, episode, train_episode_length, test_episode_length) : 
    self.mdp = mdp
    self.learning_rate = learning_rate
    self.epsilon = epsilon
    self.episode = episode 
    self.train_episode_length = train_episode_length
    self.test_episode_length = test_episode_length

    self.agent = TabularSuccessorAgent(mdp, learning_rate,epsilon)
    self.experiences = []
    self.lifetime_td_errors = []
    self.test_lengths = []

  def train_phase(self) : 
    state, _ = self.mdp.reset()
    episodic_error = []

    for i in range(self.train_episode_length) : 
      action = self.agent.sample_action(state) 
      next_state, reward, terminated, truncated,_ = self.mdp.step(action)
      self.experiences.append([state, action, next_state, reward, terminated])
      state = next_state
      if i > 1 : 
        td_sr = self.agent.update_sr(self.experiences[-2],self.experiences[-1])
        episodic_error.append(np.mean(np.abs(td_sr)))

      if terminated : 
        td_sr = self.agent.update_sr(self.experiences[-1], self.experiences[-1])
        episodic_error.append(np.mean(np.abs(td_sr)))
        break
      
    self.lifetime_td_errors.append(np.mean(episodic_error))

  def test_phase(self) : 
    state, _ = self.mdp.reset()
    for i in range(self.test_episode_length) : 
      action = self.agent.sample_action(state) 
      next_state, reward, terminated, truncated,_ = self.mdp.step(action)
      state = next_state
      if terminated : 
        self.test_lengths.append(i)
        break

  def execute(self) : 
    for i in range(self.episode) : 
      self.train_phase()
      self.test_phase()
    

