# maze_env.py

import gym
from gym import spaces
import numpy as np

class MazeMDP(gym.Env):
    def __init__(self, width, height, start_states, walls, terminal_states):
        super(MazeMDP, self).__init__()
        self.width = width
        self.height = height
        self.start_states = start_states
        self.walls = walls
        self.terminal_states = terminal_states
        
        self.action_space = spaces.Discrete(4)  # 4 actions: up, down, left, right
        self.observation_space = spaces.Discrete(width * height)

        self.state = None

    def reset(self):
        self.state = np.random.choice(self.start_states)
        return self.state, {}

    def step(self, action):
        # Define the transition dynamics here
        # For simplicity, assume each action moves the agent and there's a reward
        next_state = self.state  # Update this with actual transition logic
        reward = -1
        done = self.state in self.terminal_states
        return next_state, reward, done, {}

    def render(self, mode='human'):
        # Optionally implement rendering for your environment
        pass

def register_maze_env():
    gym.envs.registration.register(
        id='MazeMDP-v0',
        entry_point='maze_env:MazeMDP',
        max_episode_steps=1000,
        kwargs={
            'width': 36,
            'height': 24,
            'start_states': [8],
            'walls': [196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
                      220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231,
                      244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
                      268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279,
                      496, 497, 498, 499,
                      520, 521, 522, 523,
                      544, 545, 546, 547,
                      568, 569, 570, 571,
                      672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683,
                      696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707,
                      720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731,
                      744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755],
            'terminal_states': [656, 657, 658, 659,
                                680, 681, 682, 683,
                                704, 705, 706, 707,
                                728, 729, 730, 731]
        }
    )

register_maze_env()
