import gym
from gym import spaces
import numpy as np
from Utils.get_data import get_data
class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.observation_space = spaces.Discrete(5)  # Example discrete state space
        self.action_space = spaces.Discrete(3)  # Three discrete actions: 0, 1, 2
        self.timestep=200
        self.state = get_data('../Data').iloc[self.timestep:self.timestep+42]
        self.von = 1000
        self.portfolio = 1000
        self.inventory = []
    def reset(self):
        self.timestep = 200
        self.state = get_data('../Data').iloc[self.timestep:self.timestep+42]  # Reset the state
        self.von = 1000
        self.portfolio = 1000
        self.inventory = []
        return self.state

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        # Perform action and update state
        if action == 0:
            self.timestep += 1
            self.state = get_data('../Data').iloc[self.timestep:self.timestep+42]
        elif action == 1:
            print(self.state.tail(1)['cl'].item())
            self.timestep += 1
            self.state = get_data('../Data').iloc[self.timestep:self.timestep+42]
            self.inventory.append((self.von/10)/self.state.tail(1)['cl'].item())
            self.von -= self.von/10
            
        else:
            self.timestep += 1
            self.state = get_data('../Data').iloc[self.timestep:self.timestep+42]
            self.von += self.inventory[-1] * self.state.tail(1)['cl'].item()
            self.inventory = self.inventory[:-1]

        # Define reward based on the new state
        reward = self._calculate_reward()

        # Check if episode is done
        done = self._is_done()

        # Define observation (state) for the next step
        next_observation = self.state

        return next_observation, reward, done

    def _calculate_reward(self):
        portfo = 0
        for each in self.inventory:
            each *= self.state.tail(1)['cl'].item()
            portfo += each
            # print('portfo:',portfo)
        portfo += self.von
        reward = portfo - self.portfolio
        self.portfolio = portfo
        
        print('reward:',reward)
        return reward  # No reward otherwise

    def _is_done(self):
        # Define termination condition
        return self.timestep >= 500*29 + 10

