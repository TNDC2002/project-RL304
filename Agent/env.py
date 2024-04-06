import gym
from gym import spaces
import numpy as np
from Utils.get_data import get_data
import pandas as pd
class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()  # Example discrete state space
        self.action_space = spaces.Discrete(3)  # Three discrete actions: 0, 1, 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,dtype=np.float32)
        
        
        self.timestep = 200
        self.data = get_data('./Data')
        self.temp = get_data('./Data')
        drop = ['timestamp_o', 'timestamp_cl', 'ignore']
        self.data.drop(columns=drop, inplace=True)
        self.state = self.data.iloc[self.timestep:self.timestep+42]
        self.tempstate = self.temp.iloc[self.timestep:self.timestep+42]
        
        
        self.von = 1000
        self.portfolio = 1000
        self.inventory = []
    def reset(self):
        self.timestep = 200
        self.state = self.data.iloc[self.timestep:self.timestep+42]
        self.tempstate = self.temp.iloc[self.timestep:self.timestep+42]
        self.von = 1000
        self.portfolio = 1000
        self.inventory = []
        return self.state.to_numpy().reshape(1,-1)

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        print("=================================================================")

        reward = 0
        # Perform action and update state
        if action == 1 and self.von > 0 and len(self.inventory) < 10:
            
            self.inventory.append((self.von/5)/self.state.tail(1)['cl'].item())
            self.von -= self.von/5

            # Define reward based on the new state
            reward += self._calculate_reward()
            
            self.timestep += 1
            self.state = self.data.iloc[self.timestep:self.timestep+42]
            self.tempstate = self.temp.iloc[self.timestep:self.timestep+42]
            
        elif action == 2 and len(self.inventory) != 0: 
            self.von += self.inventory[-1] * self.state.tail(1)['cl'].item()
            self.inventory = self.inventory[:-1]       

            # Define reward based on the new state
            reward += self._calculate_reward()
            
            self.timestep += 1
            self.state = self.data.iloc[self.timestep:self.timestep+42]
            self.tempstate = self.temp.iloc[self.timestep:self.timestep+42]
        else:
            # Define reward based on the new state
            reward += self._calculate_reward()
            
            self.timestep += 1
            self.state = self.data.iloc[self.timestep:self.timestep+42]
            self.tempstate = self.temp.iloc[self.timestep:self.timestep+42]
        

        # Check if episode is done
        done = self._is_done()

        # Define observation (state) for the next step
        next_observation = self.state.to_numpy().reshape(1,-1)
        return next_observation, reward, done

    def _calculate_reward(self):
        portfo = 0
        for each in self.inventory:
            each *= self.state.tail(1)['cl'].item()
            portfo += each
        portfo += self.von
        reward = portfo - self.portfolio
        self.portfolio = portfo
        
        print('portfolio:',self.portfolio)
        print("--------------------------------------------")
        print('inventory:',len(self.inventory))
        print('inventory:',self.inventory)
        return reward  # No reward otherwise

    def _is_done(self):
        # Define termination condition
        
        print(((self.tempstate.tail(1)['timestamp_o'].item() - 1711339200000)*100) / 1711339200000, '%')
        return self.tempstate.tail(1)['timestamp_o'].item() >= 1711339200000

