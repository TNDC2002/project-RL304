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
        self.state = self.data.iloc[self.timestep:self.timestep+42]
        self.tempstate = self.data.iloc[self.timestep:self.timestep+42]
        drop = ['timestamp_o', 'timestamp_cl', 'ignore']
        self.state.drop(columns=drop, inplace=True)
        
        self.von = 1000
        self.portfolio = 1000
        self.inventory = []
        # print("env: ",self.state )
    def reset(self):
        self.timestep = 200
        self.state = self.data.iloc[self.timestep:self.timestep+42]
        self.tempstate = self.data.iloc[self.timestep:self.timestep+42]
        drop = ['timestamp_o', 'timestamp_cl', 'ignore']
        self.state.drop(columns=drop, inplace=True)
        
        self.von = 1000
        self.portfolio = 1000
        self.inventory = []
        return self.state.to_numpy().reshape(1,-1)

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"
        print("=================================================================")

        # Perform action and update state
        if action == 1 and self.von > 0:
            self.inventory.append((self.von/10)/self.state.tail(1)['cl'].item())
            self.von -= self.von/10
            self.timestep += 1
            self.state = self.data.iloc[self.timestep:self.timestep+42]
            self.tempstate = self.data.iloc[self.timestep:self.timestep+42]
            # print(pd.to_datetime(self.state.tail(1)['timestamp_o'], unit='ms') )
            drop = ['timestamp_o', 'timestamp_cl', 'ignore']
            self.state.drop(columns=drop, inplace=True)
            
        elif action == 2 and len(self.inventory) != 0: 
            self.von += self.inventory[-1] * self.state.tail(1)['cl'].item()
            self.inventory = self.inventory[:-1]       
            self.timestep += 1
            self.state = self.data.iloc[self.timestep:self.timestep+42]
            self.tempstate = self.data.iloc[self.timestep:self.timestep+42]
            # print(pd.to_datetime(self.state.tail(1)['timestamp_o'], unit='ms') )
            drop = ['timestamp_o', 'timestamp_cl', 'ignore']
            self.state.drop(columns=drop, inplace=True)
        else:
            self.timestep += 1
            self.state = self.data.iloc[self.timestep:self.timestep+42]
            self.tempstate = self.data.iloc[self.timestep:self.timestep+42]
            # print(pd.to_datetime(self.state.tail(1)['timestamp_o'], unit='ms') )
            drop = ['timestamp_o', 'timestamp_cl', 'ignore']
            self.state.drop(columns=drop, inplace=True)
        # Define reward based on the new state
        reward = self._calculate_reward()

        # Check if episode is done
        done = self._is_done()

        # Define observation (state) for the next step
        next_observation = self.state.to_numpy().reshape(1,-1)
        # print("next: ",next_observation )
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
        
        # print('reward:',reward)
        # print('inventory:',self.inventory)
        print('portfolio:',self.portfolio)
        print("--------------------------------------------")
        return reward  # No reward otherwise

    def _is_done(self):
        # Define termination condition
        print("is done:",pd.to_datetime(self.tempstate.tail(1)['timestamp_o'], unit='ms') )
        print('timestep:', self.timestep)
        
        print(self.tempstate.tail(1)['timestamp_o'].item(), '>=', 1711339200000, '=',self.tempstate.tail(1)['timestamp_o'].item() >= 1711339200000)
        return self.tempstate.tail(1)['timestamp_o'].item() >= 1711339200000

