import gym
from gym import spaces
import numpy as np
from Utils.get_data import get_data
import pandas as pd
class CustomEnv(gym.Env):
    
    def __init__(self):
        super(CustomEnv, self).__init__()  # Example discrete state space
        self.action_space = spaces.Discrete(2)  # Three discrete actions: 0, 1, 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(42,33),dtype=np.float32) #spaces.Box(low=-np.inf, high=np.inf, shape=(33,), dtype=np.float32)
        
        
        self.timestep = 200
        self.data = get_data('./Data')
        self.temp = get_data('./Data')
        drop = ['timestamp_o', 'timestamp_cl', 'ignore']
        self.data.drop(columns=drop, inplace=True)
        self.state = self.data.iloc[self.timestep:self.timestep+42]
        # self.tempstate = self.temp.iloc[self.timestep:self.timestep+42]
        
        
        self.von = 1000
        self.portfolio = 1000
        self.inventory = []
        self.count = 1
    def reset(self):
        self.timestep = 200
        self.state = self.data.iloc[self.timestep:self.timestep+42]
        # self.tempstate = self.temp.iloc[self.timestep:self.timestep+42]
        self.von = 1000
        self.portfolio = 1000
        self.inventory = []
        self.count = 0
        return np.expand_dims(self.state.to_numpy(), 0)

    def step(self, action):
        assert self.action_space.contains(action), str(action)
        # print("=================================================================")
        # print(pd.to_datetime(self.tempstate.tail(1)['timestamp_o'].item(), unit='ms'))
        reward = 0
        # Perform action and update state
        if action == 1:
            if self.von > 0:
                self.inventory.append((self.von/self.state.tail(1)['cl'].item())*0.999)
            
            self.timestep += 1
            self.state = self.data.iloc[self.timestep:self.timestep+42]
            # self.tempstate = self.temp.iloc[self.timestep:self.timestep+42]

            # Define reward based on the new state
            reward += self._calculate_reward(action)
            self.von = 0
            self._calculate_portfolio()
            
        elif action == 0:
            von = 0
            if len(self.inventory) != 0:
                von += (self.inventory[-1] * self.state.tail(1)['cl'].item())*0.999
            
            self.timestep += 1
            self.state = self.data.iloc[self.timestep:self.timestep+42]
            # self.tempstate = self.temp.iloc[self.timestep:self.timestep+42]
                   

            # Define reward based on the new state
            reward += self._calculate_reward(action)
            
            # sell inventory
            self.von += von
            if len(self.inventory) != 0:
                self.inventory = self.inventory[:-1]
        
            self._calculate_portfolio()


        # Check if episode is done
        self.count +=1
        done = self._is_done()
        # Define observation (state) for the next step
        # next_observation = self.state.to_numpy().reshape(1,-1) # (42, 33) -> (1, 42*33)
        next_observation = np.expand_dims(self.state.to_numpy(), 0)  # (42, 33) -> (1, 42, 33)
        # if reward < 0:
        #     reward *= 1.25
        return next_observation, reward, done, self.portfolio, self.inventory

    def _calculate_reward(self, action):
        portfo = 0
        if action == 0: #sell:
            # if len(self.inventory) != 0:
            #     for each in self.inventory:
            #         each *= (self.state.tail(1)['cl'].item()*0.999)
            #         portfo += each
            #     portfo += self.von
            #     reward = (-1)* (portfo - self.portfolio)
            #     return reward
            now = self.state.iloc[-2:-1]['cl'].item()
            next_ = self.state.tail(1)['cl'].item()
            ratio = (next_ - now)/now
            reward = (-1)*(ratio)


        elif action == 1:
            # if self.von != 0:
            #     for each in self.inventory:
            #         each *= (self.state.tail(1)['cl'].item()*0.999)
            #         portfo += each
            #     portfo += self.von
            #     reward = portfo - self.portfolio
            #     return reward
            now = self.state.iloc[-2:-1]['cl'].item()
            next_ = self.state.tail(1)['cl'].item()
            ratio = (next_ - now)/now
            reward = ratio
        return reward  # No reward otherwise
        
    def _calculate_portfolio(self):

        #  calculate portfolio at t-1
        portfo = 0
        for each in self.inventory:
                each *= self.state.iloc[-2:-1]['cl'].item()
                portfo += each
        portfo += self.von
        self.portfolio = portfo

        # print('portfolio:',self.portfolio)
        # print("_________________________________________________________________")
        # print('inventory:',len(self.inventory))
        # print('inventory:',self.inventory)

    def _is_done(self):
        # Define termination condition
        
        # print((self.count/14268)*100, '%')

        return self.count == 14258 or self.portfolio<=0

