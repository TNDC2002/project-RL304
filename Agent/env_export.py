import gym
from gym import spaces
import numpy as np
from Utils.get_data import get_data
import pandas as pd
from typing import Optional, Union
from sklearn.preprocessing import MinMaxScaler
# class CustomEnv(gym.Env):
class CustomEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    
    def __init__(self):
        super(CustomEnv, self).__init__()  # Example discrete state space
        self.action_space = spaces.Discrete(3)  # Three discrete actions: 0, 1, 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(42,33),dtype=np.float32) #spaces.Box(low=-np.inf, high=np.inf, shape=(33,), dtype=np.float32)
        
        # Initialize MinMaxScaler
        scaler = MinMaxScaler()
        
        self.prediction_expire = 6
        self.timestep = 200
        self.data = get_data('./Data')
        
        
        
        # normalized_data = scaler.fit_transform(self.data)
        # self.data = pd.DataFrame(normalized_data, columns=self.data.columns)
        
        
        # self.temp = self.data.copy()
        drop = ['timestamp_o', 'timestamp_cl', 'ignore']
        self.data.drop(columns=drop, inplace=True)
        self.state = self.data.iloc[self.timestep:self.timestep+14258]
        self.state.to_csv('unnormalized_data.csv', index=False)
        self.state_reward = self.data.iloc[self.timestep+42:self.timestep+ 42 + self.prediction_expire]
        # self.tempstate = self.temp.iloc[self.timestep:self.timestep+42]
        
        
        self.von = 1000
        self.portfolio = 1000
        self.inventory = []
        self.count = 1
    def reset(self,seed: Optional[int] = None,):
        super().reset(seed=seed)
        self.timestep = 200
        self.state = self.data.iloc[self.timestep:self.timestep+42]
        self.state_reward = self.data.iloc[self.timestep+42:self.timestep+ 42 + self.prediction_expire]
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
        if action == 0:

            reward_1 = self._calculate_reward(1)
            reward_2 = self._calculate_reward(2)

            if reward_1 < 0 and reward_2 < 0:
                reward = 0.001
            else:
                if reward_1 > reward_2:
                    reward -= reward_1
                else:
                    reward -= reward_2

            self._calculate_portfolio()

            # next state
            self.timestep += 1
            self.state = self.data.iloc[self.timestep:self.timestep+42]
            self.state_reward = self.data.iloc[self.timestep+42:self.timestep+ 42 + self.prediction_expire]

        elif action == 1:
            if self.von > 0:
                self.inventory.append((self.von/self.state.tail(1)['cl'].item())*0.999)

            # Define reward based on the new state
            reward += self._calculate_reward(action)
            self.von = 0
            self._calculate_portfolio()
            
            # next state
            self.timestep += 1
            self.state = self.data.iloc[self.timestep:self.timestep+42]
            self.state_reward = self.data.iloc[self.timestep+42:self.timestep+ 42 + self.prediction_expire]
            
        elif action == 2:
            if len(self.inventory) != 0:
                self.von += (self.inventory[-1] * self.state.tail(1)['cl'].item())*0.999
                self.inventory = self.inventory[:-1]
            # Define reward based on the new state
            reward += self._calculate_reward(action)
                   
            self._calculate_portfolio()
            
            # next state
            self.timestep += 1
            self.state = self.data.iloc[self.timestep:self.timestep+42]
            self.state_reward = self.data.iloc[self.timestep+42:self.timestep+ 42 + self.prediction_expire]
                   

            

        
        # Check if episode is done
        self.count +=1
        done = self._is_done()
        next_observation = np.expand_dims(self.state.to_numpy(), 0)  # (42, 33) -> (1, 42, 33)
        return next_observation, reward, done, self.portfolio, self.inventory

    def _calculate_reward(self, action):
        reward = 0
        if action == 2: #sell:
            now = self.state.tail(1)['cl'].item()
            next_ = np.mean(self.state_reward["cl"].to_numpy())
            ratio = (next_ - now)/now
            reward = (-1)*(ratio)


        elif action == 1:
            now = self.state.tail(1)['cl'].item()
            next_ = np.mean(self.state_reward["cl"].to_numpy())
            ratio = (next_ - now)/now
            reward = ratio
        
        # print("reward:",reward*100)
        reward -= 0.005
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

