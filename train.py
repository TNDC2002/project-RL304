# environment
import pickle
import random
import numpy as np
import torch
from Agent.env import CustomEnv
from dqn_agent import DQNAgent


env = CustomEnv()
seed = 2968686879

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

np.random.seed(seed)
random.seed(seed)
seed_torch(seed)
# parameters
step_number = 14268
memory_size = 14268
batch_size = 4
target_update = 3

# train
agent = DQNAgent(env, memory_size, batch_size, target_update, seed)
# agent.load_model()



# train
i = 0
max_inventory: float = 1

from datetime import datetime

from tqdm import tqdm
# for i in tqdm(range(100000)):
#     action = 1 #random.randint(0, 1)
#     env.step(1)
#     # agent.env.step(action)


while i < 100:
    i+=1
    agent.train()
    scores, losses, portfolio, inventory = agent.get_plot_data()
    if max(inventory) >= max_inventory:
        max_inventory = max(inventory)
        now = datetime.now()
        timestamp = now.strftime("%d-%m-%Y-%H-%M")
        filename = "CONV1_dropout20_agent_"+str(timestamp)+"max_inv="+str(max_inventory)+".pkl"
        agent.save_model(filename="CONV1_dropout20_agent_"+str(timestamp)+"max_inv="+str(max_inventory)+".pth")
        with open(filename, "wb") as file:
            pickle.dump(agent, file, protocol=pickle.HIGHEST_PROTOCOL)

    if i % 10 == 0:
        with open("agent.pkl", "wb") as file:
            pickle.dump(agent, file, protocol=pickle.HIGHEST_PROTOCOL)