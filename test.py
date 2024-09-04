import os
from typing import Dict, Tuple
import gym
import numpy as np
import torch
from tqdm import tqdm
from model import Network
from priority_replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


class DQNAgent:
    """DQN Agent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(
        self,
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        seed: int,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
        # N-step 
        
        n_step: int = 3,
        model_path: str = None,
        Net: Network = Network,
        
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
            n_step (int): step number to calculate n-step td error
        """
        # input_shape = (1386, 33)
        # obs_dim = input_shape[1]

        obs_dim = 1386
        action_dim = env.action_space.n

        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.seed = seed
        self.gamma = gamma
        self.lr = 0.001
        self.episode = 0

        
        # ploting data
        self.inventory = []
        self.portfolio = []
        self.losses = []
        self.scores = []


        # NoisyNet: All attributes related to epsilon are removed

        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha=alpha, gamma=gamma
        )

        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma
            )

        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # networks: dqn, dqn_target
        pretrained_state_dict = torch.load(model_path)
        Pretrained_dqn = pretrained_state_dict['dqn_state_dict']
        Pretrained_target_dqn = pretrained_state_dict['dqn_target_state_dict']
    
    # Load state dicts for both models
        
        self.dqn = Net(
            obs_dim, action_dim, self.atom_size, self.support, #pretrained_model = Pretrained_dqn
        ).to(self.device)

        self.dqn_target = Net(
            obs_dim, action_dim, self.atom_size, self.support, #pretrained_model = Pretrained_target_dqn
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(),lr=self.lr)

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False
    def Update_Hyperparameter(self,new_batch_size=4,new_target_update=3,new_gamma=0.99,new_lr=0.001,new_n_step=3):
            self.batch_size = new_batch_size
            self.target_update = new_target_update
            self.gamma = new_gamma
            n_step = new_n_step
            self.lr = new_lr
            self.optimizer = optim.Adam(self.dqn.parameters(),lr=new_lr)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        selected_action = self.dqn(
            torch.FloatTensor(state).to(self.device)
        ).argmax()
        selected_action = selected_action.detach().cpu().numpy()

        
        
        # if selected_action == 0:
        #     selected_action = 1
            
        #     return selected_action
        if not self.is_test:
            self.transition = [state, selected_action]
        # print("_________________________________________________________________________")
        # print("selected: ",selected_action)
        # print("_________________________________________________________________________")
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, portfolio, item= self.env.step(action)
        
        self.portfolio.append(portfolio)
        if len(item) != 0:
                self.inventory.append(item[0])
        elif len(self.inventory) != 0:
                self.inventory.append(self.inventory[-1])
        else:
                self.inventory.append(0)
        
        
            
        done = terminated

        if not self.is_test:
            self.transition += [reward, next_state, done]

            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)

        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]

        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()#loss_for_prior comprise of nan
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()

    def train(self):
        """Train the agent."""
        self.is_test = False

        state = self.env.reset()
        update_cnt = 0
        score = 0
        step = 0
        # while True:
        train_progress = tqdm(range(14258), desc="Training", position=0, leave=True)
        for step in train_progress:
            # step += 1
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            

            state = next_state
            score += reward

            self.scores.append(score*100)
            # NoisyNet: removed decrease of epsilon

            # PER: increase beta
            fraction = min((step+1) / 14258, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)
            # print("episode: ", episode)
                    

            # if training is ready
            if len(self.memory) >= self.batch_size: #batch size of replay memory
                loss = self.update_model()
                # update loss in train_progress
                train_progress.set_postfix({"loss": loss,"Portfolio":self.portfolio[-1],"Inventory":self.inventory[-1]})
                self.losses.append(loss)
                update_cnt += 1

                # if hard update is needed
                if update_cnt % self.target_update == 0: #update rate
                    self._target_hard_update()

            # plotting
            if (step+1) % 200 ==0:
                self._plot(self.episode,n_score = 100, n_portfolio=14258)
            
            # if episode ends
            if done:
                state = self.env.reset()
                score = 0
                self.episode += 1
                return
                # scores = []
                # losses = []
                # if episode % 3 == 0:
                #     self.save_model(episode=episode)



    # def test(self) -> None:
    #     """Test the agent."""
    #     self.is_test = False

    #     # for recording a video

    #     state = self.env.reset()
    #     done = False
    #     score = 0

    #     while not done:
    #         action = self.select_action(state)
    #         next_state, reward, done = self.step(action)

    #         state = next_state
    #         score += reward

    #     print("score: ", score)
    def test(self, path:str = "/home/ece/anaconda3/bin/project-RL304/project-RL304/Models/CONV1_agent_19-04-2024-20-37max_inv=0.4956216011781756.pth"):
        """Test the agent."""
        self.is_test = True
        self.dqn.load_state_dict(torch.load(path))
        state = self.env.reset()
        ori_portfo_len = len(self.portfolio)
        ori_scores_len = len(self.scores)
        score = 0
        done = False
        
        test_progress = tqdm(range(14258), desc="Testing", position=0, leave=True)
        for step in test_progress:
            # step += 1
            action = self.select_action(state)
            next_state, reward, done = self.step(action)
            
            
            state = next_state
            score += reward

            self.scores.append(score*100)
            # update loss in train_progress
            test_progress.set_postfix({"Portfolio":self.portfolio[-1],"Inventory":self.inventory[-1]})
                    
            # plotting
            if (step+1) % 200 ==0:
                self._plot(self.episode,n_score = 100, n_portfolio=14258)
            
            # if episode ends
            if done:
                state = self.env.reset()
                score = 0
                self.portfolio = self.portfolio[:-(len(self.portfolio)-ori_portfo_len)]
                self.inventory = self.inventory[:-(len(self.inventory)-ori_portfo_len)]
                self.scores = self.scores[:-(len(self.scores)-ori_scores_len)]
                return
            
    def save_model(self, directory="/home/ece/anaconda3/bin/project-RL304/project-RL304/Models",filename = ""):
        """Save the model parameters to a file."""
        filepath = os.path.join(directory, filename)
        torch.save({
            'dqn_state_dict': self.dqn.state_dict(),
            'dqn_target_state_dict': self.dqn_target.state_dict(),
        }, filepath)

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())


    def _plot(
        self,
        episode: int,
        n_score:int,
        n_portfolio:int
    ):
        """Plot the training progresses."""
        if len(self.portfolio) < 14258:
            n_portfolio = len(self.portfolio)

        clear_output(True)
        plt.figure(figsize=(15, 10))  # Adjusted figure size
        
        if len(self.losses) > 0:
            plt.subplot(221)  # Subplot 1
            plt.title('episode %s. score: %s' % (episode, np.mean(self.scores[-10:])))
            plt.plot(self.scores[-n_score:])
            plt.subplot(222)  # Subplot 2
            plt.title('loss')
            plt.plot(self.losses[-n_score:])#
        
        plt.subplot(223)  # Subplot 3
        plt.title('episode %s. portfolio: %s' % (episode, self.portfolio[-1]))
        plt.plot(self.portfolio[-n_portfolio:])
        
        plt.subplot(224)  # Subplot 4
        plt.title('episode %s. inventory: %s' % (episode, np.mean(self.inventory[-10:])))
        plt.plot(self.inventory[-n_portfolio:])#
        
        plt.tight_layout()  # Adjust subplots to fit into the figure
        plt.show()
    def _plot_all(
        self,
    ):
        self._plot(episode=1,n_score=len(self.scores),n_portfolio=len(self.portfolio))
    def get_plot_data(self):
        return self.scores, self.losses, self.portfolio, self.inventory
    def clear_plot_history(self,ALL = False,n_score = 0, n_losses = 0, n_portfolio=0, n_inventory=0):
        if ALL:
            del self.scores
            del self.losses
            del self.portfolio
            del self.inventory
            return
        self.scores = self.scores[n_score:]
        self.losses = self.losses[n_losses:]
        self.portfolio = self.portfolio[n_portfolio:]
        self.inventory = self.inventory[n_portfolio:]
    
    