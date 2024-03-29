'''
MIT License

Copyright (c) 2018 Udacity

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

'''
This code is taken from https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-bipedal/ddpg_agent.py,
with slight modifications to it.
'''

import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from .model import Actor, Critic

BUFFER_SIZE = int(1e6)    # replay buffer size
BATCH_SIZE = 1024         # minibatch size
GAMMA = 1                 # discount factor
TAU = 1e-3                # for soft update of target parameters
LR_ACTOR = 1e-4           # learning rate of the actor
LR_CRITIC = 1e-4          # learning rate of the critic
WEIGHT_DECAY = 0          # L2 weight decay
UPDATE_EVERY = 5          # how often to update the network
N_UPDATES_PER_STEP = 10   # number of updates to perform at each learning step
CLIP_GRAD_NORM = 5        # Value of gradient to clip while training

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, bid_action_size, board_action_size, seed=None, checkpoint_path=None, initial_checkpoint_path=None):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            bid_action_size (int): Dimension of each bid action
            board_action_size (int): Dimension of each board action (number of cells in the game table)
            seed (int): Random seed of PRNG engines
            checkpoint_path (string): Directory with saved model checkpoints
        """
        self.state_size = state_size
        self.bid_action_size = bid_action_size
        self.board_action_size = board_action_size
        self.action_size = bid_action_size + board_action_size
        random.seed(seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, bid_action_size, board_action_size,
                                 fc1_units=400, fc2_units=300, seed=seed).to(device)
        self.actor_target = Actor(state_size, bid_action_size, board_action_size,
                                  fc1_units=400, fc2_units=300, seed=seed).to(device)
        # Make target network equal to local network
        self.hard_update(self.actor_local, self.actor_target)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(
            state_size, self.action_size, fcs1_units=400, fc2_units=300, seed=seed).to(device)
        self.critic_target = Critic(
            state_size, self.action_size, fcs1_units=400, fc2_units=300, seed=seed).to(device)
        # Make target network equal to local network
        self.hard_update(self.critic_local, self.critic_target)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Lists of losses to keep track of trainin loss
        self.actor_losses = []
        self.critic_losses = []

        # Noise process
        self.noise = OUNoise(bid_action_size, seed)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Path to save checkpoints
        self.checkpoint_path = checkpoint_path

        # Load model if initial checkpoint exists
        if initial_checkpoint_path is not None:
            print('initial_checkpoint_path: %s' % initial_checkpoint_path)
            self.load_model(initial_checkpoint_path)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                for _ in range(N_UPDATES_PER_STEP):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    def act(self, state, train_mode=False):
        """Returns actions for given state as per current policy."""
        state = state.unsqueeze(0).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if train_mode:
            actions[:, :self.bid_action_size] += self.noise.sample()
        return np.clip(actions, 0, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_losses.append(critic_loss.item())
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping to avoid exploding gradient
        torch.nn.utils.clip_grad_norm_(
            self.critic_local.parameters(), CLIP_GRAD_NORM)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_losses.append(actor_loss.item())
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.actor_local.parameters(), CLIP_GRAD_NORM)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau*local_param.data + (1.0-tau)*target_param.data)

    def hard_update(self, local_model, target_model):
        """Hard update model parameters.
        θ_target = θ_local

        Inputs:
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def load_model(self, checkpoint_path=None):
        """Load model's checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path

        checkpoint = torch.load(checkpoint_path)

        self.t_step = checkpoint['t_step']

        self.actor_local.load_state_dict(checkpoint['actor_local'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])

        self.critic_local.load_state_dict(checkpoint['critic_local'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

    def save_model(self, checkpoint_path):
        """Save model's checkpoint"""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path

        checkpoint = {
            't_step': self.t_step,
            'actor_local': self.actor_local.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_local': self.critic_local.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)

    def get_losses(self):
        """Return list of losses obtained during training steps"""
        return {
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses
        }

    def print_lr(self):
        print('LR_ACTOR:', LR_ACTOR)
        print('LR_CRITIC:', LR_CRITIC)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.2, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.random.standard_normal(x.shape)
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
                                     "state", "action", "reward", "next_state", "done"])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack(
            [e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack(
            [e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def pop(self):
        """Remove last item from buffer"""
        item = None
        if len(self.memory) > 0:
            item = self.memory.pop()
        return item

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
