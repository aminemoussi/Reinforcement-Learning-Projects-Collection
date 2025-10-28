import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ddqn import DDQN
from numpy.random import rand
from torch._C import device


class Agent:
    def __init__(
        self,
        state_size,
        action_size,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        learning_rate=0.001,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Available device: ", self.device)

        # neural nets
        self.model = DDQN(state_size, action_size).to(
            self.device
        )  # student or online network
        self.target_model = DDQN(state_size, action_size).to(self.device)

        # initialize target model with online model's weights
        self.update_target_model()

        # adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # replay memory allocation
        self.memory = deque(maxlen=2000)

    def _build_model(self):
        """Build DQN model"""
        return DDQN(self.state_size, self.action_size).to(self.device)

    def update_target_model(self):
        """Copy weights from main to target"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """train model on a batch of previous samples"""
        if len(self.memory) < batch_size:
            return  # not enough yet

        mini_batch = random.sample(self.memory, batch_size)

        # extracting states + actions + rewards + next states + done
        states = torch.FloatTensor([e[0] for e in mini_batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in mini_batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in mini_batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in mini_batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in mini_batch]).to(self.device)

        # get current q vals
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

        # getting next q val using DDQN
        with torch.no_grad():
            # main net to get best action for next states
            next_actions = self.model(next_states).max(1)[1]

            # target net to get evaluate the actions
            next_q_vals = (
                self.target_model(next_states)
                .gather(1, next_actions.unsqueeze(1))
                .squeeze(1)
            )

        # target q vals
        target_q_vals = rewards + (self.gamma * next_q_vals * ~dones)

        # compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_vals)

        # optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # decaying epsilon  to exploit more
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        """choose action with epsilon greedy"""
        if np.random.rand() <= self.epsilon:  # explore
            return random.randrange(self.action_size)
        else:  # exploit
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return np.argmax(q_values.cpu().data.numpy())

    def save_model(self, filepath):
        """Save model weights"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            filepath,
        )

    def load_model(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
