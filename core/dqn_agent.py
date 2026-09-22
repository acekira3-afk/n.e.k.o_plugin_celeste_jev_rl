"""DQN agent for Celeste gameplay.

Deep Q-Network with:
  - 3 conv layers + 2 linear layers (matching Madeline's Policy Climb)
  - Replay buffer
  - Target network
  - Epsilon-greedy exploration
  - Optional Jev reward shaping
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CelesteDQN(nn.Module):
    """DQN architecture matching the Madeline's Policy Climb paper.

    Input: 26-dim state vector
    Output: Q-values for 128 actions (7 keys, 2^7 combos)
    """

    def __init__(self, state_dim: int = 26, action_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN agent with optional Jev reward shaping."""

    def __init__(
        self,
        state_dim: int = 26,
        action_dim: int = 128,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update: int = 10,
        logger=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update
        self.logger = logger

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = CelesteDQN(state_dim, action_dim).to(self.device)
        self.target_net = CelesteDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        self.episode_count = 0
        self.train_step = 0
        self.losses: list[float] = []

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step_update(self) -> float | None:
        """One gradient step on a batch from replay buffer."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(1)[0]
            target_q = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.train_step += 1
        self.losses.append(loss.item())

        return loss.item()

    def end_episode(self):
        """Call at the end of each episode."""
        self.episode_count += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        if self.episode_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            if self.logger:
                self.logger.info(
                    "DQN episode=%d epsilon=%.3f target_updated, avg_loss=%.4f",
                    self.episode_count,
                    self.epsilon,
                    np.mean(self.losses[-100:]) if self.losses else 0,
                )

    def save(self, path: str):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "episode_count": self.episode_count,
            "train_step": self.train_step,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.episode_count = checkpoint["episode_count"]
        self.train_step = checkpoint["train_step"]

    def get_stats(self) -> dict[str, Any]:
        return {
            "episodes": self.episode_count,
            "train_steps": self.train_step,
            "epsilon": self.epsilon,
            "buffer_size": len(self.buffer),
            "avg_loss": float(np.mean(self.losses[-100:])) if self.losses else 0.0,
            "device": str(self.device),
        }
