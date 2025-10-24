from typing import Generator
import numpy as np

import torch

class sl_buffer:
    def __init__(self, capacity):
        """
        Supervised Learning (SL) buffer for storing state, action, and mask tuples.
        
        Args:
            capacity (int): Maximum capacity of the buffer.
        """
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.masks = []
        self.t = 0

    def __len__(self) -> int:
        return len(self.states)
    
    def store(self, state, action, mask):
        """
        Stores a state, action, and mask tuple in the buffer.

        Args:
            state (torch.Tensor): The state tensor.
            action (torch.Tensor): The action taken.
            mask (torch.Tensor): The mask tensor.
        """
        if len(self) >= self.capacity:
            j = np.random.randint(0, self.t + 1)
            if j < self.capacity:
                self.states[j] = state
                self.actions[j] = action
                self.masks[j] = mask
        else:
            self.states.append(state)
            self.actions.append(action)
            self.masks.append(mask)
        self.t += 1

    def sample(self, batch_size) -> Generator[tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        """
        Generates batches of samples from the buffer.

        Args:
            batch_size (int): The number of samples to retrieve.

        Returns:
            Generator: A generator yielding batches of (states, actions, masks).
        """
        if len(self) < batch_size:
            replace=True
        else: 
            replace=False
        while True:
            idx = np.random.choice(len(self), size=batch_size, replace=replace)
            yield (torch.stack([self.states[i] for i in idx]),
                torch.stack([self.actions[i] for i in idx]),
                torch.stack([self.masks[i] for i in idx]))

class rl_buffer:
    def __init__(self, capacity):
        """
        Reinforcement Learning (RL) buffer for storing experience tuples.

        Args:
            capacity (int): Maximum capacity of the buffer.
        """
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.next_masks = []
        
    def __len__(self):
        return len(self.states)
    
    def store(self, state, action, reward, next_state, done, next_mask):
        """
        Stores an experience tuple in the buffer.

        Args:
            state (torch.Tensor): The current state tensor.
            action (torch.Tensor): The action taken.
            reward (float): The reward received.
            next_state (torch.Tensor): The next state tensor.
            done (bool): Whether the episode has ended.
            next_mask (torch.Tensor): The mask tensor for the next state.
        """
        if len(self) >= self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            self.next_masks.pop(0)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.next_masks.append(next_mask)

    def sample(self, batch_size) -> Generator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
        """
        Generates batches of samples from the buffer.

        Args:
            batch_size (int): The number of samples to retrieve.

        Returns:
            Generator: A generator yielding batches of (states, actions, rewards, next_states, dones, next_masks).
        """
        if len(self) < batch_size:
            replace=True
        else: 
            replace=False
        while True:
            idx = np.random.choice(len(self), size=batch_size, replace=replace)
            yield (torch.stack([self.states[i] for i in idx]),
                torch.stack([self.actions[i] for i in idx]),
                torch.tensor([self.rewards[i] for i in idx], dtype=torch.float32).unsqueeze(1),
                torch.stack([self.next_states[i] for i in idx]),
                torch.tensor([self.dones[i] for i in idx], dtype=torch.float32).unsqueeze(1),
                torch.stack([self.next_masks[i] for i in idx]))

if __name__ == "__main__":
    buffer = rl_buffer(20)
    for _ in range(30):
        state = torch.rand(20)
        action = torch.rand(1)
        reward = 0
        done = 1
        mask = torch.rand(3)
        buffer.store(state, action, reward, state, done, mask)

    for sample in buffer.sample(5):
        print(sample)
