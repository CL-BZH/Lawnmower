import numpy as np
import random
from collections import deque
from typing import Tuple, List, Dict, Optional
import pickle
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """DQN with improved architecture for spatial awareness"""
    
    def __init__(self, input_size: int, output_size: int = 8,  # No default for input_size
                 hidden_size: int = 128):
        super().__init__()
        
        # Network architecture
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size)
        )
        
        print(f"  Created DQN network with input_size={input_size}, hidden_size={hidden_size}")
        
    def forward(self, state):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Proper dueling architecture
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        if q_values.shape[0] == 1 and len(state.shape) == 2 and state.shape[0] == 1:
            q_values = q_values.squeeze(0)
            
        return q_values


class ReplayBuffer:
    """Experience replay buffer"""
    
    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.stack([s[0] for s in batch])
        actions = torch.tensor([s[1] for s in batch], dtype=torch.long)
        rewards = torch.tensor([s[2] for s in batch], dtype=torch.float32)
        next_states = torch.stack([s[3] for s in batch])
        dones = torch.tensor([s[4] for s in batch], dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent with auto-detected state size, warmup period, and stability improvements
    """
    
    def __init__(self, 
                 state_size: int = None,  # Optional - will be auto-detected
                 action_size: int = 8,
                 learning_rate: float = 0.00025,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.1,
                 epsilon_decay: float = 0.997,
                 tau: float = 0.01,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device)
        
        # Initialize networks later when state_size is known
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        
        # Experience replay
        self.memory = ReplayBuffer(capacity=20000)
        
        # Training parameters
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = 64
        
        # Warmup period
        self.warmup_steps = 1000
        self.total_steps = 0
        
        # Training tracking
        self.episode = 0
        self.best_reward = -float('inf')
        self.best_coverage = 0
        self.best_efficiency = 0
        self.best_steps = float('inf')
        self.episodes_without_improvement = 0
        self.loss_history = []
        
        print(f"Initializing DQN Agent:")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Epsilon: {epsilon_start} -> {epsilon_end} (decay: {epsilon_decay})")
        print(f"  Gamma: {gamma}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Warmup steps: {self.warmup_steps}")
        
        # Store learning rate for later initialization
        self.learning_rate = learning_rate
        
        if state_size:
            self._initialize_networks(state_size, learning_rate)
        else:
            print(f"  State size: Will be auto-detected from first state")
    
    def _initialize_networks(self, state_size: int, learning_rate: float = 0.00025):
        """Initialize networks once state size is known"""
        self.state_size = state_size
        self.policy_net = DQNNetwork(state_size, self.action_size, hidden_size=128).to(self.device)
        self.target_net = DQNNetwork(state_size, self.action_size, hidden_size=128).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Set networks to training mode initially
        self.policy_net.train()
        self.target_net.train()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                   lr=learning_rate)
        
        print(f"  Networks initialized with state size: {state_size}")
    
    def get_state(self, env, state) -> torch.Tensor:
        """State representation based on reachable cells"""
        features = []
        
        x, y = state.position
        view_size = 5
        radius = view_size // 2
        
        # 1. Local 5x5 grid
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                pos = (x + dx, y + dy)
                if env.is_valid_position(pos):
                    if pos in state.mowed_cells:
                        features.append(0.0)  # Mowed
                    elif pos in env.reachable_lawn_cells:
                        features.append(1.0)  # Unmowed but reachable
                    else:
                        features.append(-1.0)  # Obstacle or unreachable
                else:
                    features.append(-1.0)  # Out of bounds
        
        # 2. Progress based on REACHABLE cells
        progress = env.calculate_progress(state)
        features.append(progress)
        
        # 3. Current heading
        features.append(state.heading / 360.0)
        
        # 4. Distance to center of reachable area
        if env.reachable_lawn_cells:
            reachable_x = sum(c[0] for c in env.reachable_lawn_cells) / len(env.reachable_lawn_cells)
            reachable_y = sum(c[1] for c in env.reachable_lawn_cells) / len(env.reachable_lawn_cells)
            dist_to_reachable_center = np.sqrt((x - reachable_x)**2 + (y - reachable_y)**2)
            max_dist = np.sqrt((env.width)**2 + (env.height)**2)
            features.append(dist_to_reachable_center / max(max_dist, 1))
        else:
            features.append(0.0)
        
        # 5. Number of unmowed REACHABLE neighbors
        unmowed = env.get_unmowed_cells(state)
        neighbors = env.get_neighbors(state.position, valid_only=False)
        unmowed_neighbors = sum(1 for n in neighbors if n in unmowed)
        features.append(unmowed_neighbors / 8.0)
        
        # 6. Step count normalized
        features.append(min(state.total_steps / 1000.0, 1.0))
        
        # 7. Direction change frequency
        direction_change_rate = state.direction_changes / max(state.total_steps, 1)
        features.append(direction_change_rate)
        
        # Total features: 25 (5x5 grid) + 6 = 31 features
        state_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Initialize networks if this is the first state
        if self.policy_net is None:
            actual_state_size = state_tensor.shape[1]
            print(f"  Auto-detected state size: {actual_state_size}")
            self._initialize_networks(actual_state_size, self.learning_rate)
        
        return state_tensor
    
    def act(self, state: torch.Tensor, valid_actions: List[int], 
            training: bool = True) -> int:
        if not valid_actions:
            return 0
        
        # Epsilon-greedy
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Check if network is initialized
        if self.policy_net is None:
            print("ERROR: Network not initialized before act() call!")
            return random.choice(valid_actions)
        
        # Set to eval mode for inference
        self.policy_net.eval()
        
        with torch.no_grad():
            q_values = self.policy_net(state)
        
        # Mask invalid actions
        mask = torch.full((self.action_size,), -float('inf'), device=self.device)
        for a in valid_actions:
            mask[a] = 0
        q_values = q_values + mask
        
        # Set back to train mode
        if training:
            self.policy_net.train()
        
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        # Warmup period: don't train until we have enough experiences
        if self.total_steps < self.warmup_steps:
            return 0.0
        
        # Check if network is initialized
        if self.policy_net is None:
            print("WARNING: Attempted to train before network initialization")
            return 0.0
        
        if len(self.memory) < self.batch_size:
            return 0.0
        
        sample = self.memory.sample(self.batch_size)
        if sample is None:
            return 0.0
            
        states, actions, rewards, next_states, dones = sample
        
        # Ensure networks are in train mode
        self.policy_net.train()
        self.target_net.train()
        
        # Double DQN
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            
            # Clip target values for stability
            target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (1 - dones.unsqueeze(1))
            target_q_values = torch.clamp(target_q_values, -100, 100)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Huber loss for stability
        loss = F.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Soft update target network
        self.soft_update_target()
        
        return loss.item()
    
    def soft_update_target(self):
        if self.policy_net is None or self.target_net is None:
            return
            
        for target_param, policy_param in zip(self.target_net.parameters(), 
                                            self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + 
                                  (1 - self.tau) * target_param.data)
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, 
                          self.epsilon * self.epsilon_decay)
    
    def increment_steps(self):
        self.total_steps += 1
    
    def update_stats(self, coverage: float, efficiency: float, reward: float, steps: int, loss: float):
        self.episode += 1
        self.loss_history.append(loss)
        
        # Update best based on reward
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_coverage = coverage
            self.best_efficiency = efficiency
            self.best_steps = steps
            self.episodes_without_improvement = 0
            return True
        else:
            self.episodes_without_improvement += 1
            return False
    
    def should_stop_early(self, patience: int = 200) -> bool:
        return self.episodes_without_improvement >= patience
    
    def save(self, filepath: str):
        if self.policy_net is None:
            print("ERROR: Cannot save model - network not initialized")
            return
            
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': self.episode,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'best_coverage': self.best_coverage,
            'best_efficiency': self.best_efficiency,
            'best_steps': self.best_steps,
            'loss_history': self.loss_history,
            'state_size': self.state_size,
            'action_size': self.action_size,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # Get state size from checkpoint
            state_size = checkpoint.get('state_size', None)
            if state_size is None:
                print("ERROR: No state size in checkpoint")
                return False
            
            # Initialize networks with correct size
            self._initialize_networks(state_size, self.learning_rate)
            
            # Load state dicts
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load other parameters
            self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
            self.episode = checkpoint.get('episode', 0)
            self.total_steps = checkpoint.get('total_steps', 0)
            self.best_reward = checkpoint.get('best_reward', -float('inf'))
            self.best_coverage = checkpoint.get('best_coverage', 0)
            self.best_efficiency = checkpoint.get('best_efficiency', 0)
            self.best_steps = checkpoint.get('best_steps', float('inf'))
            self.loss_history = checkpoint.get('loss_history', [])
            
            print(f"Model loaded from {filepath}")
            print(f"  Episode: {self.episode}")
            print(f"  Total steps: {self.total_steps}")
            print(f"  Best reward: {self.best_reward:.1f}")
            print(f"  Best coverage: {self.best_coverage:.1f}%")
            print(f"  Best efficiency: {self.best_efficiency:.1f}%")
            print(f"  Best steps: {self.best_steps}")
            print(f"  Epsilon: {self.epsilon:.3f}")
            return True
        return False


def calculate_reward(env, old_state, new_state, action, step_count, max_steps) -> float:
    """
    Reward function that properly rewards completing all mowable cells efficiently
    """
    reward = 0.0
    
    # Get reachable cells (cells that can actually be mowed)
    reachable_cells = env.reachable_lawn_cells
    
    # 1. Reward for mowing new REACHABLE cells
    old_mowed_reachable = len(old_state.mowed_cells.intersection(reachable_cells))
    new_mowed_reachable = len(new_state.mowed_cells.intersection(reachable_cells))
    cells_mowed = new_mowed_reachable - old_mowed_reachable
    
    if cells_mowed > 0:
        # Base reward for mowing
        reward += cells_mowed * 5.0
        
        # Bonus for mowing multiple cells in one move
        if cells_mowed > 1:
            reward += cells_mowed * 3.0
    
    # 2. Small penalty for each step (encourages efficiency)
    reward -= 0.2
    
    # 3. Bonus for consistent direction (going straight)
    if new_state.heading == old_state.heading:
        reward += 0.3
    
    # 4. Penalty for direction changes
    if new_state.direction_changes > old_state.direction_changes:
        reward -= 0.5
    
    # 5. LARGE BONUS FOR COMPLETING ALL MOWABLE CELLS
    if env.is_complete(new_state):
        completion_bonus = 200.0  # Large fixed bonus
        
        # Efficiency bonus - fewer steps is better
        efficiency = env.calculate_efficiency(new_state)
        completion_bonus += efficiency * 1.0
        
        # Early completion bonus
        optimal_steps_estimate = len(reachable_cells) * 0.6  # Estimate optimal steps
        if new_state.total_steps < optimal_steps_estimate:
            early_completion_bonus = (optimal_steps_estimate - new_state.total_steps) * 2.0
            completion_bonus += early_completion_bonus
        
        reward += completion_bonus
        
        # Debug output
        print(f"  COMPLETION! Steps: {new_state.total_steps}, Bonus: {completion_bonus:.1f}")
    
    # 6. Progress-based reward
    progress = env.calculate_progress(new_state)
    if progress > 0.95:  # Very close to completion
        reward += 10.0
    elif progress > 0.8:  # Close to completion
        reward += 5.0
    elif progress > 0.5:  # Made good progress
        reward += 2.0
    
    # 7. Exploration bonus for finding new unmowed areas
    unmowed = env.get_unmowed_cells(new_state)
    if unmowed:
        # Reward for moving toward unmowed areas
        current_pos = new_state.position
        closest_unmowed_dist = min(abs(cell[0] - current_pos[0]) + abs(cell[1] - current_pos[1]) 
                                  for cell in unmowed)
        
        old_pos = old_state.position
        old_closest_dist = min(abs(cell[0] - old_pos[0]) + abs(cell[1] - old_pos[1]) 
                              for cell in unmowed)
        
        if closest_unmowed_dist < old_closest_dist:
            reward += 1.0  # Reward for getting closer to unmowed cells
    
    # 8. Penalty for staying in fully mowed areas
    if step_count > 20:
        mowed_around = 0
        total_around = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                pos = (new_state.position[0] + dx, new_state.position[1] + dy)
                if env.is_valid_position(pos) and pos in reachable_cells:
                    total_around += 1
                    if pos in new_state.mowed_cells:
                        mowed_around += 1
        
        if total_around > 0:
            mowed_ratio = mowed_around / total_around
            if mowed_ratio > 0.9:  # Almost everything around is mowed
                reward -= 1.0
    
    # Clip reward to prevent instability but allow completion bonus
    reward = np.clip(reward, -5.0, 250.0)
    
    return reward