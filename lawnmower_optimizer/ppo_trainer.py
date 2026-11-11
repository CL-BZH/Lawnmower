"""
PPO (Proximal Policy Optimization) Fine-tuning Module

This module implements PPO to fine-tune the agent after imitation learning,
allowing it to discover improvements beyond expert demonstrations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
from collections import deque
import random


class ValueNetwork(nn.Module):
    """
    Value network for estimating state values in PPO
    Shares architecture with policy network
    """
    
    def __init__(self, input_size: int, hidden_size: int = 256):
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)  # Single value output
        )
    
    def forward(self, x):
        return self.network(x)


class PPOTrajectory:
    """
    Store trajectory data for PPO training
    """
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def add(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def get(self):
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.LongTensor(self.actions),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.log_probs),
            torch.FloatTensor(self.values),
            torch.FloatTensor(self.dones)
        )


class PPOTrainer:
    """
    PPO trainer for fine-tuning the agent after imitation learning
    """
    
    def __init__(self, agent, env, 
                 learning_rate: float = 1e-4,  # Lower for stability
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.02,  # Higher for exploration
                 max_grad_norm: float = 0.5):
        """
        Initialize PPO trainer with improved hyperparameters
        
        Args:
            agent: LawnmowerRLAgent with pre-trained policy
            env: LawnmowerEnvironment
            learning_rate: Learning rate (lowered for stability)
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_epsilon: PPO clipping parameter
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient (increased for exploration)
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.agent = agent
        self.env = env
        self.device = agent.device
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Value network (critic) - Larger for better value estimation
        input_size = agent.encoder.get_feature_size()
        self.value_net = ValueNetwork(input_size, hidden_size=512).to(self.device)
        
        # Separate optimizers for better control
        self.policy_optimizer = optim.Adam(agent.policy.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate * 2)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        
        # Curriculum learning
        self.curriculum_stage = 0
        self.success_count = 0
        self.best_coverage = 0.0
    
    def get_curriculum_target(self) -> float:
        """Curriculum learning: gradually increase target coverage"""
        stages = [0.5, 0.7, 0.85, 0.95, 1.0]
        return stages[min(self.curriculum_stage, len(stages) - 1)]
    
    def update_curriculum(self, coverage: float):
        """Progress curriculum if agent is consistently hitting targets"""
        target = self.get_curriculum_target()
        if coverage >= target:
            self.success_count += 1
            if self.success_count >= 10 and self.curriculum_stage < 4:
                self.curriculum_stage += 1
                self.success_count = 0
                print(f"\n🎓 Curriculum advanced to stage {self.curriculum_stage + 1}!")
                print(f"   New target coverage: {self.get_curriculum_target() * 100:.0f}%\n")
        else:
            self.success_count = max(0, self.success_count - 1)
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: List of rewards
            values: List of state values
            dones: List of done flags
            next_value: Value of next state
        
        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        advantages = []
        gae = 0
        
        # Convert to numpy for easier manipulation
        rewards = rewards.cpu().numpy()
        values = values.cpu().numpy()
        dones = dones.cpu().numpy()
        
        # Compute advantages backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            # TD error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            
            # GAE: A_t = δ_t + γ*λ*δ_{t+1} + (γ*λ)^2*δ_{t+2} + ...
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def compute_reward(self, state_info: dict) -> float:
        """
        IMPROVED reward function with dense signals
        
        Reward components:
        1. Coverage progress (main signal) - MUCH LARGER
        2. Progressive completion bonuses
        3. Direction change penalties (smart)
        4. Progress toward uncovered areas
        5. Completion bonuses - MUCH LARGER
        6. Invalid move penalties
        """
        reward = 0.0
        
        # Extract state info
        new_coverage = state_info.get('new_cells_covered', 0)
        coverage_ratio = state_info.get('coverage_ratio', 0.0)
        direction_changed = state_info.get('direction_changed', False)
        step = state_info.get('step', 0)
        invalid_move = state_info.get('invalid_move', False)
        distance_to_uncovered = state_info.get('distance_to_uncovered', 0)
        
        # 1. COVERAGE REWARDS
        reward += new_coverage * 20.0 
        
        # Progressive bonus as coverage increases
        # if coverage_ratio > 0.5:
        #     reward += (coverage_ratio - 0.5) * 10.0
        # if coverage_ratio > 0.75:
        #     reward += (coverage_ratio - 0.75) * 20.0
        # if coverage_ratio > 0.90:
        #     reward += (coverage_ratio - 0.90) * 50.0
        # if coverage_ratio > 0.95:
        #     reward += (coverage_ratio - 0.95) * 100.0
        
        # Completion bonuses
        if coverage_ratio >= 1.0:
            reward += 1000.0 
            reward += max(0, 1500 - step) * 0.5  # Speed bonus
        
        # 2. EFFICIENCY REWARDS
        reward -= 1  # Step penalty
        
        # MUCH STRONGER direction change penalties to prevent zigzagging
        if direction_changed:
            if new_coverage == 0:
                reward -= 10.0  # Wasteful turn
            else:
                reward -= 2.0   # Productive turn
        else:
            if new_coverage != 0:
                reward += 2.0   # BONUS for continuing straight

        # 3. PROGRESS SIGNALS (help guide exploration)
        if distance_to_uncovered > 0:
            proximity_bonus = 0.5 / (distance_to_uncovered + 1)
            reward += proximity_bonus
        
        if distance_to_uncovered > 20:
            reward -= 0.1
        
        # 4. PENALTIES
        if invalid_move:
            reward -= 5.0
        
        if new_coverage == 0 and not invalid_move:
            reward -= 0.5  # Penalty for non-productive moves
        
        # 5. TIME-BASED PENALTIES
        if step > 500 and coverage_ratio < 0.5:
            reward -= 0.1
        if step > 1000 and coverage_ratio < 0.75:
            reward -= 0.3
        if step > 1500:
            reward -= 1
        
        return reward
    
    def collect_trajectory(self, max_steps: int = 2000) -> Tuple[PPOTrajectory, dict]:
        """
        Collect a single trajectory by running the policy
        
        Returns:
            trajectory: PPOTrajectory object
            info: Episode information dict
        """
        trajectory = PPOTrajectory()
        
        # Reset environment
        current_pos = self.env.start_pos
        mowed_cells = set()
        last_action = None
        
        # Episode statistics
        episode_reward = 0
        direction_changes = 0
        
        for step in range(max_steps):
            # Get current coverage BEFORE this step
            prev_mowed_count = len(mowed_cells & self.env.reachable_lawn_cells)
            
            # Encode state BEFORE moving
            state = self.agent.encoder.encode_state(current_pos, mowed_cells, last_action)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                action_probs = self.agent.policy.get_action_probs(state_tensor)
                action = torch.multinomial(action_probs, 1).item()
                log_prob = torch.log(action_probs[0, action])
                value = self.value_net(state_tensor).squeeze()
            
            # Execute action - MOVE FIRST
            from train_rl_agent import ActionMapper
            dx, dy = ActionMapper.action_to_delta(action)
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # Check valid move
            invalid_move = False
            if next_pos not in self.env.lawn_cells:
                invalid_move = True
                # Try random neighbor as recovery
                neighbors = self.env.get_neighbors(current_pos)
                if neighbors:
                    next_pos = random.choice(list(neighbors))
                else:
                    # Can't move, end episode
                    reward = -5.0
                    done = True
                    trajectory.add(state, action, reward, log_prob.item(), value.item(), done)
                    break
            
            # NOW update position
            current_pos = next_pos
            
            # Mow at NEW position
            blade = self.env.get_blade_coverage(current_pos)
            for cell in blade:
                if cell in self.env.reachable_lawn_cells:
                    mowed_cells.add(cell)
            
            # Calculate how many NEW cells were mowed (after the move)
            new_mowed_count = len(mowed_cells & self.env.reachable_lawn_cells)
            new_cells_covered = new_mowed_count - prev_mowed_count
            
            # Check if done
            coverage_ratio = new_mowed_count / len(self.env.reachable_lawn_cells)
            done = (coverage_ratio >= 1.0)
            
            # Direction change?
            direction_changed = (last_action is not None and action != last_action)
            if direction_changed:
                direction_changes += 1
            
            # Distance to nearest uncovered cell
            uncovered = self.env.reachable_lawn_cells - mowed_cells
            if uncovered:
                distance_to_uncovered = min(
                    abs(p[0] - current_pos[0]) + abs(p[1] - current_pos[1])
                    for p in uncovered
                )
            else:
                distance_to_uncovered = 0
            
            # Compute reward with ALL necessary information
            state_info = {
                'new_cells_covered': new_cells_covered,  # THIS WAS MISSING!
                'coverage_ratio': coverage_ratio,
                'direction_changed': direction_changed,
                'step': step,
                'invalid_move': invalid_move,
                'distance_to_uncovered': distance_to_uncovered,
            }
            reward = self.compute_reward(state_info)
            
            # Store transition
            trajectory.add(state, action, reward, log_prob.item(), value.item(), done)
            
            episode_reward += reward
            last_action = action
            
            if done:
                break
        
        # Episode info
        final_coverage = len(mowed_cells & self.env.reachable_lawn_cells)
        coverage_pct = 100 * final_coverage / len(self.env.reachable_lawn_cells)
        
        info = {
            'episode_reward': episode_reward,
            'episode_length': step + 1,
            'coverage': coverage_pct,
            'direction_changes': direction_changes
        }
        
        return trajectory, info
    
    def update_policy(self, trajectories: List[PPOTrajectory], epochs: int = 4):
        """
        Update policy using PPO
        
        Args:
            trajectories: List of collected trajectories
            epochs: Number of epochs to train on collected data
        """
        # Combine all trajectories
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []
        all_returns = []
        
        for traj in trajectories:
            states, actions, rewards, log_probs, values, dones = traj.get()
            
            # Compute advantages and returns
            with torch.no_grad():
                # Get value of last state for bootstrapping
                if len(states) > 0:
                    last_state = states[-1].unsqueeze(0).to(self.device)
                    next_value = self.value_net(last_state).item()
                else:
                    next_value = 0.0
            
            advantages, returns = self.compute_gae(rewards, values, dones, next_value)
            
            all_states.append(states)
            all_actions.append(actions)
            all_old_log_probs.append(log_probs)
            all_advantages.append(advantages)
            all_returns.append(returns)
        
        # Concatenate all data
        states = torch.cat(all_states).to(self.device)
        actions = torch.cat(all_actions).to(self.device)
        old_log_probs = torch.cat(all_old_log_probs).to(self.device)
        advantages = torch.cat(all_advantages).to(self.device)
        returns = torch.cat(all_returns).to(self.device)
        
        # PPO update for multiple epochs
        for epoch in range(epochs):
            # Forward pass
            action_probs = self.agent.policy.get_action_probs(states)
            values = self.value_net(states).squeeze()
            
            # Get log probabilities for taken actions
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())
            
            # Compute entropy
            entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=1).mean()
            
            # Compute ratio for PPO
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values, returns)
            
            # Update policy
            self.policy_optimizer.zero_grad()
            policy_loss_total = policy_loss - self.entropy_coef * entropy
            policy_loss_total.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.agent.policy.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            
            # Update value (separate)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
            self.value_optimizer.step()
            
            # Record statistics
            if epoch == epochs - 1:  # Only record on last epoch
                self.policy_losses.append(policy_loss.item())
                self.value_losses.append(value_loss.item())
                self.entropies.append(entropy.item())
    
    def train(self, num_iterations: int = 1000, 
             trajectories_per_iteration: int = 4,
             epochs_per_update: int = 4,
             eval_frequency: int = 50):
        """
        Main PPO training loop
        
        Args:
            num_iterations: Number of training iterations
            trajectories_per_iteration: Trajectories to collect per iteration
            epochs_per_update: PPO epochs per update
            eval_frequency: Evaluate every N iterations
        """
        print("\n" + "="*80)
        print("STARTING PPO FINE-TUNING")
        print("="*80)
        print(f"Iterations: {num_iterations}")
        print(f"Trajectories per iteration: {trajectories_per_iteration}")
        print(f"PPO epochs per update: {epochs_per_update}")
        print("="*80 + "\n")
        
        best_reward = -float('inf')
        
        for iteration in range(num_iterations):
            # Collect trajectories
            trajectories = []
            iteration_rewards = []
            iteration_lengths = []
            iteration_coverages = []
            
            for _ in range(trajectories_per_iteration):
                traj, info = self.collect_trajectory()
                trajectories.append(traj)
                iteration_rewards.append(info['episode_reward'])
                iteration_lengths.append(info['episode_length'])
                iteration_coverages.append(info['coverage'])
            
            # Update policy
            self.update_policy(trajectories, epochs=epochs_per_update)
            
            # Record statistics
            avg_reward = np.mean(iteration_rewards)
            avg_length = np.mean(iteration_lengths)
            avg_coverage = np.mean(iteration_coverages)
            
            self.episode_rewards.extend(iteration_rewards)
            self.episode_lengths.extend(iteration_lengths)
            
            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                self.agent.save_model('models/best_ppo_model.pt')
            
            # Print progress
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"Iteration {iteration + 1}/{num_iterations}")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Avg Length: {avg_length:.0f}")
                print(f"  Avg Coverage: {avg_coverage:.1f}%")
                print(f"  Policy Loss: {self.policy_losses[-1]:.4f}")
                print(f"  Value Loss: {self.value_losses[-1]:.4f}")
                print(f"  Entropy: {self.entropies[-1]:.4f}")
                print()
            
            # Evaluation
            if (iteration + 1) % eval_frequency == 0:
                print(f"\n{'='*80}")
                print(f"EVALUATION at iteration {iteration + 1}")
                print(f"{'='*80}")
                eval_results = self.evaluate(num_episodes=5)
                print(f"Evaluation Results:")
                print(f"  Coverage: {eval_results['avg_coverage']:.1f}%")
                print(f"  Steps: {eval_results['avg_steps']:.0f}")
                print(f"  Efficiency: {eval_results['avg_efficiency']:.1f}%")
                print(f"  Direction Changes: {eval_results['avg_direction_changes']:.0f}")
                print(f"{'='*80}\n")
        
        print("\n" + "="*80)
        print("PPO FINE-TUNING COMPLETE!")
        print(f"Best average reward: {best_reward:.2f}")
        print("="*80 + "\n")
        
        # Save final model
        self.agent.save_model('models/final_ppo_model.pt')
    
    def evaluate(self, num_episodes: int = 5) -> dict:
        """
        Evaluate current policy
        """
        self.agent.policy.eval()
        self.value_net.eval()
        
        results = {
            'coverages': [],
            'steps': [],
            'efficiencies': [],
            'direction_changes': []
        }
        
        for episode in range(num_episodes):
            _, info = self.collect_trajectory()
            
            coverage = info['coverage']
            steps = info['episode_length']
            mowed = len(self.env.reachable_lawn_cells) * coverage / 100
            efficiency = 100 * mowed / steps if steps > 0 else 0
            
            results['coverages'].append(coverage)
            results['steps'].append(steps)
            results['efficiencies'].append(efficiency)
            results['direction_changes'].append(info['direction_changes'])
        
        self.agent.policy.train()
        self.value_net.train()
        
        return {
            'avg_coverage': np.mean(results['coverages']),
            'avg_steps': np.mean(results['steps']),
            'avg_efficiency': np.mean(results['efficiencies']),
            'avg_direction_changes': np.mean(results['direction_changes'])
        }
