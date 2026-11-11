"""
Reinforcement Learning Agent for Lawnmower Path Planning
Using Imitation Learning with expert demonstrations from hybrid, ccpp, and sweepopt algorithms
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from collections import deque
from typing import List, Tuple, Dict
import pickle
import os

from utils.environment import LawnmowerEnvironment


class LawnmowerStateEncoder:
    """Enhanced state encoder with better spatial representation - 308 features"""
    
    def __init__(self, env: LawnmowerEnvironment):
        self.env = env
        self.grid_height = env.height
        self.grid_width = env.width
        self._compute_distance_fields()
        
    def _compute_distance_fields(self):
        """Precompute distance to boundaries and obstacles"""
        self.dist_to_boundary = np.zeros((self.grid_height, self.grid_width))
        self.dist_to_obstacle = np.zeros((self.grid_height, self.grid_width))
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                dist_boundary = min(x, y, self.grid_width - 1 - x, self.grid_height - 1 - y)
                self.dist_to_boundary[y, x] = dist_boundary
                
                if (x, y) not in self.env.all_lawn_cells:
                    self.dist_to_obstacle[y, x] = 0
                else:
                    min_dist = float('inf')
                    for dy in range(-10, 11):
                        for dx in range(-10, 11):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                                if (nx, ny) not in self.env.all_lawn_cells:
                                    dist = abs(dx) + abs(dy)
                                    min_dist = min(min_dist, dist)
                    self.dist_to_obstacle[y, x] = min(min_dist, 10)
        
    def encode_state(self, position: Tuple[int, int], 
                    mowed_cells: set, 
                    last_action: int = None,
                    path_history: deque = None) -> np.ndarray:
        """Enhanced state encoding - 308 features"""
        features = []
        
        # 1. Position features (6)
        norm_x = position[0] / self.grid_width
        norm_y = position[1] / self.grid_height
        dist_boundary = self.dist_to_boundary[position[1], position[0]] / max(self.grid_width, self.grid_height)
        dist_obstacle = self.dist_to_obstacle[position[1], position[0]] / 10.0
        quadrant_x = 1.0 if position[0] > self.grid_width / 2 else 0.0
        quadrant_y = 1.0 if position[1] > self.grid_height / 2 else 0.0
        features.extend([norm_x, norm_y, dist_boundary, dist_obstacle, quadrant_x, quadrant_y])
        
        # 2. Large local occupancy grid 11x11 (121 features)
        local_size = 11
        offset = local_size // 2
        local_occupancy = []
        for dy in range(-offset, offset + 1):
            for dx in range(-offset, offset + 1):
                check_pos = (position[0] + dx, position[1] + dy)
                local_occupancy.append(1.0 if check_pos in self.env.lawn_cells else 0.0)
        features.extend(local_occupancy)
        
        # 3. Large local coverage grid 11x11 (121 features)
        local_coverage = []
        for dy in range(-offset, offset + 1):
            for dx in range(-offset, offset + 1):
                check_pos = (position[0] + dx, position[1] + dy)
                local_coverage.append(1.0 if check_pos in mowed_cells else 0.0)
        features.extend(local_coverage)
        
        # 4. Multi-scale coverage (20 features)
        for radius in [1, 2, 3, 5, 10]:
            total_cells = 0
            mowed_count = 0
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dx) + abs(dy) <= radius:
                        check_pos = (position[0] + dx, position[1] + dy)
                        if check_pos in self.env.reachable_lawn_cells:
                            total_cells += 1
                            if check_pos in mowed_cells:
                                mowed_count += 1
            coverage_ratio = mowed_count / total_cells if total_cells > 0 else 0
            features.extend([coverage_ratio, mowed_count / 100.0, (total_cells - mowed_count) / 100.0])
        while len(features) < 6 + 121 + 121 + 20:
            features.append(0.0)
        
        # 5. Coverage statistics (8 features)
        total_mowable = len(self.env.reachable_lawn_cells)
        mowed_count = len(mowed_cells & self.env.reachable_lawn_cells)
        coverage_ratio = mowed_count / total_mowable if total_mowable > 0 else 0
        features.extend([
            coverage_ratio,
            mowed_count / 2000,
            (total_mowable - mowed_count) / 2000,
            coverage_ratio ** 2,
            1.0 if coverage_ratio > 0.9 else 0.0,
            1.0 if coverage_ratio > 0.95 else 0.0,
            1.0 if coverage_ratio > 0.99 else 0.0,
            1.0 if coverage_ratio >= 1.0 else 0.0,
        ])
        
        # 6. Last action one-hot (8 features)
        action_onehot = [0.0] * 8
        if last_action is not None and 0 <= last_action < 8:
            action_onehot[last_action] = 1.0
        features.extend(action_onehot)
        
        # 7. Directional coverage info (8 features)
        directions = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
        for dx, dy in directions:
            uncovered_count = 0
            for dist in range(1, 11):
                check_pos = (position[0] + dx * dist, position[1] + dy * dist)
                if check_pos in self.env.reachable_lawn_cells and check_pos not in mowed_cells:
                    uncovered_count += 1
            features.append(min(uncovered_count / 10.0, 1.0))
        
        # 8. Nearest unmowed features (6 features)
        unmowed = self.env.reachable_lawn_cells - mowed_cells
        if unmowed:
            nearest = min(unmowed, key=lambda p: abs(p[0]-position[0]) + abs(p[1]-position[1]))
            dist = abs(nearest[0] - position[0]) + abs(nearest[1] - position[1])
            dir_x = np.clip((nearest[0] - position[0]) / 20, -1, 1)
            dir_y = np.clip((nearest[1] - position[1]) / 20, -1, 1)
            norm_dist = min(dist / 50.0, 1.0)
            cluster_x = sum(p[0] for p in list(unmowed)[:100]) / min(len(unmowed), 100)
            cluster_y = sum(p[1] for p in list(unmowed)[:100]) / min(len(unmowed), 100)
            cluster_dir_x = np.clip((cluster_x - position[0]) / 20, -1, 1)
            cluster_dir_y = np.clip((cluster_y - position[1]) / 20, -1, 1)
            features.extend([dir_x, dir_y, norm_dist, cluster_dir_x, cluster_dir_y, len(unmowed) / 2000])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # 9. Path history features (10 features)
        if path_history and len(path_history) > 0:
            recent_moves = list(path_history)[-10:]
            changes = sum(1 for i in range(1, len(recent_moves)) if recent_moves[i] != recent_moves[i-1])
            direction_change_rate = changes / max(len(recent_moves) - 1, 1)
            if recent_moves:
                from collections import Counter
                most_common_dir = Counter(recent_moves).most_common(1)[0][0]
                most_common_onehot = [0.0] * 8
                if 0 <= most_common_dir < 8:
                    most_common_onehot[most_common_dir] = 1.0
            else:
                most_common_onehot = [0.0] * 8
            features.append(direction_change_rate)
            features.extend(most_common_onehot)
            features.append(len(path_history) / 2000)
        else:
            features.extend([0.0] * 10)
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_size(self) -> int:
        """Return total number of features: 308"""
        return 308


class ActionMapper:
    """Map between discrete actions and grid movements"""
    
    # 8 directions: N, NE, E, SE, S, SW, W, NW
    ACTIONS = [
        (0, -1),   # 0: North
        (1, -1),   # 1: NE
        (1, 0),    # 2: East
        (1, 1),    # 3: SE
        (0, 1),    # 4: South
        (-1, 1),   # 5: SW
        (-1, 0),   # 6: West
        (-1, -1),  # 7: NW
    ]
    
    @classmethod
    def action_to_delta(cls, action: int) -> Tuple[int, int]:
        """Convert action index to (dx, dy)"""
        return cls.ACTIONS[action]
    
    @classmethod
    def delta_to_action(cls, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
        """Convert position change to action index"""
        dx = np.clip(to_pos[0] - from_pos[0], -1, 1)
        dy = np.clip(to_pos[1] - from_pos[1], -1, 1)
        
        try:
            return cls.ACTIONS.index((dx, dy))
        except ValueError:
            # If exact delta not in actions, find closest
            distances = [abs(dx - a[0]) + abs(dy - a[1]) for a in cls.ACTIONS]
            return np.argmin(distances)
    
    @classmethod
    def is_direction_change(cls, action1: int, action2: int) -> bool:
        """Check if two actions represent a direction change"""
        if action1 is None or action2 is None:
            return False
        return action1 != action2


class PolicyNetwork(nn.Module):
    """Neural network that predicts action probabilities - optimized for 308 features"""
    
    def __init__(self, input_size: int, hidden_size: int = 512):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 8)  # 8 actions
        )
    
    def forward(self, x):
        return self.network(x)
    
    def get_action_probs(self, x):
        """Get action probabilities"""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)
    
    def get_action(self, x, deterministic=False):
        """Sample action from policy"""
        probs = self.get_action_probs(x)
        
        if deterministic:
            return torch.argmax(probs, dim=-1)
        else:
            return torch.multinomial(probs, 1).squeeze(-1)


class ExpertDemonstrationDataset(Dataset):
    """Dataset of expert demonstrations for imitation learning"""
    
    def __init__(self, demonstrations: List[Tuple[np.ndarray, int]]):
        self.states = [d[0] for d in demonstrations]
        self.actions = [d[1] for d in demonstrations]
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.states[idx]), torch.LongTensor([self.actions[idx]])


class LawnmowerRLAgent:
    """Main RL Agent using Imitation Learning"""
    
    def __init__(self, env: LawnmowerEnvironment, 
                 learning_rate: float = 1e-3,
                 device: str = 'cpu'):
        self.env = env
        self.device = torch.device(device)
        
        # State encoder
        self.encoder = LawnmowerStateEncoder(env)
        input_size = self.encoder.get_feature_size()
        
        # Policy network
        self.policy = PolicyNetwork(input_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Action mapper
        self.action_mapper = ActionMapper()
        
        # Training statistics
        self.train_losses = []
        self.val_accuracies = []
    
    def collect_demonstrations(self, algorithms: List[str], 
                               num_demos_per_algo: int = 5,
                               start_positions: List[Tuple[int, int]] = None) -> List[Tuple[np.ndarray, int]]:
        """
        Collect expert demonstrations from multiple algorithms
        
        Args:
            algorithms: List of algorithm names ['hybrid', 'ccpp', 'sweepopt', 'sweepopt_no_adaptive']
            num_demos_per_algo: Number of demonstrations per algorithm
            start_positions: List of starting positions (None = use default)
        """
        print(f"Collecting demonstrations from {len(algorithms)} algorithms...")
        
        demonstrations = []
        
        for algo_name in algorithms:
            print(f"  Collecting from {algo_name}...")
            
            for demo_idx in range(num_demos_per_algo):
                # Get algorithm module
                if algo_name == 'hybrid':
                    from algorithms.hybrid import hybrid_coverage
                    path_generator = lambda: hybrid_coverage(self.env)
                elif algo_name == 'ccpp':
                    from algorithms.ccpp import CompleteCoveragePlanner
                    planner = CompleteCoveragePlanner(self.env, show_visualization=False)
                    path_generator = lambda: planner.plan_full_coverage()
                elif algo_name == 'sweepopt':
                    from algorithms.sweep_optimized import rectangle_decomposition_sweep
                    path_generator = lambda: rectangle_decomposition_sweep(self.env)
                elif algo_name == 'sweepopt_no_adaptive':
                    # You can add non-adaptive version if needed
                    from algorithms.sweep_optimized import rectangle_decomposition_sweep
                    path_generator = lambda: rectangle_decomposition_sweep(self.env, enable_adaptive=False)
                else:
                    print(f"    Unknown algorithm: {algo_name}, skipping")
                    continue
                
                try:
                    # Generate path
                    path = list(path_generator())
                    
                    # Convert path to state-action pairs
                    mowed_cells = set()
                    last_action = None
                    
                    for i in range(len(path) - 1):
                        current_pos = path[i]
                        next_pos = path[i + 1]
                        
                        # Update mowed cells
                        blade_coverage = self.env.get_blade_coverage(current_pos)
                        for cell in blade_coverage:
                            if cell in self.env.reachable_lawn_cells:
                                mowed_cells.add(cell)
                        
                        # Encode state
                        state = self.encoder.encode_state(current_pos, mowed_cells, last_action)
                        
                        # Get action
                        action = self.action_mapper.delta_to_action(current_pos, next_pos)
                        
                        demonstrations.append((state, action))
                        last_action = action
                    
                    print(f"    Demo {demo_idx + 1}: {len(path)} steps")
                
                except Exception as e:
                    print(f"    Error collecting demo {demo_idx + 1}: {e}")
        
        print(f"Total demonstrations collected: {len(demonstrations)}")
        return demonstrations
    
    def pretrain_from_demonstrations(self, demonstrations: List[Tuple[np.ndarray, int]],
                                    batch_size: int = 64,
                                    num_epochs: int = 50,
                                    validation_split: float = 0.2):
        """
        Pre-train policy using supervised learning on expert demonstrations
        """
        print(f"\nPre-training on {len(demonstrations)} demonstrations...")
        
        # Shuffle demonstrations
        random.shuffle(demonstrations)
        
        # Split into train/validation
        split_idx = int(len(demonstrations) * (1 - validation_split))
        train_demos = demonstrations[:split_idx]
        val_demos = demonstrations[split_idx:]
        
        # Create datasets
        train_dataset = ExpertDemonstrationDataset(train_demos)
        val_dataset = ExpertDemonstrationDataset(val_demos)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            # Training
            self.policy.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for states, actions in train_loader:
                states = states.to(self.device)
                actions = actions.squeeze(1).to(self.device)
                
                # Forward pass
                logits = self.policy(states)
                loss = criterion(logits, actions)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_total += actions.size(0)
                train_correct += (predicted == actions).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # Validation
            self.policy.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for states, actions in val_loader:
                    states = states.to(self.device)
                    actions = actions.squeeze(1).to(self.device)
                    
                    logits = self.policy(states)
                    _, predicted = torch.max(logits, 1)
                    val_total += actions.size(0)
                    val_correct += (predicted == actions).sum().item()
            
            val_acc = 100 * val_correct / val_total
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('models/best_pretrained_model.pt')
            
            # Record statistics
            self.train_losses.append(avg_train_loss)
            self.val_accuracies.append(val_acc)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Train Acc: {train_acc:.2f}%, "
                      f"Val Acc: {val_acc:.2f}%")
        
        print(f"\nPre-training complete! Best validation accuracy: {best_val_acc:.2f}%")
    
    def evaluate(self, num_episodes: int = 10, max_steps: int = 2000) -> Dict:
        """
        Evaluate the trained policy
        
        Returns metrics: coverage, steps, efficiency, direction_changes
        """
        print(f"\nEvaluating policy on {num_episodes} episodes...")
        
        self.policy.eval()
        
        results = {
            'coverages': [],
            'steps': [],
            'efficiencies': [],
            'direction_changes': [],
            'paths': []
        }
        
        for episode in range(num_episodes):
            # Reset environment
            current_pos = self.env.start_pos
            mowed_cells = set()
            path = [current_pos]
            last_action = None
            direction_changes = 0
            
            for step in range(max_steps):
                # Update mowed cells
                blade_coverage = self.env.get_blade_coverage(current_pos)
                for cell in blade_coverage:
                    if cell in self.env.reachable_lawn_cells:
                        mowed_cells.add(cell)
                
                # Check if done
                coverage = len(mowed_cells & self.env.reachable_lawn_cells)
                if coverage >= len(self.env.reachable_lawn_cells):
                    break
                
                # Get action from policy
                state = self.encoder.encode_state(current_pos, mowed_cells, last_action)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action = self.policy.get_action(state_tensor, deterministic=True).item()
                
                # Take action
                dx, dy = self.action_mapper.action_to_delta(action)
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                
                # Check if valid move
                if next_pos not in self.env.lawn_cells:
                    # Invalid move - try random valid neighbor
                    neighbors = self.env.get_neighbors(current_pos)
                    if neighbors:
                        next_pos = random.choice(list(neighbors))
                    else:
                        break
                
                # Count direction change
                if last_action is not None and action != last_action:
                    direction_changes += 1
                
                path.append(next_pos)
                current_pos = next_pos
                last_action = action
            
            # Calculate metrics
            final_coverage = len(mowed_cells & self.env.reachable_lawn_cells)
            coverage_pct = 100 * final_coverage / len(self.env.reachable_lawn_cells)
            efficiency = 100 * final_coverage / len(path) if len(path) > 0 else 0
            
            results['coverages'].append(coverage_pct)
            results['steps'].append(len(path))
            results['efficiencies'].append(efficiency)
            results['direction_changes'].append(direction_changes)
            results['paths'].append(path)
            
            print(f"  Episode {episode+1}: Coverage={coverage_pct:.1f}%, "
                  f"Steps={len(path)}, Efficiency={efficiency:.1f}%, "
                  f"Direction changes={direction_changes}")
        
        # Average results
        avg_results = {
            'avg_coverage': np.mean(results['coverages']),
            'avg_steps': np.mean(results['steps']),
            'avg_efficiency': np.mean(results['efficiencies']),
            'avg_direction_changes': np.mean(results['direction_changes']),
        }
        
        print(f"\nAverage Results:")
        print(f"  Coverage: {avg_results['avg_coverage']:.1f}%")
        print(f"  Steps: {avg_results['avg_steps']:.0f}")
        print(f"  Efficiency: {avg_results['avg_efficiency']:.1f}%")
        print(f"  Direction Changes: {avg_results['avg_direction_changes']:.0f}")
        
        return avg_results, results
    
    def save_model(self, filepath: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        print(f"Model loaded from {filepath}")


def main():
    """Main training script"""
    import matplotlib.pyplot as plt
    
    # Load environment
    print("Loading environment...")
    env = LawnmowerEnvironment('maps/maze6.txt')
    
    # Create agent
    print("Creating RL agent...")
    agent = LawnmowerRLAgent(env, learning_rate=1e-3)
    
    # Collect demonstrations from best algorithms
    algorithms = ['hybrid', 'ccpp', 'sweepopt']
    demonstrations = agent.collect_demonstrations(
        algorithms=algorithms,
        num_demos_per_algo=3  # Collect 3 demonstrations per algorithm
    )
    
    # Save demonstrations for reuse
    os.makedirs('data', exist_ok=True)
    with open('data/expert_demonstrations.pkl', 'wb') as f:
        pickle.dump(demonstrations, f)
    print(f"Demonstrations saved to data/expert_demonstrations.pkl")
    
    # Pre-train on demonstrations
    agent.pretrain_from_demonstrations(
        demonstrations=demonstrations,
        batch_size=64,
        num_epochs=50,
        validation_split=0.2
    )
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(agent.train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(agent.val_accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy Over Time')
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/training_curves.png')
    print("Training curves saved to plots/training_curves.png")
    
    # Evaluate trained agent
    avg_results, detailed_results = agent.evaluate(num_episodes=10)
    
    # Save final model
    agent.save_model('models/final_model.pt')
    
    # Save evaluation results
    with open('data/evaluation_results.pkl', 'wb') as f:
        pickle.dump({'avg': avg_results, 'detailed': detailed_results}, f)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final Model Performance:")
    print(f"  Average Coverage: {avg_results['avg_coverage']:.1f}%")
    print(f"  Average Steps: {avg_results['avg_steps']:.0f}")
    print(f"  Average Efficiency: {avg_results['avg_efficiency']:.1f}%")
    print(f"  Average Direction Changes: {avg_results['avg_direction_changes']:.0f}")
    print("="*60)


if __name__ == "__main__":
    main()
