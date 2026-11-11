#!/usr/bin/env python3
"""
DQN training with improved metrics and warmup period
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from typing import List, Dict, Any
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from utils.environment import LawnmowerEnvironment
from algorithms.dqn_agent import DQNAgent, calculate_reward
from visualization.plotter import LawnmowerVisualizer


def train_dqn(env: LawnmowerEnvironment,
              episodes: int = 2000,
              max_steps: int = 300,
              model_path: str = 'models/dqn.pkl',
              early_stopping_patience: int = 200,
              learning_rate: float = 0.00025,
              epsilon_decay: float = 0.997) -> Dict[str, Any]:
    """
    Train DQN agent with warmup period and stability improvements
    """
    # Initialize agent WITHOUT specifying state_size
    agent = DQNAgent(
        learning_rate=learning_rate, 
        epsilon_decay=epsilon_decay
        # state_size will be auto-detected
    )
    
    if Path(model_path).exists():
        agent.load(model_path)
        print("Loaded existing model, continuing training...\n")
    
    stats = {
        'episodes': [],
        'coverages': [],
        'efficiencies': [],
        'rewards': [],
        'steps': [],
        'losses': [],
        'epsilons': []
    }
    
    best_path = None
    best_reward = -float('inf')
    
    print(f"\nStarting training:")
    print(f"  Episodes: {episodes}")
    print(f"  Max steps: {max_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epsilon decay: {epsilon_decay}")
    print(f"  Total reachable lawn cells: {len(env.reachable_lawn_cells)}")
    print(f"  Valid mower positions: {len(env.lawn_cells)}")
    
    for episode in range(episodes):
        state_obj = env.create_state()
        state = agent.get_state(env, state_obj)
        total_reward = 0
        episode_loss = 0
        train_count = 0
        
        path = [state_obj.position]
        
        for step in range(max_steps):
            # Get valid actions
            valid_actions = []
            for action_idx, (dx, dy) in enumerate(env.DIRECTIONS.values()):
                next_pos = (state_obj.position[0] + dx, 
                           state_obj.position[1] + dy)
                if env.is_valid_move(state_obj.position, next_pos):
                    valid_actions.append(action_idx)
            
            if not valid_actions:
                break
            
            # Choose action
            action = agent.act(state, valid_actions, training=True)
            
            # Execute action
            dx, dy = list(env.DIRECTIONS.values())[action]
            next_pos = (state_obj.position[0] + dx, 
                       state_obj.position[1] + dy)
            
            old_state_obj = state_obj
            state_obj = env.move(state_obj, next_pos)
            next_state = agent.get_state(env, state_obj)
            path.append(state_obj.position)
            
            # Calculate reward
            reward = calculate_reward(env, old_state_obj, state_obj, action, step, max_steps)
            total_reward += reward
            
            # Store experience
            done = env.is_complete(state_obj) or step >= max_steps - 1
            agent.remember(state.squeeze(0).cpu(), action, reward, 
                          next_state.squeeze(0).cpu(), done)
            
            # Increment total steps for warmup
            agent.increment_steps()
            
            # Train (will be skipped during warmup)
            if step % 4 == 0:  # Train less frequently
                loss = agent.replay()
                if loss > 0:
                    episode_loss += loss
                    train_count += 1
            
            # Update state
            state = next_state
            
            if done:
                if env.is_complete(state_obj):
                    print(f"  Episode {episode + 1}: COMPLETED in {step + 1} steps!")
                break
        
        # Update epsilon
        agent.update_epsilon()
        
        # Calculate metrics
        total_reachable_cells = len(env.reachable_lawn_cells)
        mowed_reachable = len(state_obj.mowed_cells.intersection(env.reachable_lawn_cells))
        coverage = 100 * mowed_reachable / total_reachable_cells if total_reachable_cells > 0 else 0
        efficiency = env.calculate_efficiency(state_obj)
        avg_loss = episode_loss / max(train_count, 1)
        steps = state_obj.total_steps
        
        # Update stats
        is_best = agent.update_stats(coverage, efficiency, total_reward, steps, avg_loss)
        
        # Save if best
        if is_best:
            best_path = path
            best_reward = total_reward
            agent.save(model_path)
            print(f"  🎯 NEW BEST! Episode {episode + 1}: "
                  f"Reward: {total_reward:.1f}, "
                  f"Coverage: {coverage:.1f}%, "
                  f"Efficiency: {efficiency:.1f}%, "
                  f"Steps: {steps}")
        
        # Store statistics
        stats['episodes'].append(episode + 1)
        stats['coverages'].append(coverage)
        stats['efficiencies'].append(efficiency)
        stats['rewards'].append(total_reward)
        stats['steps'].append(steps)
        stats['losses'].append(avg_loss)
        stats['epsilons'].append(agent.epsilon)
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            warmup_info = ""
            if agent.total_steps < agent.warmup_steps:
                warmup_info = f" (Warmup: {agent.total_steps}/{agent.warmup_steps})"
            
            print(f"Ep {episode + 1:4d}/{episodes} | "
                  f"Steps: {steps:3d} | "
                  f"Cov: {coverage:5.1f}% | "
                  f"Eff: {efficiency:5.1f}% | "
                  f"Reward: {total_reward:7.1f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}{warmup_info}")
        
        # Early stopping check
        if agent.should_stop_early(early_stopping_patience):
            print(f"\n⏹️ Early stopping: No improvement for {early_stopping_patience} episodes")
            break
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Episodes trained: {len(stats['episodes'])}")
    print(f"Best reward: {agent.best_reward:.1f}")
    print(f"Best coverage: {agent.best_coverage:.1f}%")
    print(f"Best efficiency: {agent.best_efficiency:.1f}%")
    print(f"Best steps: {agent.best_steps}")
    print(f"Final epsilon: {agent.epsilon:.3f}")
    print(f"Replay buffer size: {len(agent.memory)}")
    print(f"Total training steps: {agent.total_steps}")
    print(f"{'='*60}")
    
    # Plot results
    if len(stats['episodes']) > 0:
        plot_training_stats(stats, model_path.replace('.pkl', '_stats.png'))
    
    # Visualize best path
    if best_path is not None:
        print(f"\nVisualizing best path (reward: {best_reward:.1f})...")
        viz = LawnmowerVisualizer(env)
        viz.plot_path(best_path, 
                     title=f"Best Training Run\n"
                           f"Reward: {best_reward:.1f}, "
                           f"Coverage: {agent.best_coverage:.1f}%, "
                           f"Efficiency: {agent.best_efficiency:.1f}%, "
                           f"Steps: {agent.best_steps}")
    
    return {
        'agent': agent,
        'stats': stats,
        'best_path': best_path
    }


def plot_training_stats(stats: Dict[str, Any], save_path: str = None):
    """Plot training statistics"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Coverage
    axes[0, 0].plot(stats['episodes'], stats['coverages'])
    axes[0, 0].axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100%')
    axes[0, 0].set_title('Coverage of Reachable Cells')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Coverage (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Efficiency
    axes[0, 1].plot(stats['episodes'], stats['efficiencies'])
    axes[0, 1].set_title('Efficiency (Cells/Step)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Efficiency (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Reward
    axes[0, 2].plot(stats['episodes'], stats['rewards'])
    axes[0, 2].set_title('Total Reward')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Reward')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Steps
    axes[1, 0].plot(stats['episodes'], stats['steps'])
    axes[1, 0].set_title('Steps per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[1, 1].plot(stats['episodes'], stats['losses'])
    axes[1, 1].set_title('Training Loss')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Epsilon
    axes[1, 2].plot(stats['episodes'], stats['epsilons'])
    axes[1, 2].set_title('Exploration Rate (ε)')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Epsilon')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training statistics saved to {save_path}")
    
    plt.show()


def test_dqn(env: LawnmowerEnvironment,
             model_path: str = 'models/dqn.pkl',
             visualize: bool = True,
             animate: bool = False,
             max_steps: int = 500,
             num_tests: int = 5,
             save_animation: str = None) -> List[List[tuple]]:
    """
    Test DQN agent
    """
    agent = DQNAgent()
    
    if not agent.load(model_path):
        print(f"Could not load model from {model_path}")
        return []
    
    # Set epsilon to minimum for testing
    agent.epsilon = agent.epsilon_min
    
    all_paths = []
    all_metrics = []
    
    print(f"\nTesting DQN agent for {num_tests} runs...")
    print(f"Using epsilon: {agent.epsilon:.3f}")
    print(f"Testing on {len(env.reachable_lawn_cells)} reachable lawn cells")
    
    for test_num in range(num_tests):
        state_obj = env.create_state()
        state = agent.get_state(env, state_obj)
        path = [state_obj.position]
        total_reward = 0
        
        for step in range(max_steps):
            valid_actions = []
            for action_idx, (dx, dy) in enumerate(env.DIRECTIONS.values()):
                next_pos = (state_obj.position[0] + dx, 
                           state_obj.position[1] + dy)
                if env.is_valid_move(state_obj.position, next_pos):
                    valid_actions.append(action_idx)
            
            if not valid_actions:
                break
            
            action = agent.act(state, valid_actions, training=False)
            
            dx, dy = list(env.DIRECTIONS.values())[action]
            next_pos = (state_obj.position[0] + dx, 
                       state_obj.position[1] + dy)
            state_obj = env.move(state_obj, next_pos)
            path.append(state_obj.position)
            
            state = agent.get_state(env, state_obj)
            
            if env.is_complete(state_obj):
                print(f"  Test {test_num + 1}: Completed in {step + 1} steps!")
                break
        
        # Calculate metrics
        total_reachable_cells = len(env.reachable_lawn_cells)
        mowed_reachable = len(state_obj.mowed_cells.intersection(env.reachable_lawn_cells))
        coverage = 100 * mowed_reachable / total_reachable_cells if total_reachable_cells > 0 else 0
        efficiency = env.calculate_efficiency(state_obj)
        steps = len(path) - 1
        
        # Calculate reward
        reward = 0
        if coverage > 99.9:  # Complete or nearly complete
            reward = 200 + (500 - steps) * 0.5  # Bonus for fewer steps
        else:
            reward = coverage * 2 - steps * 0.1
        
        print(f"\nTest {test_num + 1}/{num_tests}:")
        print(f"  Steps: {steps:3d}")
        print(f"  Coverage: {coverage:5.1f}% of reachable cells")
        print(f"  Efficiency: {efficiency:5.1f}%")
        print(f"  Estimated reward: {reward:7.1f}")
        
        all_paths.append(path)
        all_metrics.append({
            'steps': steps,
            'coverage': coverage,
            'efficiency': efficiency,
            'reward': reward
        })
    
    # Display statistics
    if all_metrics:
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) 
                      for k in all_metrics[0].keys()}
        std_metrics = {k: np.std([m[k] for m in all_metrics]) 
                      for k in all_metrics[0].keys()}
        
        print(f"\n{'='*60}")
        print(f"TEST RESULTS (Average over {num_tests} runs):")
        print(f"{'='*60}")
        print(f"  Steps: {avg_metrics['steps']:.1f} ± {std_metrics['steps']:.1f}")
        print(f"  Coverage: {avg_metrics['coverage']:.1f}% ± {std_metrics['coverage']:.1f}%")
        print(f"  Efficiency: {avg_metrics['efficiency']:.1f}% ± {std_metrics['efficiency']:.1f}%")
        print(f"  Reward: {avg_metrics['reward']:.1f} ± {std_metrics['reward']:.1f}")
        print(f"{'='*60}")
    
    # Visualize best path
    if visualize and all_paths:
        best_idx = max(range(len(all_paths)), 
                      key=lambda i: all_metrics[i]['reward'])
        best_path = all_paths[best_idx]
        best_metrics = all_metrics[best_idx]
        
        viz = LawnmowerVisualizer(env)
        
        if animate:
            save_file = save_animation if save_animation else None
            viz.animate_path(best_path, 
                           title=f"DQN Agent - Best Test Run\n"
                                 f"Steps: {best_metrics['steps']}, "
                                 f"Cov: {best_metrics['coverage']:.1f}%, "
                                 f"Eff: {best_metrics['efficiency']:.1f}%, "
                                 f"Reward: {best_metrics['reward']:.1f}",
                           save_file=save_file)
        else:
            viz.plot_path(best_path, 
                         title=f"DQN Agent - Best Test Run\n"
                               f"Steps: {best_metrics['steps']}, "
                               f"Cov: {best_metrics['coverage']:.1f}%, "
                               f"Eff: {best_metrics['efficiency']:.1f}%, "
                               f"Reward: {best_metrics['reward']:.1f}")
    
    return all_paths


def main():
    parser = argparse.ArgumentParser(
        description='Train or test DQN agent for lawn mowing'
    )
    
    parser.add_argument('map_file', type=str,
                       help='Path to map file')
    parser.add_argument('--train', action='store_true',
                       help='Train the agent')
    parser.add_argument('--test', action='store_true',
                       help='Test the agent')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=2000,
                       help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=300,
                       help='Maximum steps per episode')
    parser.add_argument('--learning-rate', type=float, default=0.00025,
                       help='Learning rate')
    parser.add_argument('--epsilon-decay', type=float, default=0.997,
                       help='Epsilon decay rate (higher = slower decay)')
    parser.add_argument('--patience', type=int, default=200,
                       help='Early stopping patience')
    
    # Model and output
    parser.add_argument('--model', type=str, default='',
                       help='Path to save/load model')
    
    # Testing parameters
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--animate', action='store_true',
                       help='Create animation for test runs')
    parser.add_argument('--save-animation', type=str,
                       help='Save animation to this file')
    parser.add_argument('--tests', type=int, default=5,
                       help='Number of test runs')
    
    args = parser.parse_args()
    
    if not args.train and not args.test:
        parser.error("Must specify either --train or --test")
    
    # Load environment
    print(f"Loading map: {args.map_file}")
    env = LawnmowerEnvironment(args.map_file)
    
    # Generate model path
    if not args.model:
        map_name = Path(args.map_file).stem
        args.model = f'models/dqn_{map_name}.pkl'
    
    if not args.model.endswith('.pkl'):
        args.model += '.pkl'
    
    print(f"Using model file: {args.model}")
    
    # Create models directory
    Path(args.model).parent.mkdir(parents=True, exist_ok=True)
    
    if args.train:
        train_dqn(
            env, 
            episodes=args.episodes, 
            max_steps=args.max_steps,
            model_path=args.model,
            early_stopping_patience=args.patience,
            learning_rate=args.learning_rate,
            epsilon_decay=args.epsilon_decay
        )
    
    if args.test:
        test_dqn(
            env, 
            model_path=args.model, 
            visualize=not args.no_viz,
            animate=args.animate,
            max_steps=args.max_steps,
            num_tests=args.tests,
            save_animation=args.save_animation
        )


if __name__ == '__main__':
    main()