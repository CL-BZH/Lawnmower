"""
Test and visualize the trained RL agent
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from train_rl_agent import LawnmowerRLAgent
from utils.environment import LawnmowerEnvironment


def visualize_path(env: LawnmowerEnvironment, path: list, title: str, save_path: str = None):
    """
    Visualize a path on the map
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    
    # Draw obstacles (dark gray)
    for y in range(env.height):
        for x in range(env.width):
            if (x, y) not in env.all_lawn_cells:
                ax.add_patch(mpatches.Rectangle((x, y), 1, 1, 
                                                facecolor='#333333', 
                                                edgecolor='none'))
    
    # Draw lawn area (light green background)
    for x, y in env.all_lawn_cells:
        ax.add_patch(mpatches.Rectangle((x, y), 1, 1, 
                                        facecolor='#90EE90', 
                                        edgecolor='none', 
                                        alpha=0.3))
    
    # Draw path
    if len(path) > 1:
        xs = [p[0] + 0.5 for p in path]
        ys = [p[1] + 0.5 for p in path]
        ax.plot(xs, ys, 'purple', linewidth=1.5, alpha=0.7, label='Path')
    
    # Draw start (blue) and end (orange)
    if path:
        ax.plot(path[0][0] + 0.5, path[0][1] + 0.5, 'bo', markersize=12, label='Start')
        ax.plot(path[-1][0] + 0.5, path[-1][1] + 0.5, 'o', color='orange', markersize=12, label='End')
    
    # Grid and labels
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_algorithms(env: LawnmowerEnvironment, rl_agent: LawnmowerRLAgent, num_runs: int = 5):
    """
    Compare RL agent against expert algorithms
    """
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON")
    print("="*80)
    
    results = {}
    
    # Test expert algorithms
    expert_algos = {
        'hybrid': None,
        'ccpp': None,
        'sweepopt': None,
    }
    
    for algo_name in expert_algos.keys():
        print(f"\nTesting {algo_name}...")
        
        try:
            if algo_name == 'hybrid':
                from algorithms.hybrid import hybrid_coverage
                path = list(hybrid_coverage(env))
            elif algo_name == 'ccpp':
                from algorithms.ccpp import CompleteCoveragePlanner
                planner = CompleteCoveragePlanner(env, show_visualization=False)
                path = planner.plan_full_coverage()
            elif algo_name == 'sweepopt':
                from algorithms.sweep_optimized import rectangle_decomposition_sweep
                path = list(rectangle_decomposition_sweep(env))
            
            # Calculate metrics
            mowed = set()
            direction_changes = 0
            last_action = None
            
            for i, pos in enumerate(path):
                blade = env.get_blade_coverage(pos)
                for cell in blade:
                    if cell in env.reachable_lawn_cells:
                        mowed.add(cell)
                
                if i > 0:
                    from train_rl_agent import ActionMapper
                    action = ActionMapper.delta_to_action(path[i-1], pos)
                    if last_action is not None and action != last_action:
                        direction_changes += 1
                    last_action = action
            
            coverage = 100 * len(mowed) / len(env.reachable_lawn_cells)
            efficiency = 100 * len(mowed) / len(path) if len(path) > 0 else 0
            
            results[algo_name] = {
                'coverage': coverage,
                'steps': len(path),
                'efficiency': efficiency,
                'direction_changes': direction_changes,
                'path': path
            }
            
            print(f"  Coverage: {coverage:.1f}%")
            print(f"  Steps: {len(path)}")
            print(f"  Efficiency: {efficiency:.1f}%")
            print(f"  Direction Changes: {direction_changes}")
        
        except Exception as e:
            print(f"  Error: {e}")
            results[algo_name] = None
    
    # Test RL agent
    print(f"\nTesting RL Agent (averaged over {num_runs} runs)...")
    
    rl_coverages = []
    rl_steps = []
    rl_efficiencies = []
    rl_direction_changes = []
    rl_paths = []
    
    for run in range(num_runs):
        # Run episode
        current_pos = env.start_pos
        mowed_cells = set()
        path = [current_pos]
        last_action = None
        direction_changes = 0
        
        for step in range(2000):
            # Update mowed cells
            blade_coverage = env.get_blade_coverage(current_pos)
            for cell in blade_coverage:
                if cell in env.reachable_lawn_cells:
                    mowed_cells.add(cell)
            
            # Check if done
            if len(mowed_cells & env.reachable_lawn_cells) >= len(env.reachable_lawn_cells):
                break
            
            # Get action from policy
            state = rl_agent.encoder.encode_state(current_pos, mowed_cells, last_action)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(rl_agent.device)
            
            with torch.no_grad():
                action = rl_agent.policy.get_action(state_tensor, deterministic=True).item()
            
            # Take action
            from train_rl_agent import ActionMapper
            dx, dy = ActionMapper.action_to_delta(action)
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # Check if valid move
            if next_pos not in env.lawn_cells:
                # Try random valid neighbor
                neighbors = env.get_neighbors(current_pos)
                if neighbors:
                    next_pos = list(neighbors)[0]
                else:
                    break
            
            # Count direction change
            if last_action is not None and action != last_action:
                direction_changes += 1
            
            path.append(next_pos)
            current_pos = next_pos
            last_action = action
        
        # Calculate metrics
        final_coverage = len(mowed_cells & env.reachable_lawn_cells)
        coverage_pct = 100 * final_coverage / len(env.reachable_lawn_cells)
        efficiency = 100 * final_coverage / len(path) if len(path) > 0 else 0
        
        rl_coverages.append(coverage_pct)
        rl_steps.append(len(path))
        rl_efficiencies.append(efficiency)
        rl_direction_changes.append(direction_changes)
        rl_paths.append(path)
    
    results['rl_agent'] = {
        'coverage': np.mean(rl_coverages),
        'steps': np.mean(rl_steps),
        'efficiency': np.mean(rl_efficiencies),
        'direction_changes': np.mean(rl_direction_changes),
        'path': rl_paths[0]  # Use first path for visualization
    }
    
    print(f"  Average Coverage: {results['rl_agent']['coverage']:.1f}%")
    print(f"  Average Steps: {results['rl_agent']['steps']:.0f}")
    print(f"  Average Efficiency: {results['rl_agent']['efficiency']:.1f}%")
    print(f"  Average Direction Changes: {results['rl_agent']['direction_changes']:.0f}")
    
    # Print comparison table
    print("\n" + "="*80)
    print(f"{'Algorithm':<15} {'Steps':<10} {'Coverage':<12} {'Efficiency':<12} {'Dir Changes':<12}")
    print("="*80)
    
    for algo_name, result in results.items():
        if result:
            print(f"{algo_name:<15} {result['steps']:<10.0f} {result['coverage']:<12.1f} "
                  f"{result['efficiency']:<12.1f} {result['direction_changes']:<12.0f}")
    
    print("="*80)
    
    return results


def visualize_comparison(results: dict, save_path: str = 'plots/algorithm_comparison.png'):
    """
    Create visualization comparing algorithms
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    algorithms = [name for name, res in results.items() if res is not None]
    
    # Coverage comparison
    coverages = [results[algo]['coverage'] for algo in algorithms]
    axes[0, 0].bar(algorithms, coverages, color=['green', 'blue', 'orange', 'purple'][:len(algorithms)])
    axes[0, 0].set_ylabel('Coverage (%)')
    axes[0, 0].set_title('Coverage Comparison')
    axes[0, 0].set_ylim([90, 105])
    axes[0, 0].grid(True, alpha=0.3)
    
    # Steps comparison
    steps = [results[algo]['steps'] for algo in algorithms]
    axes[0, 1].bar(algorithms, steps, color=['green', 'blue', 'orange', 'purple'][:len(algorithms)])
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Steps Comparison (Lower is Better)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Efficiency comparison
    efficiencies = [results[algo]['efficiency'] for algo in algorithms]
    axes[1, 0].bar(algorithms, efficiencies, color=['green', 'blue', 'orange', 'purple'][:len(algorithms)])
    axes[1, 0].set_ylabel('Efficiency (%)')
    axes[1, 0].set_title('Efficiency Comparison')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Direction changes comparison
    dir_changes = [results[algo]['direction_changes'] for algo in algorithms]
    axes[1, 1].bar(algorithms, dir_changes, color=['green', 'blue', 'orange', 'purple'][:len(algorithms)])
    axes[1, 1].set_ylabel('Direction Changes')
    axes[1, 1].set_title('Direction Changes (Lower is Better)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Comparison visualization saved to {save_path}")
    plt.close()


def main():
    """Main testing script"""
    import os
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained RL agent')
    parser.add_argument('--model', type=str, default='best_ppo_model.pt',
                       help='Model to test (default: best_ppo_model.pt)')
    parser.add_argument('--num-runs', type=int, default=5,
                       help='Number of test runs per algorithm (default: 5)')
    args = parser.parse_args()
    
    # Load environment
    print("Loading environment...")
    env = LawnmowerEnvironment('maps/maze6.txt')
    
    # Create and load trained agent
    print("Loading trained RL agent...")
    agent = LawnmowerRLAgent(env)
    
    # Load specified model
    model_path = f'models/{args.model}'
    if not os.path.exists(model_path):
        # Try some common alternatives
        alternatives = [
            'models/best_ppo_model.pt',
            'models/final_ppo_model.pt',
            'models/best_pretrained_model.pt',
            'models/pretrained_il_model.pt'
        ]
        for alt in alternatives:
            if os.path.exists(alt):
                model_path = alt
                print(f"Specified model not found, using {alt} instead...")
                break
    
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"No trained model found. Please train the agent first.")
        return
    
    # Compare against expert algorithms
    results = compare_algorithms(env, agent, num_runs=args.num_runs)
    
    # Visualize comparison
    os.makedirs('plots', exist_ok=True)
    visualize_comparison(results)
    
    # Visualize individual paths
    for algo_name, result in results.items():
        if result and result['path']:
            visualize_path(
                env,
                result['path'],
                title=f"{algo_name.upper()} - Steps: {result['steps']:.0f}, Coverage: {result['coverage']:.1f}%",
                save_path=f"plots/{algo_name}_path.png"
            )
    
    print("\n" + "="*80)
    print("Testing complete! Check the 'plots' directory for visualizations.")
    print("="*80)


if __name__ == "__main__":
    main()
