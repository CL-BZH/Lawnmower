#!/usr/bin/env python3
"""
Lawnmower Optimizer - Main execution script
NO CACHING VERSION - Always runs algorithms fresh
"""

import argparse
import sys
from pathlib import Path
import time
from collections import deque

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.environment import LawnmowerEnvironment
from utils.config import ALGO_PARAMS
from visualization.plotter import LawnmowerVisualizer
from algorithms.greedy import greedy_algorithm
from algorithms.astar import a_star_coverage
from algorithms.boustrophedon import boustrophedon_algorithm
from algorithms.hamiltonian import hamiltonian_path
from algorithms.mst import mst_coverage
from algorithms.hybrid import hybrid_coverage
from algorithms.ccpp import ccpp_algorithm
from algorithms.decomposed_boustrophedon import decomposed_boustrophedon
from algorithms.mst_boustrophedon import mst_boustrophedon_hybrid
from algorithms.cell_decomposition import cell_decomposition_boustrophedon
from algorithms.sweep_optimized import rectangle_decomposition_sweep as sweep_optimized_coverage
from train_rl_agent import LawnmowerRLAgent


def simulate_path_with_proper_coverage(env, path):
    """
    Simulate a path with proper coverage accounting for diagonal moves
    and blade coverage throughout movement
    """
    mowed_cells = set()
    state = env.create_state()
    
    if not path:
        return mowed_cells
    
    # Add starting position coverage
    start_blade = env.get_blade_coverage(path[0])
    for cell in start_blade:
        if cell in env.reachable_lawn_cells:
            mowed_cells.add(cell)
    
    # Simulate the entire path
    for i in range(1, len(path)):
        from_pos = path[i-1]
        to_pos = path[i]
        
        if not env.is_valid_move(from_pos, to_pos):
            print(f"  ⚠️ Invalid move at step {i}: {from_pos} -> {to_pos}")
            continue
        
        # Get the move components
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        # Mow at starting position (already done for first position)
        if i > 1:
            from_blade = env.get_blade_coverage(from_pos)
            for cell in from_blade:
                if cell in env.reachable_lawn_cells:
                    mowed_cells.add(cell)
        
        # Mow at ending position
        to_blade = env.get_blade_coverage(to_pos)
        for cell in to_blade:
            if cell in env.reachable_lawn_cells:
                mowed_cells.add(cell)
        
        # For diagonal moves, mow the intermediate cells
        if dx != 0 and dy != 0:
            # Mow the corners that are physically crossed
            corner1 = (from_pos[0] + dx, from_pos[1])
            corner2 = (from_pos[0], from_pos[1] + dy)
            
            for corner in [corner1, corner2]:
                corner_blade = env.get_blade_coverage(corner)
                for cell in corner_blade:
                    if cell in env.reachable_lawn_cells:
                        mowed_cells.add(cell)
    
    return mowed_cells


def run_algorithm(env, algorithm_name, visualize=True, 
                  animate=False, save_path=None, **kwargs):
    """
    Run a specific algorithm on the environment
    NO CACHING - Always runs fresh
    """
    print(f"\n{'='*60}")
    print(f"Running: {algorithm_name.upper()}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Select and run algorithm
    if algorithm_name == 'greedy':
        path_gen = greedy_algorithm(
            env, 
            look_ahead=ALGO_PARAMS['greedy'][
                'look_ahead_distance'])
    elif algorithm_name == 'astar':
        path_gen = a_star_coverage(env)
    elif algorithm_name == 'boustrophedon':
        path_gen = boustrophedon_algorithm(env)
    elif algorithm_name == 'hamiltonian':
        path_gen = hamiltonian_path(env)
    elif algorithm_name == 'mst':
        path_gen = mst_coverage(env)
    elif algorithm_name == 'hybrid':
        path_gen = hybrid_coverage(env)
    elif algorithm_name == 'ccpp':
        path_gen = ccpp_algorithm(env, 
                                 show_visualization=not animate)  # Pass flag to skip built-in viz if animating
    elif algorithm_name == 'decomposed':
        path_gen = decomposed_boustrophedon(env)
    elif algorithm_name == 'mstboustrophedon':
        path_gen = mst_boustrophedon_hybrid(env)
    elif algorithm_name == 'celldecomp':
        path_gen = cell_decomposition_boustrophedon(env)
    elif algorithm_name == 'sweepopt':
        path_gen = sweep_optimized_coverage(env)
    elif algorithm_name == 'rl_agent':
        # Load trained RL agent
        import os
        import torch
        
        print(f"  Loading RL agent...")
        agent = LawnmowerRLAgent(env)
        
        # Try to load best PPO model, fall back to IL model
        model_path = kwargs.get('rl_model', 'models/best_ppo_model.pt')
        if not os.path.exists(model_path):
            model_path = 'models/best_pretrained_model.pt'
            print(f"  PPO model not found, using IL model...")
        
        if os.path.exists(model_path):
            agent.load_model(model_path)
            print(f"  Loaded model from {model_path}")
        else:
            raise FileNotFoundError(f"No trained RL model found at {model_path}. Please train first.")
        
        # Generate path using RL agent
        print(f"  Running RL agent...")
        agent.policy.eval()
        
        current_pos = env.start_pos
        mowed_cells = set()
        path = [current_pos]
        last_action = None
        path_history = deque(maxlen=50)  # Track last 50 actions for state encoding
        
        # Loop detection: track position visits
        position_visits = {}
        max_visits_per_position = 5  # Allow max 5 visits to same position
        
        # Track if making progress
        last_coverage_check = 0
        steps_since_progress = 0
        max_steps_without_progress = 100  # Stop if no progress for 100 steps
        
        with torch.no_grad():
            for step in range(kwargs.get('max_steps', 2500)):
                # Loop detection: check if stuck at same position
                position_visits[current_pos] = position_visits.get(current_pos, 0) + 1
                if position_visits[current_pos] > max_visits_per_position:
                    print(f"  Breaking: Agent stuck at position {current_pos} (visited {position_visits[current_pos]} times)")
                    break
                
                # Encode state with path history
                state = agent.encoder.encode_state(current_pos, mowed_cells, last_action, path_history)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                
                # Get action (stochastic to avoid getting stuck in loops)
                # Using stochastic policy adds exploration to escape local minima
                action = agent.policy.get_action(state_tensor, deterministic=False).item()
                
                # Convert action to movement
                from train_rl_agent import ActionMapper
                dx, dy = ActionMapper.action_to_delta(action)
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                
                # Check if valid move - must be in valid mower positions
                if next_pos not in env.lawn_cells:
                    # Invalid move - stay in place
                    continue
                
                # CRITICAL CHECK: Verify blade won't hit obstacles
                # (lawn_cells should already exclude these, but double-check)
                blade_test = env.get_blade_coverage(next_pos)
                obstacle_hit = [cell for cell in blade_test if cell in env.obstacle_cells]
                if obstacle_hit:
                    print(f"  ERROR: Position {next_pos} in lawn_cells but blade would hit obstacles {obstacle_hit}!")
                    print(f"  This means env.lawn_cells is computed incorrectly!")
                    continue
                
                # Valid move
                path.append(next_pos)
                current_pos = next_pos
                last_action = action
                path_history.append(action)
                
                # Update mowed cells
                blade = env.get_blade_coverage(current_pos)
                for cell in blade:
                    if cell in env.reachable_lawn_cells:
                        mowed_cells.add(cell)
                
                # Check if done or making progress
                current_coverage = len(mowed_cells & env.reachable_lawn_cells)
                coverage_ratio = current_coverage / len(env.reachable_lawn_cells)
                
                # Track progress
                if current_coverage > last_coverage_check:
                    last_coverage_check = current_coverage
                    steps_since_progress = 0
                else:
                    steps_since_progress += 1
                
                # Break if no progress for too long
                if steps_since_progress > max_steps_without_progress:
                    print(f"  Breaking: No coverage progress for {max_steps_without_progress} steps")
                    break
                
                if coverage_ratio >= 0.999:  # 99.9% coverage
                    break
        
        # Return as generator (wrap in iterator)
        path_gen = iter(path)
    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm_name}")
    
    # Convert generator to list
    print(f"  Executing algorithm...")
    path = list(path_gen)
    elapsed_time = time.time() - start_time
    
    # Calculate metrics using proper coverage simulation
    print(f"  Calculating coverage...")
    mowed_cells = simulate_path_with_proper_coverage(env, path)
    
    total_reachable = len(env.reachable_lawn_cells)
    
    metrics = {
        'steps': len(path),
        'coverage': (100 * len(mowed_cells) / total_reachable 
                    if total_reachable > 0 else 0),
        'efficiency': (100 * len(mowed_cells) / len(path) 
                      if path else 0),
        'time': elapsed_time,
        'mowed_cells': len(mowed_cells),
        'total_reachable': total_reachable,
        'unmowed_cells': total_reachable - len(mowed_cells)
    }
    
    print(f"\nResults:")
    print(f"  Total steps: {metrics['steps']}")
    print(f"  Coverage: {metrics['coverage']:.1f}% ({metrics['mowed_cells']}/{metrics['total_reachable']})")
    print(f"  Efficiency: {metrics['efficiency']:.1f}%")
    print(f"  Time: {metrics['time']:.3f}s")
    
    if metrics['unmowed_cells'] > 0:
        print(f"  ⚠️  {metrics['unmowed_cells']} cells not covered!")
        # Show first few unmowed cells for debugging
        unmowed = env.reachable_lawn_cells - mowed_cells
        if unmowed:
            print(f"  First 5 unmowed cells: {list(unmowed)[:5]}")
    
    # Visualization
    # Only show if visualize is True OR animate is True
    if visualize or animate:
        viz = LawnmowerVisualizer(env)
        
        if animate:
            save_file = (f"{save_path}_{algorithm_name}.gif" 
                        if save_path else None)
            viz.animate_path(
                path, 
                title=f"{algorithm_name.upper()} Algorithm\n"
                      f"Steps: {metrics['steps']}, "
                      f"Coverage: {metrics['coverage']:.1f}%", 
                save_file=save_file)
        else:
            save_file = (f"{save_path}_{algorithm_name}.png" 
                        if save_path else None)
            viz.plot_path(
                path, 
                title=f"{algorithm_name.upper()} Algorithm\n"
                      f"Steps: {metrics['steps']}, "
                      f"Coverage: {metrics['coverage']:.1f}%",
                save_file=save_file)
    
    return path, metrics


def compare_algorithms(env, algorithms, save_path=None, rl_model=None):
    """
    Run multiple algorithms and compare results
    NO CACHING
    """
    results = {}
    
    for algo in algorithms:
        print(f"\n{'='*60}")
        print(f"Testing: {algo.upper()}")
        print(f"{'='*60}")
        
        path, metrics = run_algorithm(
            env, algo, visualize=False, rl_model=rl_model)
        results[algo] = (path, metrics)
    
    # Print comparison table
    print(f"\n{'='*90}")
    print(f"{'Algorithm':<20} {'Steps':<10} "
          f"{'Coverage':<12} {'Efficiency':<12} {'Time':<10} {'Unmowed':<10}")
    print(f"{'='*90}")
    
    for algo, (path, metrics) in results.items():
        print(f"{algo:<20} {metrics['steps']:<10} "
              f"{metrics['coverage']:<12.1f} "
              f"{metrics['efficiency']:<12.1f} "
              f"{metrics['time']:<10.3f} "
              f"{metrics['unmowed_cells']:<10}")
    
    print(f"{'='*90}")
    
    # Find best algorithm by coverage, then efficiency
    best_algo = max(results.keys(), 
                    key=lambda a: (results[a][1]['coverage'], 
                                  -results[a][1]['steps']))
    best_metrics = results[best_algo][1]
    
    print(f"\n🏆 BEST ALGORITHM: {best_algo.upper()}")
    print(f"   Coverage: {best_metrics['coverage']:.1f}%")
    print(f"   Steps: {best_metrics['steps']}")
    print(f"   Efficiency: {best_metrics['efficiency']:.1f}%")
    print(f"   Unmowed cells: {best_metrics['unmowed_cells']}")


def main():
    parser = argparse.ArgumentParser(
        description='Lawnmower Path Optimization - NO CACHING',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run greedy algorithm on a map (fresh every time)
  python main.py maps/maze6.txt --algorithm greedy
  
  # Run the CCPP algorithm
  python main.py maps/maze6.txt --algorithm ccpp
  
  # Run CCPP with animation
  python main.py maps/maze6.txt --algorithm ccpp --animate
  
  # Compare multiple algorithms (all run fresh)
  python main.py maps/maze6.txt --compare greedy astar boustrophedon ccpp
  
  # Create animation and save it
  python main.py maps/maze6.txt --algorithm astar --animate --save results/animation
  
  # Save results without visualization
  python main.py maps/maze6.txt --algorithm ccpp --save results/output --no-viz
        """
    )
    
    parser.add_argument('map_file', type=str,
                       help='Path to map file (text grid)')
    parser.add_argument(
        '--algorithm', '-a', type=str,
        choices=['greedy', 'astar', 'boustrophedon', 
                'hamiltonian', 'mst', 'hybrid', 'ccpp', 'decomposed', 'mstboustrophedon', 'celldecomp', 'sweepopt', 'rl_agent'],
        help='Algorithm to run')
    parser.add_argument(
        '--compare', '-c', nargs='+',
        choices=['greedy', 'astar', 'boustrophedon', 
                'hamiltonian', 'mst', 'hybrid', 'ccpp', 'decomposed', 'mstboustrophedon', 'celldecomp', 'sweepopt', 'rl_agent'],
        help='Compare multiple algorithms')
    parser.add_argument('--animate', action='store_true',
                       help='Create animation instead of static plot')
    parser.add_argument('--save', '-s', type=str,
                       help='Path prefix for saving results')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable all visualization (including animations)')
    parser.add_argument('--blade-diameter', type=float, default=70.0,
                       help='Mower blade diameter in cm (default: 70)')
    parser.add_argument('--rl-model', type=str, default='models/best_ppo_model.pt',
                       help='Path to trained RL model (default: models/best_ppo_model.pt)')
    parser.add_argument('--debug', action='store_true',
                       help='Show debug information')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.algorithm and not args.compare:
        parser.error(
            "Must specify either --algorithm or --compare")
    
    if args.algorithm and args.compare:
        parser.error(
            "Cannot specify both --algorithm and --compare")
    
    # Load environment
    print(f"Loading map: {args.map_file}")
    env = LawnmowerEnvironment(args.map_file, 
                               blade_diameter=args.blade_diameter)
    
    print(f"Grid size: {env.width} x {env.height}")
    print(f"All lawn cells: {len(env.all_lawn_cells)}")
    print(f"Reachable lawn cells: {len(env.reachable_lawn_cells)}")
    print(f"Obstacles: {len(env.obstacle_cells)}")
    
    if args.debug:
        print(f"\nDebug Info:")
        print(f"  Blade coverage pattern for position (0,0): {env.get_blade_coverage((0, 0))}")
        print(f"  Example reachable cell: {next(iter(env.reachable_lawn_cells)) if env.reachable_lawn_cells else 'None'}")
    
    # Run algorithm(s)
    if args.compare:
        compare_algorithms(
            env, args.compare, 
            save_path=args.save,
            rl_model=args.rl_model)
    else:
        run_algorithm(
            env, args.algorithm, 
            visualize=not args.no_viz,
            animate=args.animate,
            save_path=args.save,
            rl_model=args.rl_model)


if __name__ == '__main__':
    main()
