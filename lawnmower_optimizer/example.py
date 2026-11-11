#!/usr/bin/env python3
"""
Example usage script - Demonstrates basic functionality
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.environment import LawnmowerEnvironment
from algorithms.greedy import greedy_algorithm
from algorithms.astar import a_star_coverage
from algorithms.boustrophedon import boustrophedon_algorithm
from algorithms.mst import mst_coverage
from visualization.plotter import LawnmowerVisualizer


def main():
    print("="*70)
    print("Lawnmower Optimizer - Example Usage")
    print("="*70)
    
    # Load environment
    map_file = 'maps/maze6.txt'
    print(f"\n1. Loading map: {map_file}")
    env = LawnmowerEnvironment(map_file)
    print(f"   Grid: {env.width}x{env.height}")
    print(f"   Lawn cells: {len(env.lawn_cells)}")
    print(f"   Obstacles: {len(env.obstacle_cells)}")
    
    # Run algorithms
    algorithms = {
        'Greedy': greedy_algorithm,
        'A*': a_star_coverage,
        'Boustrophedon': boustrophedon_algorithm,
        'MST': mst_coverage,
    }
    
    results = {}
    
    for name, algo_func in algorithms.items():
        print(f"\n2. Running {name} algorithm...")
        path = list(algo_func(env))
        
        mowed = set(path) & env.lawn_cells
        coverage = 100 * len(mowed) / len(env.lawn_cells)
        efficiency = 100 * len(mowed) / len(path) if path else 0
        
        results[name] = (path, {
            'steps': len(path),
            'coverage': coverage,
            'efficiency': efficiency
        })
        
        print(f"   Steps: {len(path)}")
        print(f"   Coverage: {coverage:.1f}%")
        print(f"   Efficiency: {efficiency:.1f}%")
    
    # Visualize comparison
    print(f"\n3. Creating comparison visualization...")
    viz = LawnmowerVisualizer(env)
    viz.compare_algorithms(results)
    
    print("\n" + "="*70)
    print("Example complete! Check the visualization window.")
    print("="*70)


if __name__ == '__main__':
    main()
