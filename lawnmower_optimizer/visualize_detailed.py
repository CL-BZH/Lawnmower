"""
Detailed visualization for debugging:
- Grid labels every 5 units
- A* transition paths in different color (red)
- Boustrophedon sweep paths in green
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils.environment import LawnmowerEnvironment
from algorithms.sweep_optimized import rectangle_decomposition_sweep
import numpy as np

def visualize_detailed(map_file='maps/maze6.txt'):
    env = LawnmowerEnvironment(map_file)
    
    print(f"Generating path...")
    
    # Track which segments are A* vs boustrophedon
    # We'll detect A* by looking for transitions between components
    path = list(rectangle_decomposition_sweep(env))
    
    # Identify A* segments: transitions that cross large distances or change direction abruptly
    astar_segments = []
    boustro_segments = []
    
    current_segment = []
    for i in range(len(path)):
        current_segment.append(path[i])
        
        if i < len(path) - 1:
            # Check if next move looks like A* (large jump or end of sweep)
            dist = abs(path[i+1][0] - path[i][0]) + abs(path[i+1][1] - path[i][1])
            
            # Heuristic: if distance > 2 or we're jumping to a different region, it's A*
            if dist > 10 or (i > 0 and dist > 2):
                # Current segment ends, start A* segment
                if len(current_segment) > 1:
                    boustro_segments.append(current_segment.copy())
                current_segment = [path[i]]  # Start new segment with current position
    
    # Add last segment
    if len(current_segment) > 1:
        boustro_segments.append(current_segment)
    
    # Now detect A* transitions by finding gaps between boustrophedon segments
    covered_indices = set()
    for seg in boustro_segments:
        for pos in seg:
            covered_indices.add(path.index(pos))
    
    # A* segments are the gaps
    i = 0
    while i < len(path):
        if i not in covered_indices:
            # Start of A* segment
            astar_seg = []
            while i < len(path) and i not in covered_indices:
                astar_seg.append(path[i])
                i += 1
            if len(astar_seg) > 0:
                astar_segments.append(astar_seg)
        else:
            i += 1
    
    print(f"Path: {len(path)} total positions")
    print(f"Boustrophedon segments: {len(boustro_segments)}")
    print(f"A* segments: {len(astar_segments)}")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(20, 18))
    
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
    
    # Draw boustrophedon paths (green)
    for segment in boustro_segments:
        if len(segment) > 1:
            xs = [p[0] + 0.5 for p in segment]
            ys = [p[1] + 0.5 for p in segment]
            ax.plot(xs, ys, 'g-', linewidth=2, alpha=0.7, label='Boustrophedon' if segment == boustro_segments[0] else '')
    
    # Draw A* paths (red)
    for segment in astar_segments:
        if len(segment) > 1:
            xs = [p[0] + 0.5 for p in segment]
            ys = [p[1] + 0.5 for p in segment]
            ax.plot(xs, ys, 'r-', linewidth=3, alpha=0.9, label='A* transitions' if segment == astar_segments[0] else '')
    
    # Draw start and end
    ax.plot(env.start_pos[0] + 0.5, env.start_pos[1] + 0.5, 'bo', 
           markersize=15, markeredgewidth=3, markeredgecolor='white', 
           label='Start', zorder=10)
    
    if path:
        ax.plot(path[-1][0] + 0.5, path[-1][1] + 0.5, 'rs', 
               markersize=15, markeredgewidth=3, markeredgecolor='white', 
               label='End', zorder=10)
    
    # Grid labels every 5 units
    ax.set_xticks(range(0, env.width, 5))
    ax.set_yticks(range(0, env.height, 5))
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Minor grid every 1 unit
    ax.set_xticks(range(0, env.width, 1), minor=True)
    ax.set_yticks(range(0, env.height, 1), minor=True)
    ax.grid(True, which='minor', alpha=0.1, linewidth=0.3)
    
    ax.set_xlabel('X', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y', fontsize=14, fontweight='bold')
    
    # Calculate coverage
    unique_positions = set(path)
    coverage_pct = 100 * len(unique_positions) / len(env.reachable_lawn_cells)
    
    # Count direction changes
    direction_changes = 0
    for i in range(2, len(path)):
        prev_dir = (path[i-1][0] - path[i-2][0], path[i-1][1] - path[i-2][1])
        curr_dir = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        if prev_dir != curr_dir:
            direction_changes += 1
    
    title = f'SWEEPOPT Algorithm - Detailed View\n'
    title += f'Steps: {len(path)}, Coverage: {coverage_pct:.1f}%, Direction Changes: {direction_changes} ({100*direction_changes/len(path):.1f}%)\n'
    title += f'Green = Boustrophedon sweep, Red = A* transitions'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('./visualization/sweep_detailed.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved to ./visualization/sweep_detailed.png")
    
    plt.close()

if __name__ == '__main__':
    visualize_detailed()
