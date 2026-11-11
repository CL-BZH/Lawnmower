"""
Visualization script for sweep algorithm debugging
Shows: initial rectangles, merged groups, path with invalid moves
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils.environment import LawnmowerEnvironment
from algorithms.sweep_optimized import _decompose_into_rectangles, _merge_adjacent_rectangles, rectangle_decomposition_sweep
import numpy as np

def visualize_all(map_file='maps/maze6.txt'):
    env = LawnmowerEnvironment(map_file)
    
    # Step 1: Decompose into rectangles
    matrix = np.zeros((env.height, env.width), dtype=int)
    for x, y in env.all_lawn_cells:  # Using ALL lawn cells (1752 cells)
        matrix[y, x] = 1
    
    rectangles = _decompose_into_rectangles(matrix, env)
    print(f"Step 1: Decomposed into {len(rectangles)} rectangles")
    
    # Step 2: Merge rectangles into groups
    groups = _merge_adjacent_rectangles(rectangles, env)
    print(f"Step 2: Merged into {len(groups)} groups")
    
    # Step 3: Generate path
    print(f"Step 3: Generating path...")
    path = list(rectangle_decomposition_sweep(env))
    
    # Find invalid moves
    invalid_moves = []
    for i in range(1, len(path)):
        if path[i] not in env.get_neighbors(path[i-1]):
            dist = abs(path[i][0] - path[i-1][0]) + abs(path[i][1] - path[i-1][1])
            if dist > 2:
                invalid_moves.append((i, path[i-1], path[i], dist))
    
    print(f"Found {len(invalid_moves)} invalid moves")
    
    # Create 3-panel visualization
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    
    colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 
              'brown', 'pink', 'lime', 'indigo', 'gold', 'navy', 'maroon', 'teal']
    
    # ====================
    # PANEL 1: Initial Rectangles
    # ====================
    ax = axes[0]
    
    # Draw obstacles (black)
    for y in range(env.height):
        for x in range(env.width):
            if (x, y) not in env.all_lawn_cells:
                ax.add_patch(mpatches.Rectangle((x, y), 1, 1, facecolor='black'))
    
    # Draw each rectangle with different color
    for i, rect in enumerate(rectangles):
        color = colors[i % len(colors)]
        x, y = rect['x'], rect['y']
        w, h = rect['width'], rect['height']
        
        ax.add_patch(mpatches.Rectangle((x, y), w, h, 
                                        facecolor=color, alpha=0.3,
                                        edgecolor=color, linewidth=2))
        ax.text(x + w/2, y + h/2, f'R{i}\n{w}x{h}', 
               fontsize=7, fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='white'))
    
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(f'Initial Rectangles: {len(rectangles)}\n(Decomposed from all_lawn_cells={len(env.all_lawn_cells)})', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    # ====================
    # PANEL 2: Merged Groups
    # ====================
    ax = axes[1]
    
    # Draw obstacles
    for y in range(env.height):
        for x in range(env.width):
            if (x, y) not in env.all_lawn_cells:
                ax.add_patch(mpatches.Rectangle((x, y), 1, 1, facecolor='black'))
    
    # Draw each group's rectangles in same color
    for group_idx, group_rect_ids in enumerate(groups):
        color = colors[group_idx % len(colors)]
        
        # Draw all rectangles in this group with same color
        for rect_id in group_rect_ids:
            rect = rectangles[rect_id]
            x, y = rect['x'], rect['y']
            w, h = rect['width'], rect['height']
            
            ax.add_patch(mpatches.Rectangle((x, y), w, h, 
                                            facecolor=color, alpha=0.4,
                                            edgecolor=color, linewidth=2))
        
        # Calculate group bounding box for label
        group_rects = [rectangles[i] for i in group_rect_ids]
        min_x = min(r['x'] for r in group_rects)
        max_x = max(r['x'] + r['width'] for r in group_rects)
        min_y = min(r['y'] for r in group_rects)
        max_y = max(r['y'] + r['height'] for r in group_rects)
        
        # Label at center
        ax.text((min_x + max_x)/2, (min_y + max_y)/2, 
               f'Group {group_idx}\n{len(group_rect_ids)} rects', 
               fontsize=12, fontweight='bold', ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=2))
    
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(f'Merged Groups: {len(groups)}\n(100% full-side sharing with flood-fill)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    # ====================
    # PANEL 3: Path with Invalid Moves
    # ====================
    ax = axes[2]
    
    # Draw obstacles
    for y in range(env.height):
        for x in range(env.width):
            if (x, y) not in env.all_lawn_cells:
                ax.add_patch(mpatches.Rectangle((x, y), 1, 1, facecolor='black'))
    
    # Draw path (thin green line)
    if len(path) > 1:
        path_x = [p[0] + 0.5 for p in path]
        path_y = [p[1] + 0.5 for p in path]
        ax.plot(path_x, path_y, 'g-', linewidth=0.5, alpha=0.3)
    
    # Highlight invalid moves in RED
    for i, prev, curr, dist in invalid_moves:
        ax.plot([prev[0] + 0.5, curr[0] + 0.5], 
               [prev[1] + 0.5, curr[1] + 0.5], 
               'r-', linewidth=4, alpha=0.9)
        ax.plot(prev[0] + 0.5, prev[1] + 0.5, 'ro', markersize=10)
        ax.plot(curr[0] + 0.5, curr[1] + 0.5, 'rx', markersize=10, markeredgewidth=3)
        ax.text((prev[0] + curr[0])/2 + 0.5, (prev[1] + curr[1])/2 + 0.5, 
               f'd={dist}', fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Start position
    ax.plot(env.start_pos[0] + 0.5, env.start_pos[1] + 0.5, 'bo', markersize=12, 
           markeredgewidth=2, markeredgecolor='white', label='Start')
    
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_title(f'Path: {len(path)} steps, {len(invalid_moves)} invalid moves\n(Coverage: {100*len(set(path))/len(env.lawn_cells):.1f}%)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('./visualization/sweep_debug.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved to ./visualization/sweep_debug.png")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Map cells:")
    print(f"  all_lawn_cells (total non-obstacle): {len(env.all_lawn_cells)}")
    print(f"  reachable_lawn_cells (can be mowed): {len(env.reachable_lawn_cells)}")
    print(f"  lawn_cells (mower positions): {len(env.lawn_cells)}")
    print(f"\nRectangles: {len(rectangles)}")
    print(f"Groups: {len(groups)}")
    for i, g in enumerate(groups):
        print(f"  Group {i}: {len(g)} rectangles")
    print(f"\nPath: {len(path)} steps")
    print(f"Unique positions visited: {len(set(path))}")
    print(f"Invalid moves: {len(invalid_moves)}")
    
    plt.close()

if __name__ == '__main__':
    visualize_all()
