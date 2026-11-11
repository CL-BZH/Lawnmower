"""
SIMPLE, RELIABLE LAWNMOWER ALGORITHM

No fancy algorithms, just simple back-and-forth
Manual obstacle checking
Lane skipping when possible
"""
import numpy as np
from typing import Tuple, List, Generator, Set
from collections import defaultdict


def hybrid_coverage(env) -> Generator[Tuple[int, int], None, None]:
    """
    Simple, reliable lawnmower algorithm using environment validation
    """
    print("Building simple reliable lawnmower coverage...")
    
    # Use environment's valid positions
    safe_positions = env.lawn_cells
    
    print(f"Safe positions: {len(safe_positions)}")
    
    # 2. Group safe positions by row for boustrophedon pattern
    rows = defaultdict(list)
    for x, y in safe_positions:
        rows[y].append((x, y))
    
    # Sort rows
    row_numbers = sorted(rows.keys())
    
    if not row_numbers:
        return
    
    print(f"Rows to cover: {len(row_numbers)}")
    
    # 3. Build simple back-and-forth path
    path = []
    current = env.start_pos
    
    # Ensure start is safe
    if current not in safe_positions:
        # Find nearest safe position
        nearest = min(safe_positions,
                     key=lambda p: (p[0]-current[0])**2 + (p[1]-current[1])**2)
        # Move to nearest safely
        move_to = _move_safely_env(env, current, nearest, safe_positions)
        if move_to:
            path.extend(move_to)
            current = nearest
    
    path.append(current)
    
    # 4. Process rows in boustrophedon with lane skipping
    direction = 'right'  # Start going right
    rows_visited = set()
    
    # Determine if we should skip rows (every 2 rows)
    skip_rows = True  # Always try to skip
    
    i = 0
    while i < len(row_numbers):
        row_y = row_numbers[i]
        
        if row_y in rows_visited:
            i += 1
            continue
        
        # Get positions in this row
        row_positions = rows[row_y]
        if not row_positions:
            i += 1
            continue
        
        # Sort row by x
        row_sorted = sorted(row_positions, key=lambda p: p[0])
        
        # Move to this row if needed
        if current[1] != row_y:
            # Choose entry point based on direction
            if direction == 'right':
                target = row_sorted[0]  # Leftmost
            else:
                target = row_sorted[-1]  # Rightmost
            
            # Move safely to target using environment pathfinding
            move_path = _move_safely_env(env, current, target, safe_positions)
            if move_path:
                path.extend(move_path)
                current = target
        
        # Traverse row based on direction
        if direction == 'right':
            # Go from left to right
            row_to_cover = sorted([p for p in row_sorted if p != current],
                                 key=lambda p: p[0])
        else:
            # Go from right to left
            row_to_cover = sorted([p for p in row_sorted if p != current],
                                 key=lambda p: p[0], reverse=True)
        
        # Visit each position in row
        for target in row_to_cover:
            # Check if we can move directly
            if env.is_valid_move(current, target):
                path.append(target)
                current = target
            else:
                # Need pathfinding
                move_path = _find_safe_path_env(env, current, target, safe_positions)
                if move_path:
                    path.extend(move_path)
                    current = target
        
        rows_visited.add(row_y)
        
        # Change direction for next row
        direction = 'left' if direction == 'right' else 'right'
        
        # Lane skipping: try to skip one row
        if skip_rows and i + 2 < len(row_numbers):
            next_row = row_numbers[i + 2]
            if next_row not in rows_visited:
                # Skip one row!
                i += 1  # Skip next iteration
                print(f"  Lane skipping: row {row_y} → {next_row}")
        
        i += 1
    
    # 5. Calculate coverage using environment
    covered = set()
    for pos in path:
        blade = env.get_blade_coverage(pos)
        for cell in blade:
            if cell in env.reachable_lawn_cells:
                covered.add(cell)
    
    total_lawn = len(env.reachable_lawn_cells)
    
    print(f"\nResults:")
    print(f"  Steps: {len(path)}")
    print(f"  Covered: {len(covered)}/{total_lawn} ({100*len(covered)/total_lawn:.1f}%)")
    print(f"  Unique positions: {len(set(path))}")
    print(f"  Efficiency: {100*len(set(path))/len(path):.1f}%")
    
    # 6. Verify all moves are safe
    unsafe_count = 0
    for j in range(len(path) - 1):
        if not env.is_valid_move(path[j], path[j+1]):
            unsafe_count += 1
    
    if unsafe_count == 0:
        print(f"  All moves are safe!")
    else:
        print(f"  WARNING: {unsafe_count} unsafe moves!")
    
    for pos in path:
        yield pos


def _move_safely_env(env, start, goal, valid_positions):
    """Move safely from start to goal using environment validation"""
    if start == goal:
        return []
    
    path = []
    current = start
    
    # Try direct path first - horizontal then vertical
    # Move horizontally
    while current[0] != goal[0]:
        step = 1 if goal[0] > current[0] else -1
        next_pos = (current[0] + step, current[1])
        
        if next_pos not in valid_positions:
            break
        
        if not env.is_valid_move(current, next_pos):
            break
        
        path.append(next_pos)
        current = next_pos
    
    # Move vertically
    while current[1] != goal[1]:
        step = 1 if goal[1] > current[1] else -1
        next_pos = (current[0], current[1] + step)
        
        if next_pos not in valid_positions:
            break
        
        if not env.is_valid_move(current, next_pos):
            break
        
        path.append(next_pos)
        current = next_pos
    
    if not path or path[-1] != goal:
        # Couldn't reach directly, use A*
        return _find_safe_path_env(env, start, goal, valid_positions)
    
    return path


def _find_safe_path_env(env, start, goal, valid_positions):
    """Find safe path using A* with environment validation"""
    if start == goal:
        return []
    
    from collections import deque
    import heapq
    
    # A* search
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    
    valid_set = set(valid_positions)
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path
        
        # Check all neighbors using environment
        for neighbor in env.get_neighbors(current):
            if neighbor not in valid_set:
                continue
            
            tentative_g = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                
                # Manhattan distance heuristic
                h = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                f = tentative_g + h
                
                heapq.heappush(open_set, (f, neighbor))
    
    # No path found
    return []
