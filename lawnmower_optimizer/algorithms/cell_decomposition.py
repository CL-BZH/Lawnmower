"""
Cell Decomposition Boustrophedon
Splits map into natural regions separated by obstacles, then sweeps each region

This is the "proper" Boustrophedon approach for complex environments
"""

from typing import Tuple, List, Generator, Set, Dict
from collections import defaultdict
from utils.environment import LawnmowerEnvironment


def cell_decomposition_boustrophedon(env: LawnmowerEnvironment) -> Generator[Tuple[int, int], None, None]:
    """
    Cell decomposition boustrophedon - splits map into convex regions
    """
    print("Building cell decomposition boustrophedon...")
    
    # Step 1: Decompose map into regions using connectivity
    regions = _decompose_by_connectivity(env)
    print(f"Decomposed into {len(regions)} regions")
    
    # Step 2: Order regions by proximity
    region_order = _order_regions_greedy(env, regions)
    
    # Step 3: Sweep each region
    current_pos = env.start_pos
    all_positions = []
    
    for region_id in region_order:
        region_positions = regions[region_id]
        
        # Sweep this region (starting from current position if it's in the region)
        if current_pos in region_positions:
            region_path = _sweep_region_boustrophedon(env, region_positions, current_pos)
        else:
            # Find entry point closest to current position
            entry_point = min(region_positions, 
                             key=lambda p: abs(p[0]-current_pos[0]) + abs(p[1]-current_pos[1]))
            
            # Only add entry point, don't path there yet
            # The sweep will naturally connect
            region_path = _sweep_region_boustrophedon(env, region_positions, entry_point)
        
        # Add region path
        all_positions.extend(region_path)
        
        if region_path:
            current_pos = region_path[-1]
    
    # Yield all positions
    for pos in all_positions:
        yield pos
    
    print(f"Cell decomposition completed: {len(set(all_positions))} unique positions visited")


def _decompose_by_connectivity(env: LawnmowerEnvironment) -> Dict[int, Set[Tuple[int, int]]]:
    """
    Decompose map into regions using connectivity analysis
    
    Strategy: Find connected components, then split large components
    by obstacle barriers (vertical or horizontal splits)
    """
    positions = set(env.lawn_cells)
    regions = {}
    region_id = 0
    
    # First pass: connected components
    visited_global = set()
    
    for start_pos in positions:
        if start_pos in visited_global:
            continue
        
        # BFS to find connected component
        component = set()
        queue = [start_pos]
        visited = {start_pos}
        
        while queue:
            pos = queue.pop(0)
            component.add(pos)
            visited_global.add(pos)
            
            for neighbor in env.get_neighbors(pos):
                if neighbor in positions and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # Split large components into sweep-friendly regions
        if len(component) > 50:
            # Split this component
            sub_regions = _split_component(env, component)
            for sub_region in sub_regions:
                if len(sub_region) >= 5:
                    regions[region_id] = sub_region
                    region_id += 1
        else:
            regions[region_id] = component
            region_id += 1
    
    return regions


def _split_component(env: LawnmowerEnvironment, 
                     component: Set[Tuple[int, int]]) -> List[Set[Tuple[int, int]]]:
    """
    Split a large component into smaller sweep-friendly regions
    
    Strategy: Find "critical lines" (vertical or horizontal) where splitting
    makes sense (e.g., narrow passages, obstacle barriers)
    """
    if len(component) < 50:
        return [component]
    
    # Get bounding box
    min_x = min(p[0] for p in component)
    max_x = max(p[0] for p in component)
    min_y = min(p[1] for p in component)
    max_y = max(p[1] for p in component)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Decide split direction
    split_vertically = width > height
    
    if split_vertically:
        # Find best vertical split line
        best_split_x = None
        min_connectivity = float('inf')
        
        # Try split lines in middle third
        for x in range(min_x + width//3, min_x + 2*width//3):
            # Count positions on this line
            on_line = sum(1 for p in component if p[0] == x)
            
            if on_line < min_connectivity:
                min_connectivity = on_line
                best_split_x = x
        
        if best_split_x and min_connectivity < 5:
            # Split at this line
            left = {p for p in component if p[0] < best_split_x}
            right = {p for p in component if p[0] >= best_split_x}
            
            # Recursively split if still large
            result = []
            if len(left) > 50:
                result.extend(_split_component(env, left))
            else:
                result.append(left)
            
            if len(right) > 50:
                result.extend(_split_component(env, right))
            else:
                result.append(right)
            
            return [r for r in result if r]
    
    else:
        # Find best horizontal split line
        best_split_y = None
        min_connectivity = float('inf')
        
        for y in range(min_y + height//3, min_y + 2*height//3):
            on_line = sum(1 for p in component if p[1] == y)
            
            if on_line < min_connectivity:
                min_connectivity = on_line
                best_split_y = y
        
        if best_split_y and min_connectivity < 5:
            top = {p for p in component if p[1] < best_split_y}
            bottom = {p for p in component if p[1] >= best_split_y}
            
            result = []
            if len(top) > 50:
                result.extend(_split_component(env, top))
            else:
                result.append(top)
            
            if len(bottom) > 50:
                result.extend(_split_component(env, bottom))
            else:
                result.append(bottom)
            
            return [r for r in result if r]
    
    # Can't split effectively, return as is
    return [component]


def _order_regions_greedy(env: LawnmowerEnvironment,
                         regions: Dict[int, Set[Tuple[int, int]]]) -> List[int]:
    """Order regions by nearest neighbor"""
    if not regions:
        return []
    
    # Start with region closest to start position
    centroids = {rid: _centroid(reg) for rid, reg in regions.items()}
    
    current = min(regions.keys(),
                 key=lambda rid: abs(centroids[rid][0] - env.start_pos[0]) + 
                                abs(centroids[rid][1] - env.start_pos[1]))
    
    order = [current]
    remaining = set(regions.keys()) - {current}
    
    while remaining:
        # Find nearest unvisited region
        next_region = min(remaining,
                         key=lambda rid: abs(centroids[rid][0] - centroids[current][0]) +
                                        abs(centroids[rid][1] - centroids[current][1]))
        order.append(next_region)
        remaining.remove(next_region)
        current = next_region
    
    return order


def _sweep_region_boustrophedon(env: LawnmowerEnvironment,
                                region: Set[Tuple[int, int]],
                                start: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Sweep a region with boustrophedon pattern
    """
    if not region:
        return []
    
    # Determine sweep direction
    min_x = min(p[0] for p in region)
    max_x = max(p[0] for p in region)
    min_y = min(p[1] for p in region)
    max_y = max(p[1] for p in region)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Sweep in the longer direction
    sweep_horizontal = width >= height
    
    path = []
    visited = set()
    
    if sweep_horizontal:
        # Horizontal sweep - go row by row
        # Group by rows
        rows = defaultdict(list)
        for pos in region:
            rows[pos[1]].append(pos)
        
        # Sort rows
        row_ys = sorted(rows.keys())
        
        for i, y in enumerate(row_ys):
            # Alternate direction
            row_positions = sorted(rows[y], key=lambda p: p[0], reverse=(i % 2 == 1))
            
            for pos in row_positions:
                if pos not in visited:
                    path.append(pos)
                    visited.add(pos)
    
    else:
        # Vertical sweep - go column by column
        cols = defaultdict(list)
        for pos in region:
            cols[pos[0]].append(pos)
        
        col_xs = sorted(cols.keys())
        
        for i, x in enumerate(col_xs):
            col_positions = sorted(cols[x], key=lambda p: p[1], reverse=(i % 2 == 1))
            
            for pos in col_positions:
                if pos not in visited:
                    path.append(pos)
                    visited.add(pos)
    
    return path


def _simple_path(env: LawnmowerEnvironment, 
                start: Tuple[int, int], 
                goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Simple pathfinding between regions"""
    if start == goal:
        return []
    
    # Try direct path first
    path = [start]
    current = start
    
    # Move horizontally then vertically
    while current[0] != goal[0]:
        step = 1 if goal[0] > current[0] else -1
        next_pos = (current[0] + step, current[1])
        
        if next_pos in env.lawn_cells and env.is_valid_move(current, next_pos):
            path.append(next_pos)
            current = next_pos
        else:
            break
    
    while current[1] != goal[1]:
        step = 1 if goal[1] > current[1] else -1
        next_pos = (current[0], current[1] + step)
        
        if next_pos in env.lawn_cells and env.is_valid_move(current, next_pos):
            path.append(next_pos)
            current = next_pos
        else:
            break
    
    # If direct path failed, try A*
    if current != goal:
        from algorithms.mst import _find_path_astar
        astar_path = _find_path_astar(env, start, goal, env.lawn_cells)
        if astar_path:
            return astar_path
    
    return path


def _centroid(positions: Set[Tuple[int, int]]) -> Tuple[int, int]:
    """Calculate centroid of positions"""
    if not positions:
        return (0, 0)
    x_sum = sum(p[0] for p in positions)
    y_sum = sum(p[1] for p in positions)
    n = len(positions)
    return (x_sum // n, y_sum // n)
