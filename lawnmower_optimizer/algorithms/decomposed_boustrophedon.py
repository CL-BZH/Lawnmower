"""
Decomposed Boustrophedon Coverage
Combines the best of MST (region decomposition) and Boustrophedon (systematic sweeping)

Strategy:
1. Decompose map into convex/simple regions
2. Apply boustrophedon sweeping within each region  
3. Use MST to order regions and connect them efficiently
"""

import heapq
from typing import Tuple, List, Generator, Set, Dict
from collections import defaultdict
from utils.environment import LawnmowerEnvironment


def decomposed_boustrophedon(env: LawnmowerEnvironment) -> Generator[Tuple[int, int], None, None]:
    """
    Decomposed boustrophedon coverage algorithm
    """
    print("Building decomposed boustrophedon coverage...")
    
    # Step 1: Decompose map into regions
    regions = _decompose_into_regions(env)
    print(f"Decomposed map into {len(regions)} regions")
    
    # Step 2: Build MST over region centroids to determine visit order
    region_order = _order_regions_mst(env, regions)
    print(f"Region visit order determined")
    
    # Step 3: Sweep each region in order
    state = env.create_state()
    yield state.position
    
    for region_id in region_order:
        region_positions = regions[region_id]
        
        # Sweep this region
        region_path = _sweep_region(env, state.position, region_positions)
        
        # Execute path
        for pos in region_path:
            if pos != state.position and env.is_valid_move(state.position, pos):
                state = env.move(state, pos)
                yield state.position
            elif pos != state.position:
                # Need pathfinding to reach
                from algorithms.mst import _find_path_astar
                subpath = _find_path_astar(env, state.position, pos, env.lawn_cells)
                if subpath and len(subpath) > 1:
                    for p in subpath[1:]:
                        state = env.move(state, p)
                        yield state.position
    
    print(f"Decomposed boustrophedon completed: {len(state.mowed_cells)}/{len(env.reachable_lawn_cells)} cells")
    print(f"Efficiency: {env.calculate_efficiency(state):.1f}%")


def _decompose_into_regions(env: LawnmowerEnvironment) -> Dict[int, Set[Tuple[int, int]]]:
    """
    Decompose map into regions based on connectivity and geometry
    
    Uses a simple approach:
    - Group positions by vertical slices
    - Within each slice, group by horizontal connectivity
    """
    positions = list(env.lawn_cells)
    
    # Group by vertical slices (every N columns)
    slice_width = 10  # Adjustable parameter
    
    regions = {}
    region_id = 0
    
    # Get bounding box
    min_x = min(p[0] for p in positions)
    max_x = max(p[0] for p in positions)
    min_y = min(p[1] for p in positions)
    max_y = max(p[1] for p in positions)
    
    # Create vertical slices
    for slice_x in range(min_x, max_x + 1, slice_width):
        slice_end_x = min(slice_x + slice_width, max_x + 1)
        
        # Get positions in this slice
        slice_positions = [p for p in positions if slice_x <= p[0] < slice_end_x]
        
        if not slice_positions:
            continue
        
        # Further decompose slice by connected components
        visited = set()
        
        for pos in slice_positions:
            if pos in visited:
                continue
            
            # BFS to find connected component
            component = set()
            queue = [pos]
            visited.add(pos)
            component.add(pos)
            
            while queue:
                curr = queue.pop(0)
                
                # Check neighbors within slice
                for neighbor in env.get_neighbors(curr):
                    if neighbor in slice_positions and neighbor not in visited:
                        visited.add(neighbor)
                        component.add(neighbor)
                        queue.append(neighbor)
            
            # Add as region if big enough
            if len(component) >= 3:
                regions[region_id] = component
                region_id += 1
    
    # Merge small regions into nearest larger region
    large_regions = {rid: reg for rid, reg in regions.items() if len(reg) >= 10}
    small_regions = {rid: reg for rid, reg in regions.items() if len(reg) < 10}
    
    for small_id, small_reg in small_regions.items():
        if not small_reg:
            continue
        
        # Find nearest large region
        small_center = _get_centroid(small_reg)
        
        best_large = None
        min_dist = float('inf')
        
        for large_id, large_reg in large_regions.items():
            large_center = _get_centroid(large_reg)
            dist = abs(small_center[0] - large_center[0]) + abs(small_center[1] - large_center[1])
            
            if dist < min_dist:
                min_dist = dist
                best_large = large_id
        
        if best_large is not None:
            large_regions[best_large] |= small_reg
    
    return large_regions if large_regions else regions


def _order_regions_mst(env: LawnmowerEnvironment, 
                       regions: Dict[int, Set[Tuple[int, int]]]) -> List[int]:
    """
    Order regions using MST over region centroids
    """
    if len(regions) <= 1:
        return list(regions.keys())
    
    # Calculate centroids
    centroids = {rid: _get_centroid(reg) for rid, reg in regions.items()}
    
    # Build MST over centroids
    start_region = min(regions.keys(), 
                       key=lambda rid: abs(centroids[rid][0] - env.start_pos[0]) + 
                                      abs(centroids[rid][1] - env.start_pos[1]))
    
    visited = set([start_region])
    order = [start_region]
    
    # Greedy MST construction
    while len(visited) < len(regions):
        min_dist = float('inf')
        next_region = None
        
        for visited_rid in visited:
            for candidate_rid in regions:
                if candidate_rid in visited:
                    continue
                
                dist = abs(centroids[visited_rid][0] - centroids[candidate_rid][0]) + \
                       abs(centroids[visited_rid][1] - centroids[candidate_rid][1])
                
                if dist < min_dist:
                    min_dist = dist
                    next_region = candidate_rid
        
        if next_region is not None:
            visited.add(next_region)
            order.append(next_region)
        else:
            break
    
    return order


def _sweep_region(env: LawnmowerEnvironment,
                  start_pos: Tuple[int, int],
                  region: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Sweep a region using boustrophedon pattern
    """
    if not region:
        return []
    
    # Get region bounding box
    min_x = min(p[0] for p in region)
    max_x = max(p[0] for p in region)
    min_y = min(p[1] for p in region)
    max_y = max(p[1] for p in region)
    
    # Decide sweep direction based on aspect ratio
    width = max_x - min_x
    height = max_y - min_y
    
    sweep_horizontal = width >= height
    
    path = []
    
    if sweep_horizontal:
        # Horizontal sweep (row by row)
        # Sweep every 2 rows since blade covers +/-1 in perpendicular direction
        visited_rows = set()
        y = min_y
        direction = 1  # 1 = left-to-right, -1 = right-to-left
        
        while y <= max_y:
            if y in visited_rows:
                y += 1
                continue
            
            row_positions = sorted([p for p in region if p[1] == y], 
                                  key=lambda p: p[0], 
                                  reverse=(direction == -1))
            
            if row_positions:
                for pos in row_positions:
                    if not path or pos != path[-1]:
                        path.append(pos)
                
                visited_rows.add(y)
                direction *= -1  # Alternate direction
                
                # Skip next row since blade covers it
                y += 2
            else:
                y += 1
        
        # Second pass: catch any missed rows
        for y in range(min_y, max_y + 1):
            if y not in visited_rows:
                row_positions = sorted([p for p in region if p[1] == y], 
                                      key=lambda p: p[0])
                for pos in row_positions:
                    if not path or pos != path[-1]:
                        path.append(pos)
                        
    else:
        # Vertical sweep (column by column)
        visited_cols = set()
        x = min_x
        direction = 1
        
        while x <= max_x:
            if x in visited_cols:
                x += 1
                continue
            
            col_positions = sorted([p for p in region if p[0] == x],
                                  key=lambda p: p[1],
                                  reverse=(direction == -1))
            
            if col_positions:
                for pos in col_positions:
                    if not path or pos != path[-1]:
                        path.append(pos)
                
                visited_cols.add(x)
                direction *= -1
                
                # Skip next column since blade covers it
                x += 2
            else:
                x += 1
        
        # Second pass: catch any missed columns
        for x in range(min_x, max_x + 1):
            if x not in visited_cols:
                col_positions = sorted([p for p in region if p[0] == x],
                                      key=lambda p: p[1])
                for pos in col_positions:
                    if not path or pos != path[-1]:
                        path.append(pos)
    
    return path


def _get_centroid(positions: Set[Tuple[int, int]]) -> Tuple[int, int]:
    """Calculate centroid of a set of positions"""
    if not positions:
        return (0, 0)
    
    x_sum = sum(p[0] for p in positions)
    y_sum = sum(p[1] for p in positions)
    n = len(positions)
    
    return (x_sum // n, y_sum // n)
