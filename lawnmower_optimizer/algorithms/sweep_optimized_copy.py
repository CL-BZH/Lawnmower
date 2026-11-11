"""
Rectangle Decomposition Coverage with Boustrophedon
- Decompose map into rectangles
- Merge rectangles that share full sides
- Split into connected components
- Sweep each component with efficient boustrophedon (skip rows when blade covers ±1)
- Use A* between components
"""

import numpy as np
from typing import Tuple, List, Generator, Set, Dict
from utils.environment import LawnmowerEnvironment
from algorithms.mst import _find_path_astar


def rectangle_decomposition_sweep(env: LawnmowerEnvironment) -> Generator[Tuple[int, int], None, None]:
    """
    Coverage using rectangle decomposition and boustrophedon sweep.
    
    ALGORITHM OVERVIEW:
    1. Decompose map into rectangles (from all_lawn_cells = 1752 cells)
    2. Merge rectangles into groups (100% full-side sharing, flood-fill)
    3. Extract mower positions from each group (only lawn_cells = 1317)
    4. Split groups into connected components
    5. Sweep each component with boustrophedon (with lane skipping)
    6. Use A* for transitions between components
    
    CRITICAL ISSUES TO FIX:
    - BUG 1: Lane skipping creates gaps - skipped cells not marked as covered
            When we skip row y=11 (because blade at y=10 covers it), those
            cells at y=11 are NEVER added to 'covered', so they go to mop-up!
            Result: 549 cells in mop-up instead of ~20
    
    - BUG 2: A* paths between segments within component not fully added
            When boustrophedon calls A* to connect segments, some positions
            in the A* path are not being added to the returned path
            Result: 3-4 invalid moves (jumps)
    
    - BUG 3: Lane skipping logic may be wrong
            Only 4% of moves are lane skips, should be much higher
            Current logic: if gap >= 2, sweep the row
            This means we sweep BOTH y=10 and y=12, not skipping y=11
    """
    print("Building rectangle decomposition coverage...")
    
    # Step 1: Create binary matrix from ALL lawn cells (not obstacles)
    # Uses all_lawn_cells (1752) which includes cells mower can't reach
    matrix = _create_binary_matrix(env)
    
    # Step 2: Decompose into rectangles using greedy largest-rectangle
    # Creates rectangles that cover all non-obstacle cells
    all_rectangles = _decompose_into_rectangles(matrix, env)
    print(f"Initial decomposition: {len(all_rectangles)} rectangles")
    
    # Step 3: Merge rectangles into groups
    # Uses flood-fill with 100% full-side sharing criterion
    # Returns: list of lists, each inner list has rectangle indices for one group
    merged_groups = _merge_adjacent_rectangles(all_rectangles, env)
    print(f"After merging: {len(merged_groups)} groups")
    
    # Step 4: For each group, get MOWER POSITIONS (lawn_cells) from rectangles
    # IMPORTANT: Rectangles were built from all_lawn_cells (1752) but we can
    # only visit positions in lawn_cells (1317) where mower can be placed
    all_components = []
    for group_idx, group_rect_ids in enumerate(merged_groups):
        # Get cells from ALL rectangles in this group (only mower positions)
        group_cells = set()
        for rect_id in group_rect_ids:
            rect = all_rectangles[rect_id]
            for dy in range(rect['height']):
                for dx in range(rect['width']):
                    pos = (int(rect['x'] + dx), int(rect['y'] + dy))
                    # FILTER: only include positions where mower can be placed
                    if pos in env.lawn_cells:  
                        group_cells.add(pos)
        
        # Split group into connected components (4-connectivity within group_cells)
        if group_cells:
            components = _find_connected_components(group_cells, env)
            print(f"  Group {group_idx}: {len(group_cells)} cells → {len(components)} components")
            all_components.extend(components)
    
    print(f"  Total: {len(all_components)} independent components")
    
    # Step 5: Sweep each component with boustrophedon, use A* between components
    current_pos = env.start_pos
    covered = set()
    
    remaining_components = list(range(len(all_components)))
    
    while remaining_components:
        # Find closest component
        def comp_dist(idx):
            return min(abs(p[0]-current_pos[0]) + abs(p[1]-current_pos[1]) 
                      for p in all_components[idx])
        
        closest_idx = min(remaining_components, key=comp_dist)
        remaining_components.remove(closest_idx)
        component = all_components[closest_idx]
        
        # Filter out already covered cells
        to_sweep = component - covered
        if not to_sweep:
            continue
        
        print(f"  Sweeping component {closest_idx}: {len(to_sweep)} cells")
        
        # A* to entry point
        entry = min(to_sweep, key=lambda p: abs(p[0]-current_pos[0]) + abs(p[1]-current_pos[1]))
        print(f"Entry: {entry}. Current pos: {current_pos}")

        path_to_entry = []
        if current_pos != entry:
            path_to_entry = _find_path_astar(env, current_pos, entry, env.lawn_cells)
            if path_to_entry and len(path_to_entry) > 1:
                for pos in path_to_entry[1:]:
                    covered.add(pos)
                    yield pos
                    current_pos = pos
        print(f"\tPath: {path_to_entry}")

        # Boustrophedon sweep this component
        sweep_path = _boustrophedon_component(to_sweep, current_pos, env)
        
        for pos in sweep_path:
            covered.add(pos)
            yield pos
            current_pos = pos
    
    # Mop up any remaining
    # remaining = set(env.lawn_cells) - covered
    # if remaining:
    #     print(f"  Mopping up {len(remaining)} cells")
    #     while remaining:
    #         nearest = min(remaining, key=lambda p: abs(p[0]-current_pos[0]) + abs(p[1]-current_pos[1]))
    #         path = _find_path_astar(env, current_pos, nearest, env.lawn_cells)
    #         if path and len(path) > 1:
    #             for pos in path[1:]:
    #                 covered.add(pos)
    #                 yield pos
    #                 current_pos = pos
    #                 remaining.discard(pos)
    #         else:
    #             remaining.remove(nearest)
    
    print(f"Completed: {len(covered)}/{len(env.lawn_cells)} cells")


def _boustrophedon_component(cells: Set[Tuple[int, int]], 
                             start: Tuple[int, int],
                             env: LawnmowerEnvironment) -> List[Tuple[int, int]]:
    """
    Efficient boustrophedon sweep of a connected component.
    
    LANE SKIPPING STRATEGY:
    - Blade diameter = 2 cells (covers ±1 from center)
    - When mower is at row y=10, blade covers y=9,10,11
    - So we can skip row y=11 and jump to y=12
    - Current logic: if gap >= 2 between rows, add both rows
      (This is WRONG - we're not actually skipping!)
    
    CURRENT BUGS:
    1. Lane skipping doesn't work - only 4% of moves skip lanes
    2. Skipped cells not marked as covered - causes huge mop-up
    3. A* paths between segments not fully added to output
    
    INPUT:
    - cells: Set of positions to sweep (from env.lawn_cells)
    - start: Starting position for sweep
    - env: Environment
    
    OUTPUT:
    - List of positions forming the sweep path
    - DOES NOT include skipped positions (they need to be marked elsewhere!)
    """
    if not cells:
        return []
    
    xs = [p[0] for p in cells]
    ys = [p[1] for p in cells]
    sweep_horizontal = (max(xs) - min(xs)) >= (max(ys) - min(ys))
    
    path = []
    covered = set()
    
    if sweep_horizontal:
        # Group by row
        rows = {}
        for pos in cells:
            if pos[1] not in rows:
                rows[pos[1]] = []
            rows[pos[1]].append(pos)
        
        sorted_rows = sorted(rows.keys())
        
        # Select rows to sweep (skip rows within blade radius of previous swept row)
        rows_to_sweep = []
        for y in sorted_rows:
            if len(rows_to_sweep) == 0:
                rows_to_sweep.append(y)
            else:
                prev_y = rows_to_sweep[-1]
                gap = y - prev_y
                if gap >= 2:
                    # Gap of 2+ means row y is NOT covered by blade at prev_y
                    rows_to_sweep.append(y)
                # if gap == 1, skip this row (blade at prev_y covers it)
        
        # Sweep selected rows
        for row_idx, y in enumerate(rows_to_sweep):
            row_cells = sorted(rows[y], key=lambda p: p[0])
            if row_idx % 2 == 1:
                row_cells.reverse()
            
            # A* to start of row
            if path and row_cells and row_cells[0] not in env.get_neighbors(path[-1]):
                trans = _find_path_astar(env, path[-1], row_cells[0], env.lawn_cells)
                if trans and len(trans) > 1:
                    for p in trans[1:]:
                        path.append(p)  # Add ALL positions, even outside component
                        if p in cells:
                            covered.add(p)
            
            # Sweep row
            for cell in row_cells:
                if cell in covered:
                    continue
                
                # A* if not adjacent
                if path and cell not in env.get_neighbors(path[-1]):
                    sub = _find_path_astar(env, path[-1], cell, env.lawn_cells)
                    if not sub or len(sub) <= 1:
                        # Can't reach - will be mopped up later
                        continue
                    for p in sub[1:]:
                        path.append(p)
                        if p in cells:
                            covered.add(p)
                
                if cell not in covered:
                    path.append(cell)
                    covered.add(cell)
    
    else:
        # Group by column
        cols = {}
        for pos in cells:
            if pos[0] not in cols:
                cols[pos[0]] = []
            cols[pos[0]].append(pos)
        
        sorted_cols = sorted(cols.keys())
        
        # Select columns to sweep (skip columns within blade radius)
        cols_to_sweep = []
        for x in sorted_cols:
            if len(cols_to_sweep) == 0:
                cols_to_sweep.append(x)
            else:
                prev_x = cols_to_sweep[-1]
                gap = x - prev_x
                if gap >= 2:
                    # Gap of 2+ means column x is NOT covered by blade at prev_x
                    cols_to_sweep.append(x)
                # if gap == 1, skip this column (blade at prev_x covers it)
        
        # Sweep selected columns
        for col_idx, x in enumerate(cols_to_sweep):
            col_cells = sorted(cols[x], key=lambda p: p[1])
            if col_idx % 2 == 1:
                col_cells.reverse()
            
            # A* to start of column
            if path and col_cells and col_cells[0] not in env.get_neighbors(path[-1]):
                trans = _find_path_astar(env, path[-1], col_cells[0], env.lawn_cells)
                if trans and len(trans) > 1:
                    for p in trans[1:]:
                        path.append(p)  # Add ALL positions
                        if p in cells:
                            covered.add(p)
            
            # Sweep column
            for cell in col_cells:
                if cell in covered:
                    continue
                
                # A* if not adjacent
                if path and cell not in env.get_neighbors(path[-1]):
                    sub = _find_path_astar(env, path[-1], cell, env.lawn_cells)
                    if not sub or len(sub) <= 1:
                        # Can't reach - will be mopped up later
                        continue
                    for p in sub[1:]:
                        path.append(p)
                        if p in cells:
                            covered.add(p)
                
                if cell not in covered:
                    path.append(cell)
                    covered.add(cell)
    
    return path


def _find_connected_components(cells: Set[Tuple[int, int]], 
                               env: LawnmowerEnvironment) -> List[Set[Tuple[int, int]]]:
    """
    Find connected components via BFS through env.lawn_cells.
    Only considers paths that stay within cells from the component.
    """
    remaining = set(cells)
    components = []
    
    while remaining:
        start = next(iter(remaining))
        component = {start}
        remaining.remove(start)
        queue = [start]
        
        while queue:
            current = queue.pop(0)
            # Check 4-connected neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in remaining:  # Only check if in remaining, not lawn_cells
                    component.add(neighbor)
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        
        components.append(component)
    
    return components


def _create_binary_matrix(env: LawnmowerEnvironment) -> np.ndarray:
    """Create binary matrix: 1 = lawn (not obstacle), 0 = obstacle"""
    matrix = np.zeros((env.height, env.width), dtype=int)
    for x, y in env.all_lawn_cells:  # ALL lawn cells from map
        matrix[y, x] = 1
    return matrix


def _decompose_into_rectangles(matrix: np.ndarray, env: LawnmowerEnvironment) -> List[Dict]:
    """Decompose map into rectangles (min 1x1 to cover everything)"""
    height, width = matrix.shape
    remaining = matrix.copy()
    rectangles = []
    
    while np.any(remaining == 1):
        largest = _find_largest_rectangle(remaining)
        if largest is None or largest['area'] < 1:  # Changed from 4 to 1
            break
        
        rectangles.append(largest)
        
        for dy in range(largest['height']):
            for dx in range(largest['width']):
                y, x = largest['y'] + dy, largest['x'] + dx
                if 0 <= y < height and 0 <= x < width:
                    remaining[y, x] = 0
    
    return rectangles


def _find_largest_rectangle(matrix: np.ndarray) -> Dict:
    """Find largest rectangle using histogram method"""
    if not np.any(matrix == 1):
        return None
    
    height, width = matrix.shape
    heights = np.zeros(width, dtype=int)
    best = None
    max_area = 0
    
    for y in range(height):
        for x in range(width):
            heights[x] = heights[x] + 1 if matrix[y, x] == 1 else 0
        
        stack = []
        for x in range(width):
            h = heights[x]
            start_x = x
            
            while stack and stack[-1][1] > h:
                prev_x, prev_h = stack.pop()
                area = prev_h * (x - prev_x)
                if area > max_area:
                    max_area = area
                    best = {'x': prev_x, 'y': y - prev_h + 1,
                           'width': x - prev_x, 'height': prev_h, 'area': area}
                start_x = prev_x
            
            if not stack or stack[-1][1] < h:
                stack.append((start_x, h))
        
        while stack:
            prev_x, prev_h = stack.pop()
            area = prev_h * (width - prev_x)
            if area > max_area:
                max_area = area
                best = {'x': prev_x, 'y': y - prev_h + 1,
                       'width': width - prev_x, 'height': prev_h, 'area': area}
    
    return best


def _merge_adjacent_rectangles(rectangles: List[Dict], env: LawnmowerEnvironment) -> List[Dict]:
    """
    Merge rectangles using flood-fill algorithm.
    A candidate joins a group if it shares 100% of one of ITS sides with any rectangle in the group.
    If a rectangle can join multiple groups, it joins the one with the biggest shared side.
    """
    if not rectangles:
        return []
    
    def shared_edge_length(rect1, rect2):
        """Calculate shared edge length between two rectangles (gap ≤ 1)"""
        # Horizontal adjacency
        y_overlap = max(0, min(rect1['y']+rect1['height'], rect2['y']+rect2['height']) - max(rect1['y'], rect2['y']))
        y_ratio = min(rect1['height'], rect2['height']) / max(rect1['height'], rect2['height'])
        if y_overlap > 0:
            x_gap = 0
            if rect1['x'] + rect1['width'] <= rect2['x']:
                x_gap = rect2['x'] - (rect1['x'] + rect1['width'])
            elif rect2['x'] + rect2['width'] <= rect1['x']:
                x_gap = rect1['x'] - (rect2['x'] + rect2['width'])
            if x_gap <= 1:
                return y_overlap, y_ratio 
        
        # Vertical adjacency
        x_overlap = max(0, min(rect1['x']+rect1['width'], rect2['x']+rect2['width']) - max(rect1['x'], rect2['x']))
        x_ratio = min(rect1['width'], rect2['width']) / max(rect1['width'], rect2['width'])
        if x_overlap > 0:
            y_gap = 0
            if rect1['y'] + rect1['height'] <= rect2['y']:
                y_gap = rect2['y'] - (rect1['y'] + rect1['height'])
            elif rect2['y'] + rect2['height'] <= rect1['y']:
                y_gap = rect1['y'] - (rect2['y'] + rect2['height'])
            if y_gap <= 1:
                return x_overlap, x_ratio 
        return 0, 0
    
    def shares_full_side_with_group(candidate_idx, group_indices):
        """Check if candidate shares 100% of one of its sides with any rectangle in group"""
        candidate = rectangles[candidate_idx]
        candidate_width = candidate['width']
        candidate_height = candidate['height']
        
        for group_idx in group_indices:
            edge_len, _ = shared_edge_length(candidate, rectangles[group_idx])
            # Check if this edge is 100% of one of candidate's sides
            if edge_len == candidate_width or edge_len == candidate_height:
                return True
        return False
    
    # Build groups using flood-fill
    groups = []
    links = {}
    used = set()
    
    for start_idx in range(len(rectangles)):
        if start_idx in used:
            continue
        
        # Start new group with this rectangle
        group = [start_idx]
        used.add(start_idx)
        queue = [start_idx]
        links[start_idx] = start_idx
        
        # Flood-fill: find all rectangles that share full side with group
        while queue:
            current_idx = queue.pop(0)
            
            for candidate_idx in range(len(rectangles)):
                if candidate_idx not in used:
                    if shares_full_side_with_group(candidate_idx, group):
                        group.append(candidate_idx)
                        used.add(candidate_idx)
                        links[candidate_idx] = current_idx
                        queue.append(candidate_idx)
                elif (candidate_idx != current_idx and 
                      candidate_idx != links[candidate_idx]):
                    boundery_len, ratio = shared_edge_length(rectangles[candidate_idx], rectangles[current_idx])
                    if boundery_len > 0:
                        #print(f"Rectangle R{candidate_idx} is already used but it could be moved to group of R{current_idx}")
                        # Find in which group is rectangles[candidate_idx]
                        linked_to = links[candidate_idx]
                        if linked_to !=  links[current_idx]:
                            prev_boundery_len, prev_ratio = shared_edge_length(rectangles[candidate_idx], rectangles[linked_to])
                            try:
                                prev_group = None
                                for group_x in groups:
                                    if candidate_idx in group_x:
                                        prev_group = group_x
                                        break
                                if prev_group is None:
                                    raise ValueError(f"Candidate {candidate_idx} not found in any group.\nGroups are {groups}")
                            except ValueError as e:
                                print(f"Error: {e}.")
                                exit()
                            # prev_boundery_len = 0 means that candidate_idx is linked to current_idx but not directly in contact with it
                            if prev_boundery_len > 0:
                                if ((boundery_len > prev_boundery_len) or
                                    ((boundery_len == prev_boundery_len) and (ratio > prev_ratio))):
                                    #print(f"Rectangle R{candidate_idx} is linked to R{linked_to} but should move to group of R{current_idx}")
                                    prev_group.remove(candidate_idx)
                                    group.append(candidate_idx)
                                    links[candidate_idx] = current_idx
                                    #queue.append(candidate_idx)
                                    print(f"Rectangle R{candidate_idx} was in group {prev_group} and" 
                                          f" moved to the group of rectangle {current_idx}: {group}")       
                            else:
                                # Find the rectangle that it is in contact with and check the bounderies ratio
                                best_prev_ratio = 0
                                best_prev_idx = None
                                for rectangle_idx in prev_group:
                                    if candidate_idx == rectangle_idx:
                                        continue
                                    prev_boundery_len, prev_ratio = shared_edge_length(rectangles[candidate_idx],
                                                                                       rectangles[rectangle_idx])
                                    if prev_ratio > best_prev_ratio:
                                        best_prev_ratio = prev_ratio
                                        best_prev_idx = rectangle_idx
                                if best_prev_ratio < ratio:
                                    print(f"Rectangle R{candidate_idx} is in group {prev_group} move to group {group}")
                                    prev_group.remove(candidate_idx)
                                    group.append(candidate_idx)
                                    links[candidate_idx] = current_idx
                                    #queue.append(candidate_idx)
        
        groups.append(group)
    
    # Return groups as lists of rectangle indices (not bounding boxes!)
    return groups


def _resolve_overlaps(rectangles: List[Dict], env: LawnmowerEnvironment) -> List[Dict]:
    """Resolve overlaps by shrinking, preserve rect_ids"""
    sorted_rects = sorted(rectangles, key=lambda r: r['area'], reverse=True)
    result = []
    occupied = set()
    
    for rect in sorted_rects:
        rect_cells = set()
        for dy in range(rect['height']):
            for dx in range(rect['width']):
                pos = (rect['x'] + dx, rect['y'] + dy)
                if pos in env.all_lawn_cells and pos not in occupied:  # Changed to all_lawn_cells
                    rect_cells.add(pos)
        
        if rect_cells:
            xs = [c[0] for c in rect_cells]
            ys = [c[1] for c in rect_cells]
            new_rect = {
                'x': min(xs), 'y': min(ys),
                'width': max(xs) - min(xs) + 1,
                'height': max(ys) - min(ys) + 1,
                'area': len(rect_cells),
                'rect_ids': rect.get('rect_ids', [])  # Preserve rect_ids
            }
            result.append(new_rect)
            occupied.update(rect_cells)
    
    return result
