"""
Rectangle Decomposition Coverage with Boustrophedon - ADAPTIVE LANE SKIPPING
- Uses blade width (±1 cell) to skip every other lane when possible
- Adaptive to geometry and entry points
- Maintains strict safety checks
"""

import numpy as np
import heapq
from typing import Tuple, List, Generator, Set, Dict
from utils.environment import LawnmowerEnvironment
from algorithms.mst import _find_path_astar


def rectangle_decomposition_sweep(env: LawnmowerEnvironment, enable_mop_up: bool = False) -> Generator[Tuple[int, int], None, None]:
    """
    Coverage using rectangle decomposition with ADAPTIVE lane skipping
    
    KEY IMPROVEMENT: Uses blade width (±1 cell) to skip lanes
    """
    print("Building rectangle decomposition coverage with adaptive lane skipping...")
    
    # Step 1: Create binary matrix from ALL lawn cells (ORIGINAL)
    matrix = _create_binary_matrix(env)
    
    # Step 2: Decompose into rectangles using greedy largest-rectangle (ORIGINAL)
    all_rectangles = _decompose_into_rectangles(matrix, env)
    print(f"Initial decomposition: {len(all_rectangles)} rectangles")
    
    # Step 3: Merge rectangles into groups (ORIGINAL)
    merged_groups = _merge_adjacent_rectangles(all_rectangles, env)
    print(f"After merging: {len(merged_groups)} groups")
    
    # Step 4: For each group, get MOWER POSITIONS (lawn_cells) from rectangles (ORIGINAL)
    all_components = []
    for group_idx, group_rect_ids in enumerate(merged_groups):
        group_cells = set()
        for rect_id in group_rect_ids:
            rect = all_rectangles[rect_id]
            for dy in range(rect['height']):
                for dx in range(rect['width']):
                    pos = (int(rect['x'] + dx), int(rect['y'] + dy))
                    if pos in env.lawn_cells: 
                        group_cells.add(pos)
        
        # Split group into connected components (8-connectivity like original)
        if group_cells:
            components = _find_connected_components(group_cells, env)
            print(f"  Group {group_idx}: {len(group_cells)} cells → {len(components)} components")
            all_components.extend(components)
    
    print(f"  Total: {len(all_components)} independent components")
    
    # Step 5: Sweep each component with ADAPTIVE lane skipping
    current_pos = env.start_pos
    covered = set()  # Track visited positions
    mowed_cells = set()  # Track actual mowed cells (by blade coverage)
    
    # Start by mowing the start position
    start_blade = env.get_blade_coverage(current_pos)
    for cell in start_blade:
        if cell in env.reachable_lawn_cells:
            mowed_cells.add(cell)
    yield current_pos
    
    remaining_components = list(range(len(all_components)))
    
    while remaining_components:
        # Find closest component
        def comp_dist(idx):
            return min(abs(p[0]-current_pos[0]) + abs(p[1]-current_pos[1]) 
                      for p in all_components[idx])
        
        closest_idx = min(remaining_components, key=comp_dist)
        remaining_components.remove(closest_idx)
        component = all_components[closest_idx]
        
        print(f"  Sweeping component {closest_idx}: {len(component)} cells")
        
        # A* to entry point with STRICT SAFETY path
        entry = min(component, key=lambda p: abs(p[0]-current_pos[0]) + abs(p[1]-current_pos[1]))
        
        # Move to entry point with COMPLETE strict safe path
        if current_pos != entry:
            safe_path = _find_complete_strict_safe_path(env, current_pos, entry)
            if safe_path and len(safe_path) > 1:
                for i in range(1, len(safe_path)):
                    next_pos = safe_path[i]
                    # Update coverage
                    blade_cells = env.get_blade_coverage(next_pos)
                    for cell in blade_cells:
                        if cell in env.reachable_lawn_cells:
                            mowed_cells.add(cell)
                    covered.add(next_pos)
                    yield next_pos
                    current_pos = next_pos
        
        # ADAPTIVE lane skipping boustrophedon sweep
        print(f"DEBUG: Calling boustrophedon with current_pos={current_pos}")
        sweep_path = _adaptive_boustrophedon_sweep(component, current_pos, env, mowed_cells)
        
        for pos in sweep_path:
            if pos != current_pos:
                # Update coverage
                blade_cells = env.get_blade_coverage(pos)
                for cell in blade_cells:
                    if cell in env.reachable_lawn_cells:
                        mowed_cells.add(cell)
                covered.add(pos)
                yield pos
                current_pos = pos
    
    # Coverage report
    total_reachable = len(env.reachable_lawn_cells)
    coverage = 100 * len(mowed_cells) / total_reachable if total_reachable > 0 else 0
    print(f"Completed: {len(mowed_cells)}/{total_reachable} cells mowed ({coverage:.1f}%)")
    
    # Simple mop-up only if enabled AND coverage is low
    if enable_mop_up and coverage < 99.5:
        print(f"  Starting mop-up for {total_reachable - len(mowed_cells)} unmowed cells")
        # Call mop-up as a separate generator
        for pos in _strict_safety_mop_up_generator(env, current_pos, mowed_cells, covered):
            yield pos


def _adaptive_boustrophedon_sweep(cells: Set[Tuple[int, int]], 
                                  start: Tuple[int, int],
                                  env: LawnmowerEnvironment,
                                  mowed_cells: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Adaptive boustrophedon sweep with lane skipping
    Uses blade width (±1 cell) to skip every other lane when possible
    """
    if not cells:
        return []
    
    # Determine sweep direction based on component shape
    xs = [p[0] for p in cells]
    ys = [p[1] for p in cells]
    sweep_horizontal = (max(xs) - min(xs)) >= (max(ys) - min(ys))
    
    path = []
    current = start
    
    if sweep_horizontal:
        # Group by row for horizontal sweep
        rows = {}
        for pos in cells:
            rows.setdefault(pos[1], []).append(pos)
        
        # Sort rows
        sorted_rows = sorted(rows.items())
        
        # ADAPTIVE: Select which rows to sweep based on blade coverage
        rows_to_sweep = _select_rows_to_sweep(sorted_rows, start[1], env)
        
        # OPTIMIZATION: Reorder rows to start from closest to current position
        if rows_to_sweep:
            closest_idx = min(range(len(rows_to_sweep)),
                            key=lambda i: abs(rows_to_sweep[i][0] - start[1]))
            # Reorder: start from closest, alternate up/down
            reordered = [rows_to_sweep[closest_idx]]
            dist = 1
            while len(reordered) < len(rows_to_sweep):
                # Try down first (increasing y)
                if closest_idx + dist < len(rows_to_sweep):
                    reordered.append(rows_to_sweep[closest_idx + dist])
                # Then up (decreasing y)  
                if closest_idx - dist >= 0:
                    reordered.append(rows_to_sweep[closest_idx - dist])
                dist += 1
            rows_to_sweep = reordered
        
        print(f"DEBUG: Start y={start[1]}, Rows selected: {[y for y, _ in rows_to_sweep]}")
        
        # Sweep selected rows
        for row_idx, (y, row_cells) in enumerate(rows_to_sweep):
            row_cells.sort(key=lambda p: p[0])
            
            # Determine sweep direction (alternating)
            if row_idx % 2 == 0:
                # Left to right
                target_cell = row_cells[0]
                cell_order = row_cells
            else:
                # Right to left
                target_cell = row_cells[-1]
                cell_order = list(reversed(row_cells))
            
            # Move to start of row
            if current != target_cell:
                safe_path = _find_complete_strict_safe_path(env, current, target_cell)
                if safe_path and len(safe_path) > 1:
                    for i in range(1, len(safe_path)):
                        path.append(safe_path[i])
                        current = safe_path[i]
            
            # Sweep the row
            for cell in cell_order:
                if cell == current:
                    continue
                
                if _is_strictly_safe_move(env, current, cell):
                    path.append(cell)
                    current = cell
                else:
                    safe_path = _find_complete_strict_safe_path(env, current, cell)
                    if safe_path and len(safe_path) > 1:
                        for i in range(1, len(safe_path)):
                            path.append(safe_path[i])
                            current = safe_path[i]
        
        # Mark cells covered by skipped rows
        _mark_skipped_row_coverage(sorted_rows, rows_to_sweep, env, mowed_cells)
    
    else:
        # Vertical sweep - group by column
        cols = {}
        for pos in cells:
            cols.setdefault(pos[0], []).append(pos)
        
        sorted_cols = sorted(cols.items())
        
        # ADAPTIVE: Select which columns to sweep
        cols_to_sweep = _select_columns_to_sweep(sorted_cols, start[0], env)
        
        # OPTIMIZATION: Reorder columns to start from closest to current position
        if cols_to_sweep:
            closest_idx = min(range(len(cols_to_sweep)),
                            key=lambda i: abs(cols_to_sweep[i][0] - start[0]))
            # Reorder: start from closest, alternate left/right
            reordered = [cols_to_sweep[closest_idx]]
            dist = 1
            while len(reordered) < len(cols_to_sweep):
                # Try right first (increasing x)
                if closest_idx + dist < len(cols_to_sweep):
                    reordered.append(cols_to_sweep[closest_idx + dist])
                # Then left (decreasing x)
                if closest_idx - dist >= 0:
                    reordered.append(cols_to_sweep[closest_idx - dist])
                dist += 1
            cols_to_sweep = reordered
        
        # Sweep selected columns
        for col_idx, (x, col_cells) in enumerate(cols_to_sweep):
            col_cells.sort(key=lambda p: p[1])
            
            if col_idx % 2 == 0:
                # Top to bottom
                target_cell = col_cells[0]
                cell_order = col_cells
            else:
                # Bottom to top
                target_cell = col_cells[-1]
                cell_order = list(reversed(col_cells))
            
            # Move to start of column
            if current != target_cell:
                safe_path = _find_complete_strict_safe_path(env, current, target_cell)
                if safe_path and len(safe_path) > 1:
                    for i in range(1, len(safe_path)):
                        path.append(safe_path[i])
                        current = safe_path[i]
            
            # Sweep the column
            for cell in cell_order:
                if cell == current:
                    continue
                
                if _is_strictly_safe_move(env, current, cell):
                    path.append(cell)
                    current = cell
                else:
                    safe_path = _find_complete_strict_safe_path(env, current, cell)
                    if safe_path and len(safe_path) > 1:
                        for i in range(1, len(safe_path)):
                            path.append(safe_path[i])
                            current = safe_path[i]
        
        # Mark cells covered by skipped columns
        _mark_skipped_column_coverage(sorted_cols, cols_to_sweep, env, mowed_cells)
    
    return path


def _select_rows_to_sweep(sorted_rows: List[Tuple[int, List[Tuple[int, int]]]], 
                          start_y: int,
                          env: LawnmowerEnvironment) -> List[Tuple[int, List[Tuple[int, int]]]]:
    """
    Select which rows to sweep based on adaptive lane skipping
    Returns list of (y, row_cells) to actually sweep
    """
    if len(sorted_rows) <= 3:
        # For small components, sweep all rows
        return sorted_rows
    
    # Find the row closest to start_y
    closest_idx = min(range(len(sorted_rows)), 
                     key=lambda i: abs(sorted_rows[i][0] - start_y))
    
    rows_to_sweep = []
    
    # Start from the closest row and use skipping pattern
    # Blade covers ±1 cell, so we can skip every other row
    
    # First, add the closest row
    rows_to_sweep.append(sorted_rows[closest_idx])
    
    # Sweep forward (increasing y)
    i = closest_idx + 2  # Skip next row (covered by blade)
    while i < len(sorted_rows):
        rows_to_sweep.append(sorted_rows[i])
        i += 2  # Skip every other row
    
    # Sweep backward (decreasing y)
    i = closest_idx - 2  # Skip previous row (covered by blade)
    while i >= 0:
        rows_to_sweep.insert(0, sorted_rows[i])  # Insert at beginning
        i -= 2  # Skip every other row
    
    # Sort by y (for consistent ordering)
    rows_to_sweep.sort(key=lambda x: x[0])
    
    # ADAPTIVE CHECK: Ensure coverage at edges
    # Always include first and last rows to guarantee edge coverage
    first_y = sorted_rows[0][0]
    last_y = sorted_rows[-1][0]
    
    # Add first row if not already included
    if sorted_rows[0] not in rows_to_sweep:
        rows_to_sweep.insert(0, sorted_rows[0])
    
    # Add last row if not already included
    if sorted_rows[-1] not in rows_to_sweep:
        rows_to_sweep.append(sorted_rows[-1])
    
    # ADAPTIVE CHECK: In narrow areas, sweep every row
    # If any row has cells adjacent to obstacles on both sides, we might need to sweep it
    for y, row_cells in sorted_rows:
        if (y, row_cells) in rows_to_sweep:
            continue
        
        # Check if this row might have coverage gaps
        needs_sweeping = False
        
        # Check each cell in the skipped row
        for cell in row_cells:
            x, y_cell = cell
            
            # Check if cell might not be covered by adjacent swept rows
            covered = False
            
            # Check rows that will be swept
            for sweep_y, sweep_cells in rows_to_sweep:
                if abs(sweep_y - y_cell) > 1:
                    continue  # Too far to cover
                
                # Check if any cell in the swept row covers this cell
                for sweep_cell in sweep_cells:
                    blade = env.get_blade_coverage(sweep_cell)
                    if cell in blade:
                        covered = True
                        break
                
                if covered:
                    break
            
            if not covered:
                needs_sweeping = True
                break
        
        if needs_sweeping:
            # Insert this row in the correct position
            insert_idx = next(i for i, (ry, _) in enumerate(rows_to_sweep) if ry > y)
            rows_to_sweep.insert(insert_idx, (y, row_cells))
    
    return rows_to_sweep


def _select_columns_to_sweep(sorted_cols: List[Tuple[int, List[Tuple[int, int]]]], 
                             start_x: int,
                             env: LawnmowerEnvironment) -> List[Tuple[int, List[Tuple[int, int]]]]:
    """
    Select which columns to sweep based on adaptive lane skipping
    """
    if len(sorted_cols) <= 3:
        return sorted_cols
    
    # Find the column closest to start_x
    closest_idx = min(range(len(sorted_cols)), 
                     key=lambda i: abs(sorted_cols[i][0] - start_x))
    
    cols_to_sweep = []
    
    # Start from the closest column
    cols_to_sweep.append(sorted_cols[closest_idx])
    
    # Sweep forward (increasing x)
    i = closest_idx + 2
    while i < len(sorted_cols):
        cols_to_sweep.append(sorted_cols[i])
        i += 2
    
    # Sweep backward (decreasing x)
    i = closest_idx - 2
    while i >= 0:
        cols_to_sweep.insert(0, sorted_cols[i])
        i -= 2
    
    # Sort by x
    cols_to_sweep.sort(key=lambda x: x[0])
    
    # Ensure coverage at edges
    # Always include first and last columns to guarantee edge coverage
    first_x = sorted_cols[0][0]
    last_x = sorted_cols[-1][0]
    
    # Add first column if not already included
    if sorted_cols[0] not in cols_to_sweep:
        cols_to_sweep.insert(0, sorted_cols[0])
    
    # Add last column if not already included
    if sorted_cols[-1] not in cols_to_sweep:
        cols_to_sweep.append(sorted_cols[-1])
    
    # Check for coverage gaps in narrow areas
    for x, col_cells in sorted_cols:
        if (x, col_cells) in cols_to_sweep:
            continue
        
        needs_sweeping = False
        
        for cell in col_cells:
            x_cell, y = cell
            
            covered = False
            
            for sweep_x, sweep_cells in cols_to_sweep:
                if abs(sweep_x - x_cell) > 1:
                    continue
                
                for sweep_cell in sweep_cells:
                    blade = env.get_blade_coverage(sweep_cell)
                    if cell in blade:
                        covered = True
                        break
                
                if covered:
                    break
            
            if not covered:
                needs_sweeping = True
                break
        
        if needs_sweeping:
            insert_idx = next(i for i, (cx, _) in enumerate(cols_to_sweep) if cx > x)
            cols_to_sweep.insert(insert_idx, (x, col_cells))
    
    return cols_to_sweep


def _mark_skipped_row_coverage(all_rows: List[Tuple[int, List[Tuple[int, int]]]], 
                               swept_rows: List[Tuple[int, List[Tuple[int, int]]]],
                               env: LawnmowerEnvironment,
                               mowed_cells: Set[Tuple[int, int]]) -> None:
    """
    Mark cells in skipped rows as mowed if they're covered by blade from swept rows
    """
    for y, row_cells in all_rows:
        if (y, row_cells) in swept_rows:
            continue  # Already swept
        
        # Check each cell in the skipped row
        for cell in row_cells:
            if cell in mowed_cells:
                continue
            
            # Check if covered by any swept row
            for sweep_y, sweep_cells in swept_rows:
                if abs(sweep_y - y) > 1:
                    continue  # Too far to cover
                
                # Check if any cell in the swept row covers this cell
                for sweep_cell in sweep_cells:
                    blade = env.get_blade_coverage(sweep_cell)
                    if cell in blade and cell in env.reachable_lawn_cells:
                        mowed_cells.add(cell)
                        break


def _mark_skipped_column_coverage(all_cols: List[Tuple[int, List[Tuple[int, int]]]], 
                                  swept_cols: List[Tuple[int, List[Tuple[int, int]]]],
                                  env: LawnmowerEnvironment,
                                  mowed_cells: Set[Tuple[int, int]]) -> None:
    """
    Mark cells in skipped columns as mowed if they're covered by blade from swept columns
    """
    for x, col_cells in all_cols:
        if (x, col_cells) in swept_cols:
            continue
        
        for cell in col_cells:
            if cell in mowed_cells:
                continue
            
            for sweep_x, sweep_cells in swept_cols:
                if abs(sweep_x - x) > 1:
                    continue
                
                for sweep_cell in sweep_cells:
                    blade = env.get_blade_coverage(sweep_cell)
                    if cell in blade and cell in env.reachable_lawn_cells:
                        mowed_cells.add(cell)
                        break


def _is_strictly_safe_move(env: LawnmowerEnvironment, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> bool:
    """
    STRICT safety check for ANY move
    Returns True ONLY if:
    1. The move is valid
    2. Blade at from_pos doesn't touch obstacles
    3. Blade at to_pos doesn't touch obstacles
    4. For diagonal moves: intermediate cells are also safe
    """
    # Basic validity check
    if not env.is_valid_move(from_pos, to_pos):
        return False
    
    # Check blade coverage at from_pos doesn't touch obstacles
    from_blade = env.get_blade_coverage(from_pos)
    for cell in from_blade:
        if cell in env.obstacle_cells:
            return False
    
    # Check blade coverage at to_pos doesn't touch obstacles
    to_blade = env.get_blade_coverage(to_pos)
    for cell in to_blade:
        if cell in env.obstacle_cells:
            return False
    
    # For diagonal moves, check intermediate cells
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    
    if dx != 0 and dy != 0:
        # Check the cells that are physically crossed
        intermediate_cells = [
            (from_pos[0] + dx, from_pos[1]),  # Horizontal step
            (from_pos[0], from_pos[1] + dy),  # Vertical step
        ]
        
        for cell in intermediate_cells:
            if not env.is_valid_position(cell):
                return False
            
            blade = env.get_blade_coverage(cell)
            for blade_cell in blade:
                if blade_cell in env.obstacle_cells:
                    return False
    
    # Additional safety: check cells adjacent to the path
    path_cells = {from_pos, to_pos}
    if dx != 0 and dy != 0:
        path_cells.add((from_pos[0] + dx, from_pos[1]))
        path_cells.add((from_pos[0], from_pos[1] + dy))
    
    for cell in path_cells:
        # Check all adjacent cells for obstacles
        for adj_dx, adj_dy in [(1, 0), (-1, 0), (0, 1), (0, -1), 
                              (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            adj_cell = (cell[0] + adj_dx, cell[1] + adj_dy)
            if adj_cell in env.obstacle_cells:
                # If obstacle is diagonally adjacent, blade corner might hit it
                if abs(adj_dx) == 1 and abs(adj_dy) == 1:
                    return False
                # If obstacle is cardinally adjacent, blade edge touches it
                blade = env.get_blade_coverage(cell)
                if adj_cell in blade:
                    return False
    
    return True


def _find_complete_strict_safe_path(env: LawnmowerEnvironment, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Find COMPLETE strict safe path between two points
    Uses STRICT safety checks for all moves
    """
    if start == goal:
        return [start]
    
    # Try direct move first if it's strictly safe
    if _is_strictly_safe_move(env, start, goal):
        # Check if it's a single step
        dx = abs(goal[0] - start[0])
        dy = abs(goal[1] - start[1])
        if dx <= 1 and dy <= 1:
            return [start, goal]
    
    # Use A* with STRICT safety checks
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    f_score = {start: _manhattan_heuristic(start, goal)}
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        if current == goal:
            # Reconstruct complete path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        # Check 8 neighbors
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Skip if not a valid position
            if not env.is_valid_position(neighbor):
                continue
            
            # STRICT safety check
            if not _is_strictly_safe_move(env, current, neighbor):
                continue
            
            # Cost (slightly penalize diagonals)
            cost = 1.0 if dx == 0 or dy == 0 else 1.414
            
            tentative_g = g_score[current] + cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + _manhattan_heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return []  # No path found


def _strict_safety_mop_up_generator(env: LawnmowerEnvironment, current_pos: Tuple[int, int], 
                                   mowed_cells: Set[Tuple[int, int]], covered: Set[Tuple[int, int]]) -> Generator[Tuple[int, int], None, None]:
    """
    Strict safety mop-up generator
    Only visits positions that are strictly safe
    """
    # Find all unmowed reachable cells
    unmowed = env.reachable_lawn_cells - mowed_cells
    
    if not unmowed:
        return
    
    print(f"  Mop-up: {len(unmowed)} cells to cover")
    
    # Group unmowed cells by finding mower positions that can cover them
    position_to_cells = {}
    
    for cell in unmowed:
        # Find mower positions that can cover this cell
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                pos = (cell[0] - dx, cell[1] - dy)
                if pos in env.lawn_cells:
                    # Check if this position is strictly safe
                    blade = env.get_blade_coverage(pos)
                    if cell in blade:
                        # Verify position is safe
                        safe = True
                        for blade_cell in blade:
                            if blade_cell in env.obstacle_cells:
                                safe = False
                                break
                        if safe:
                            position_to_cells.setdefault(pos, []).append(cell)
    
    # Process positions by how many cells they cover
    sorted_positions = sorted(position_to_cells.items(), 
                             key=lambda x: len(x[1]), 
                             reverse=True)
    
    for target_pos, cells_to_cover in sorted_positions:
        # Skip if already at this position
        if current_pos == target_pos:
            continue
        
        # Skip if all cells are already mowed
        cells_to_cover = [c for c in cells_to_cover if c in unmowed]
        if not cells_to_cover:
            continue
        
        # Move to this position with STRICT safety
        safe_path = _find_complete_strict_safe_path(env, current_pos, target_pos)
        if safe_path and len(safe_path) > 1:
            for i in range(1, len(safe_path)):
                pos = safe_path[i]
                # Update coverage
                blade_cells = env.get_blade_coverage(pos)
                for cell in blade_cells:
                    if cell in env.reachable_lawn_cells:
                        mowed_cells.add(cell)
                covered.add(pos)
                yield pos
                current_pos = pos
        
        # Mark cells as covered
        for cell in cells_to_cover:
            if cell in unmowed:
                unmowed.remove(cell)
        
        # Early exit if all done
        if not unmowed:
            break
    
    if unmowed:
        print(f"  Warning: {len(unmowed)} cells still unmowed after mop-up")


def _manhattan_heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Manhattan distance heuristic"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ========== ORIGINAL FUNCTIONS (keep as is) ==========

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
        if largest is None or largest['area'] < 1:
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
            # Check 8-connected neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in remaining:  # Only check if in remaining, not lawn_cells
                    component.add(neighbor)
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        
        components.append(component)
    
    return components