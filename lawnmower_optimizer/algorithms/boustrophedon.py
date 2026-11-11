"""
Boustrophedon (Lawnmower Pattern) Algorithm - Systematic back-and-forth coverage
"""
import numpy as np
from typing import Tuple, List, Generator, Set
from utils.environment import LawnmowerEnvironment, MowerState


def boustrophedon_algorithm(env: LawnmowerEnvironment) -> Generator[Tuple[int, int], None, None]:
    """
    Boustrophedon coverage algorithm (classic lawnmower pattern)
    
    Strategy:
    - Sweep left-to-right (or right-to-left)
    - Move down one row
    - Sweep in opposite direction
    - Repeat until complete
    
    Yields:
        (x, y) tuples representing the path
    """
    state = env.create_state()
    yield state.position
    
    # Determine sweep direction based on grid shape
    sweep_horizontally = env.width >= env.height
    
    # Get bounding box of lawn cells
    lawn_positions = list(env.lawn_cells)
    if not lawn_positions:
        return
    
    xs = [x for x, y in lawn_positions]
    ys = [y for x, y in lawn_positions]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    if sweep_horizontally:
        path = _horizontal_sweep(env, state, min_x, max_x, min_y, max_y)
    else:
        path = _vertical_sweep(env, state, min_x, max_x, min_y, max_y)
    
    for pos in path:
        if pos != state.position:
            if env.is_valid_position(pos) and env.is_valid_move(state.position, pos):
                state = env.move(state, pos)
                yield state.position
            else:
                print(f"Warning: Invalid move from {state.position} to {pos}, skipping")
    
    print(f"Boustrophedon completed: {len(state.mowed_cells)}/"
          f"{len(env.all_lawn_cells)} cells")
    print(f"Efficiency: {env.calculate_efficiency(state):.1f}%")
    print(f"Direction changes: {state.direction_changes}")


def _horizontal_sweep(env: LawnmowerEnvironment, 
                     state: MowerState,
                     min_x: int, max_x: int,
                     min_y: int, max_y: int) -> List[Tuple[int, int]]:
    """
    Sweep horizontally (left-right, then right-left alternating)
    
    Ensures all positions are connected with valid moves using A* when needed
    """
    path = []
    current_pos = state.position
    
    # Sweep every 2 rows to be efficient
    y_positions = list(range(min_y, max_y + 1, 2))
    
    for i, y in enumerate(y_positions):
        # Collect all valid positions in this row
        row_positions = []
        
        # Determine sweep direction
        if i % 2 == 0:
            x_range = range(min_x, max_x + 1, 1)
        else:
            x_range = range(max_x, min_x - 1, -1)

        for x in x_range:
            pos = (x, y)
            if pos in env.lawn_cells:
                row_positions.append(pos)
        
        if not row_positions:
            continue
        
        # Connect to first position in row
        start_of_row = row_positions[0]
        
        if not path:
            # Very first position
            if start_of_row != current_pos:
                subpath = _connect_positions(env, current_pos, start_of_row)
                if subpath:
                    path.extend(subpath)
            else:
                path.append(start_of_row)
        else:
            # Connect from last position to start of this row
            last = path[-1]
            subpath = _connect_positions(env, last, start_of_row)
            if subpath:
                path.extend(subpath[1:])  # Skip duplicate start
        
        # Connect all positions within the row
        for j in range(1, len(row_positions)):
            prev = row_positions[j-1] if j > 0 else path[-1]
            curr = row_positions[j]
            
            # Try direct move first
            if path and env.is_valid_move(path[-1], curr):
                path.append(curr)
            else:
                # Use A* to connect
                subpath = _connect_positions(env, path[-1] if path else current_pos, curr)
                if subpath:
                    path.extend(subpath[1:])
    
    return path


def _vertical_sweep(env: LawnmowerEnvironment,
                   state: MowerState,
                   min_x: int, max_x: int,
                   min_y: int, max_y: int) -> List[Tuple[int, int]]:
    """
    Sweep vertically (top-bottom, then bottom-top alternating)
    
    Ensures all positions are connected with valid moves using A* when needed
    """
    path = []
    current_pos = state.position
    
    # Sweep every 2 columns to be efficient
    x_positions = list(range(min_x, max_x + 1, 2))
    
    for i, x in enumerate(x_positions):
        # Collect all valid positions in this column
        col_positions = []
        
        # Determine sweep direction
        if i % 2 == 0:
            y_range = range(min_y, max_y + 1, 1)
        else:
            y_range = range(max_y, min_y - 1, -1)
        
        for y in y_range:
            pos = (x, y)
            if pos in env.lawn_cells:
                col_positions.append(pos)
        
        if not col_positions:
            continue
        
        # Connect to first position in column
        start_of_col = col_positions[0]
        
        if not path:
            # Very first position
            if start_of_col != current_pos:
                subpath = _connect_positions(env, current_pos, start_of_col)
                if subpath:
                    path.extend(subpath)
            else:
                path.append(start_of_col)
        else:
            # Connect from last position to start of this column
            last = path[-1]
            subpath = _connect_positions(env, last, start_of_col)
            if subpath:
                path.extend(subpath[1:])  # Skip duplicate start
        
        # Connect all positions within the column
        for j in range(1, len(col_positions)):
            curr = col_positions[j]
            
            # Try direct move first
            if path and env.is_valid_move(path[-1], curr):
                path.append(curr)
            else:
                # Use A* to connect
                subpath = _connect_positions(env, path[-1] if path else current_pos, curr)
                if subpath:
                    path.extend(subpath[1:])
    
    return path


def _connect_positions(env: LawnmowerEnvironment,
                      start: Tuple[int, int],
                      end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Find path between two positions using A* pathfinding
    
    This allows Boustrophedon to navigate around obstacles to reach
    disconnected sweep regions
    """
    from .astar import a_star_path
    
    # Use A* to find path
    path = a_star_path(env, start, end)
    
    if path is None:
        # Can't reach - return empty
        return []
    
    return path
