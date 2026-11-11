"""
Greedy Algorithm - Always move to the nearest unmowed lawn cell
"""
import numpy as np
from typing import Tuple, List, Generator, Set
from utils.environment import LawnmowerEnvironment, MowerState


def greedy_algorithm(env: LawnmowerEnvironment, 
                    look_ahead: int = 5) -> Generator[Tuple[int, int], None, None]:
    """
    Greedy lawn mowing algorithm
    
    Strategy:
    - From current position, look in each direction
    - Score each direction based on:
      1. Number of unmowed cells visible
      2. Distance to first unmowed cell
      3. Preference for continuing straight
    - Move to highest scoring direction
    
    Args:
        env: LawnmowerEnvironment instance
        look_ahead: Maximum distance to look in each direction
    
    Yields:
        (x, y) tuples representing the path
    """
    state = env.create_state()
    yield state.position
    
    max_iterations = len(env.lawn_cells) * 3  # Safety limit
    iteration = 0
    
    while not env.is_complete(state) and iteration < max_iterations:
        iteration += 1
        
        # Score each possible direction
        best_score = -float('inf')
        best_direction = None
        best_next_pos = None
        
        unmowed_cells = env.get_unmowed_cells(state)
        
        for heading, (dx, dy) in env.DIRECTIONS.items():
            next_pos = (state.position[0] + dx, state.position[1] + dy)
            
            # Skip invalid moves
            if not env.is_valid_move(state.position, next_pos):
                continue
            
            score = _score_direction(
                env, state, next_pos, heading, look_ahead, unmowed_cells
            )
            
            if score > best_score:
                best_score = score
                best_direction = heading
                best_next_pos = next_pos
        
        if best_next_pos is None:
            # No valid moves - try to jump to nearest unmowed area
            print(f"Greedy stuck at {state.position}, attempting jump...")
            
            # Find position that mows unmowed cells
            best_jump = None
            min_dist = float('inf')
            
            for pos in env.lawn_cells:
                blade = env.get_blade_coverage(pos)
                if blade & unmowed_cells:
                    dist = abs(pos[0] - state.position[0]) + abs(pos[1] - state.position[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_jump = pos
            
            if best_jump:
                # Use A* to path to it
                from .astar import a_star_path
                path = a_star_path(env, state.position, best_jump, set())
                if path and len(path) > 1:
                    # Follow path
                    for pos in path[1:]:
                        state = env.move(state, pos)
                        yield state.position
                    continue
            
            # Still stuck
            print(f"Greedy cannot escape, terminating")
            break
        
        # Make the move
        state = env.move(state, best_next_pos)
        yield state.position
    
    print(f"Greedy completed: {len(state.mowed_cells)}/"
          f"{len(env.reachable_lawn_cells)} cells")
    print(f"Efficiency: {env.calculate_efficiency(state):.1f}%")
    print(f"Direction changes: {state.direction_changes}")


def _score_direction(env: LawnmowerEnvironment, 
                    state: MowerState,
                    next_pos: Tuple[int, int],
                    heading: int,
                    look_ahead: int,
                    unmowed_cells: Set[Tuple[int, int]]) -> float:
    """
    Score a direction based on potential for mowing
    
    Returns higher scores for directions with:
    - More unmowed cells in sight
    - Closer unmowed cells
    - Continuation of current heading (minimize turns)
    """
    dx, dy = env.DIRECTIONS[heading]
    score = 0.0
    
    # Bonus for continuing straight (reduce direction changes)
    if heading == state.heading:
        score += 50.0  # Strong incentive to go straight
    
    # Immediate bonus if next position mows unmowed cells
    blade_coverage = env.get_blade_coverage(next_pos)
    if blade_coverage & unmowed_cells:
        score += 20.0
    
    # Look ahead in this direction
    unmowed_count = 0
    distance_to_first = None
    
    x, y = next_pos
    for step in range(1, look_ahead + 1):
        look_pos = (x + dx * step, y + dy * step)
        
        if not env.is_valid_position(look_pos):
            break
        
        # Check if this position would mow unmowed cells
        if look_pos in env.lawn_cells:
            look_blade = env.get_blade_coverage(look_pos)
            if look_blade & unmowed_cells:
                unmowed_count += 1
                if distance_to_first is None:
                    distance_to_first = step
    
    # Score based on unmowed cells visible
    score += unmowed_count * 5.0
    
    # Score inversely proportional to distance to first unmowed
    if distance_to_first is not None:
        score += 10.0 / distance_to_first
    
    return score
