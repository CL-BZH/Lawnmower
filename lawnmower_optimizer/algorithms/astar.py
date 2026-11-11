"""
A* Coverage Algorithm - Find optimal coverage path using A* between unmowed cells
"""
import heapq
import numpy as np
from typing import Tuple, List, Set, Optional, Generator
from utils.environment import LawnmowerEnvironment, MowerState


def a_star_coverage(env: LawnmowerEnvironment) -> Generator[Tuple[int, int], None, None]:
    """
    A* based coverage algorithm with improved heuristics
    
    Strategy:
    1. Find positions that mow the most unmowed cells
    2. Prefer positions far from recently visited areas
    3. Use A* to path efficiently
    
    Yields:
        (x, y) tuples representing the path
    """
    state = env.create_state()
    yield state.position
    
    visited_positions = {state.position}
    recent_positions = []  # Track recent positions to avoid ping-ponging
    max_recent = 20
    
    max_iterations = len(env.lawn_cells) * 2
    iterations = 0
    
    while not env.is_complete(state) and iterations < max_iterations:
        iterations += 1
        
        # Get unmowed cells
        unmowed_cells = env.get_unmowed_cells(state)
        
        if not unmowed_cells:
            break
        
        # Find best position based on multiple criteria
        best_pos = None
        best_score = -float('inf')
        
        for pos in env.lawn_cells:
            if pos in visited_positions:
                continue
            
            blade = env.get_blade_coverage(pos)
            new_cells = blade & unmowed_cells
            
            if not new_cells:
                continue
            
            # Score based on:
            # 1. Number of new cells this position mows (most important)
            # 2. Distance from current (prefer nearby)
            # 3. Distance from recent positions (avoid ping-ponging)
            
            num_new_cells = len(new_cells)
            dist_from_current = abs(pos[0] - state.position[0]) + abs(pos[1] - state.position[1])
            
            # Penalty for being near recent positions
            recent_penalty = 0
            for recent_pos in recent_positions[-max_recent:]:
                dist = abs(pos[0] - recent_pos[0]) + abs(pos[1] - recent_pos[1])
                if dist < 5:
                    recent_penalty += (5 - dist) * 2
            
            score = (num_new_cells * 10.0  # Prioritize mowing many cells
                    - dist_from_current * 0.5  # Slight preference for nearby
                    - recent_penalty)  # Avoid recent areas
            
            if score > best_score:
                best_score = score
                best_pos = pos
        
        if best_pos is None:
            break
        
        # Find path to best position
        path = a_star_path(env, state.position, best_pos, set())
        
        if not path or len(path) < 2:
            visited_positions.add(best_pos)
            continue
        
        # Follow the path
        for pos in path[1:]:
            state = env.move(state, pos)
            visited_positions.add(pos)
            recent_positions.append(pos)
            if len(recent_positions) > max_recent:
                recent_positions.pop(0)
            yield state.position
    
    print(f"A* completed: {len(state.mowed_cells)}/"
          f"{len(env.reachable_lawn_cells)} cells")
    print(f"Efficiency: {env.calculate_efficiency(state):.1f}%")
    print(f"Direction changes: {state.direction_changes}")


def a_star_path(env: LawnmowerEnvironment,
                start: Tuple[int, int],
                goal: Tuple[int, int],
                visited: Set[Tuple[int, int]] = None) -> Optional[
                    List[Tuple[int, int]]]:
    """
    A* pathfinding between two points
    
    Uses valid moves only (diagonal moves blocked if adjacent 
    cells are obstacles)
    
    Args:
        env: Environment
        start: Starting position
        goal: Goal position
        visited: Set of already visited cells (for preference)
    
    Returns:
        List of positions from start to goal, or None if no path
    """
    if visited is None:
        visited = set()
    
    # Priority queue: (f_score, counter, position)
    counter = 0
    open_set = [(0, counter, start)]
    came_from = {}
    
    g_score = {start: 0}
    f_score = {start: _heuristic(start, goal)}
    
    while open_set:
        _, _, current = heapq.heappop(open_set)
        
        if current == goal:
            return _reconstruct_path(came_from, current)
        
        # Get valid neighbors (includes diagonal check)
        for neighbor in env.get_neighbors(current):
            # Calculate movement cost
            dx = abs(neighbor[0] - current[0])
            dy = abs(neighbor[1] - current[1])
            move_distance = dx + dy
            
            # Prefer 2-cell moves over 1-cell moves
            if move_distance == 1:
                cost = 1.5  # 1-cell move penalty
            else:
                cost = 1.0  # 2-cell move (preferred)
            
            # Additional penalty for revisiting already mowed cells
            if neighbor in visited:
                cost += 0.5
            
            tentative_g = g_score[current] + cost
            
            if (neighbor not in g_score or 
                tentative_g < g_score[neighbor]):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + _heuristic(neighbor, goal)
                f_score[neighbor] = f
                
                counter += 1
                heapq.heappush(open_set, (f, counter, neighbor))
    
    return None  # No path found


def _heuristic(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """
    Heuristic for A* (Euclidean distance for 8-directional movement)
    """
    dx = abs(pos1[0] - pos2[0])
    dy = abs(pos1[1] - pos2[1])
    # Chebyshev distance (max of dx, dy) is admissible for 8-directional
    return max(dx, dy)


def _manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def _reconstruct_path(came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Reconstruct path from A* came_from dict"""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
