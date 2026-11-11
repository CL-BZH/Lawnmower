"""
Hamiltonian Path Algorithm - Find path visiting all lawn cells 
exactly once

WARNING: This is NP-complete and can be very slow on large grids!
Uses heuristics to make it practical:
- Prioritizes continuing in same direction
- Tries closest unvisited neighbors first
- Has timeout to avoid infinite computation
"""
import random
import time
from typing import Tuple, List, Generator, Set, Optional
from utils.environment import LawnmowerEnvironment, MowerState


def hamiltonian_path(
        env: LawnmowerEnvironment,
        timeout: float = 30.0) -> Generator[Tuple[int, int], 
                                           None, None]:
    """
    Attempt to find Hamiltonian path (visit all cells exactly once)
    
    This is NP-complete, so we use heuristics and timeout.
    
    Args:
        env: LawnmowerEnvironment instance
        timeout: Maximum seconds to search (default 30)
    
    Yields:
        (x, y) tuples representing the path
    """
    print(f"Attempting Hamiltonian path...")
    print(f"WARNING: This may take up to {timeout}s on complex maps")
    
    start_time = time.time()
    lawn_cells = list(env.lawn_cells)
    
    if len(lawn_cells) > 50:
        print(f"WARNING: {len(lawn_cells)} cells - this may be "
              f"very slow!")
    
    # Build adjacency for lawn cells only
    adjacency = _build_lawn_adjacency(env, lawn_cells)
    
    # Try to find Hamiltonian path
    path = _find_hamiltonian_path(
        start=env.start_pos,
        lawn_cells=set(lawn_cells),
        adjacency=adjacency,
        timeout=timeout,
        start_time=start_time
    )
    
    elapsed = time.time() - start_time
    
    if path:
        print(f"Hamiltonian path found in {elapsed:.1f}s!")
        print(f"Path length: {len(path)} (optimal)")
        print(f"Efficiency: 100.0% (no revisits)")
        
        for pos in path:
            yield pos
    else:
        print(f"No Hamiltonian path found in {timeout}s")
        print(f"Falling back to greedy traversal...")
        
        # Fallback: greedy nearest neighbor
        for pos in _greedy_fallback(env, lawn_cells):
            yield pos


def _build_lawn_adjacency(
        env: LawnmowerEnvironment,
        lawn_cells: List[Tuple[int, int]]) -> dict:
    """Build adjacency dict for lawn cells"""
    adjacency = {}
    lawn_set = set(lawn_cells)
    
    for cell in lawn_cells:
        neighbors = []
        for neighbor in env.get_neighbors(cell):
            if neighbor in lawn_set:
                # Calculate distance (1 for orthogonal, ~1.4 for diagonal)
                dx = abs(neighbor[0] - cell[0])
                dy = abs(neighbor[1] - cell[1])
                dist = (dx**2 + dy**2) ** 0.5
                neighbors.append((neighbor, dist))
        
        # Sort by distance (prefer close neighbors)
        neighbors.sort(key=lambda x: x[1])
        adjacency[cell] = neighbors
    
    return adjacency


def _find_hamiltonian_path(
        start: Tuple[int, int],
        lawn_cells: Set[Tuple[int, int]],
        adjacency: dict,
        timeout: float,
        start_time: float) -> Optional[List[Tuple[int, int]]]:
    """
    Recursive backtracking to find Hamiltonian path
    
    Uses heuristics to speed up search:
    - Try to continue in same direction
    - Shuffle neighbors for randomization
    - Timeout to avoid infinite search
    """
    
    def backtrack(path: List[Tuple[int, int]], 
                  visited: Set[Tuple[int, int]],
                  last_direction: Optional[Tuple[int, int]] = None
                  ) -> Optional[List[Tuple[int, int]]]:
        """Backtracking helper"""
        
        # Check timeout
        if time.time() - start_time > timeout:
            return None
        
        # Base case: found complete path
        if len(visited) == len(lawn_cells):
            return path
        
        current = path[-1]
        neighbors = adjacency.get(current, [])
        
        if not neighbors:
            return None
        
        # Prioritize continuing in same direction
        if last_direction:
            dx, dy = last_direction
            preferred = (current[0] + dx, current[1] + dy)
            
            # Move preferred neighbor to front if it exists
            neighbor_positions = [n[0] for n in neighbors]
            if preferred in neighbor_positions and preferred not in visited:
                idx = neighbor_positions.index(preferred)
                neighbors = [neighbors[idx]] + neighbors[:idx] + neighbors[idx+1:]
        
        # Try each neighbor
        for neighbor, _ in neighbors:
            if neighbor not in visited:
                # Calculate direction
                direction = (neighbor[0] - current[0], 
                           neighbor[1] - current[1])
                
                # Recurse
                path.append(neighbor)
                visited.add(neighbor)
                
                result = backtrack(path, visited, direction)
                
                if result is not None:
                    return result
                
                # Backtrack
                path.pop()
                visited.remove(neighbor)
        
        return None
    
    # Start search
    initial_path = [start]
    initial_visited = {start}
    
    return backtrack(initial_path, initial_visited)


def _greedy_fallback(
        env: LawnmowerEnvironment,
        lawn_cells: List[Tuple[int, int]]
        ) -> Generator[Tuple[int, int], None, None]:
    """
    Greedy nearest-neighbor fallback when Hamiltonian fails
    """
    unvisited = set(lawn_cells)
    current = env.start_pos
    
    yield current
    if current in unvisited:
        unvisited.remove(current)
    
    while unvisited:
        # Find nearest unvisited
        nearest = min(unvisited, 
                     key=lambda p: (p[0]-current[0])**2 + 
                                   (p[1]-current[1])**2)
        
        # Move to it (may need to navigate around obstacles)
        path = _simple_path(env, current, nearest, unvisited)
        
        for pos in path:
            yield pos
            if pos in unvisited:
                unvisited.remove(pos)
        
        current = nearest


def _simple_path(
        env: LawnmowerEnvironment,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        valid_cells: Set[Tuple[int, int]]
        ) -> List[Tuple[int, int]]:
    """Simple greedy path between two points"""
    path = []
    current = start
    
    max_steps = abs(goal[0]-start[0]) + abs(goal[1]-start[1]) + 10
    steps = 0
    
    while current != goal and steps < max_steps:
        steps += 1
        
        # Move toward goal
        dx = 1 if goal[0] > current[0] else (-1 if goal[0] < current[0] else 0)
        dy = 1 if goal[1] > current[1] else (-1 if goal[1] < current[1] else 0)
        
        # Try diagonal, then cardinal
        for try_dx, try_dy in [(dx, dy), (dx, 0), (0, dy)]:
            if try_dx == 0 and try_dy == 0:
                continue
            
            next_pos = (current[0] + try_dx, current[1] + try_dy)
            
            if (env.is_valid_move(current, next_pos) and 
                (next_pos in valid_cells or next_pos == goal)):
                current = next_pos
                path.append(current)
                break
        else:
            # Stuck
            break
    
    return path
