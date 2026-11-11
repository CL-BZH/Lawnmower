"""
MST Coverage - Clean version
Builds MST over valid mower POSITIONS (not cells to mow)
"""
import heapq
from typing import Tuple, List, Generator, Set, Dict, Optional
from utils.environment import LawnmowerEnvironment


def mst_coverage(env: LawnmowerEnvironment) -> Generator[Tuple[int, int], None, None]:
    """
    MST-based coverage that builds spanning tree over mower positions
    """
    print("Building Minimum Spanning Tree...")
    
    # Build MST over mower positions
    lawn_positions = list(env.lawn_cells)
    if not lawn_positions:
        return
    
    print(f"Building MST over {len(lawn_positions)} valid mower positions...")
    
    # Build adjacency with direction penalties
    adjacency = _build_adjacency(env, lawn_positions)
    
    # Build MST
    edges, parent, children, edge_directions = _build_mst(
        env.start_pos, lawn_positions, adjacency)
    
    print(f"MST built with {len(edges)} edges")
    
    # Traverse MST
    path = _traverse_mst(env, env.start_pos, parent, children, 
                        lawn_positions, edge_directions)
    
    print(f"MST completed: {len(set(path))}/{len(lawn_positions)} positions")
    print(f"Total steps: {len(path)}")
    
    for pos in path:
        yield pos


def _build_adjacency(env: LawnmowerEnvironment,
                    lawn_cells: List[Tuple[int, int]]) -> Dict:
    """Build adjacency with diagonal penalties"""
    lawn_set = set(lawn_cells)
    adjacency = {}
    
    for cell in lawn_cells:
        neighbors = []
        
        for neighbor in env.get_neighbors(cell):
            if neighbor in lawn_set:
                dx = abs(neighbor[0] - cell[0])
                dy = abs(neighbor[1] - cell[1])
                
                # Orthogonal moves preferred
                if dx == 0 or dy == 0:
                    weight = (dx + dy)  # Distance 1
                else:
                    # Diagonal move - penalty
                    weight = (dx*dx + dy*dy) ** 0.5 * 5.0
                
                neighbors.append((neighbor, weight))
        
        adjacency[cell] = neighbors
    
    return adjacency


def _build_mst(
        start: Tuple[int, int],
        lawn_cells: List[Tuple[int, int]],
        adjacency: Dict) -> Tuple[List, Dict, Dict, Dict]:
    """Build Minimum Spanning Tree using Prim's algorithm"""
    lawn_set = set(lawn_cells)
    
    # Initialize
    visited = set()
    parent = {}
    children = {cell: [] for cell in lawn_cells}
    edges = []
    edge_directions = {}
    
    # Priority queue
    pq = [(0, start, None, None)]
    
    while pq and len(visited) < len(lawn_cells):
        weight, current, par, direction = heapq.heappop(pq)
        
        if current in visited:
            continue
        
        visited.add(current)
        
        if par is not None:
            parent[current] = par
            children[par].append(current)
            edges.append((par, current))
            if direction is not None:
                edge_directions[(par, current)] = direction
        
        # Get incoming direction
        incoming_direction = edge_directions.get((parent.get(current), current))
        
        for neighbor, edge_weight in adjacency.get(current, []):
            if neighbor not in visited:
                # Calculate direction
                dx = neighbor[0] - current[0]
                dy = neighbor[1] - current[1]
                new_direction = (dx, dy)
                
                # Direction change penalty
                penalty = 0
                if incoming_direction is not None and new_direction != incoming_direction:
                    penalty = 50.0
                
                total_weight = edge_weight + penalty
                heapq.heappush(pq, (total_weight, neighbor, current, new_direction))
    
    return edges, parent, children, edge_directions


def _traverse_mst(
        env: LawnmowerEnvironment,
        start: Tuple[int, int],
        parent: Dict,
        children: Dict,
        lawn_cells: List[Tuple[int, int]],
        edge_directions: Dict) -> List[Tuple[int, int]]:
    """Traverse MST with direction-aware child ordering"""
    path = [start]
    visited = {start}
    current = start
    
    lawn_set = set(lawn_cells)
    
    while len(visited) < len(lawn_cells):
        node_children = children.get(current, [])
        unvisited_children = [c for c in node_children if c not in visited]
        
        if unvisited_children:
            # Sort by direction alignment
            incoming_edge = (parent[current], current) if current in parent else None
            
            if incoming_edge and incoming_edge in edge_directions:
                incoming_dir = edge_directions[incoming_edge]
                
                def direction_score(child):
                    child_edge = (current, child)
                    if child_edge in edge_directions:
                        child_dir = edge_directions[child_edge]
                        if child_dir == incoming_dir:
                            return 0
                        if child_dir == (-incoming_dir[0], -incoming_dir[1]):
                            return 3
                        return 1
                    return 2
                
                unvisited_children.sort(key=direction_score)
            
            next_node = unvisited_children[0]
            
            # Move to next node
            if env.is_valid_move(current, next_node):
                path.append(next_node)
                visited.add(next_node)
                current = next_node
            else:
                # Need pathfinding
                subpath = _find_path_astar(env, current, next_node, lawn_set)
                if subpath and subpath[-1] == next_node:
                    for pos in subpath[1:]:
                        path.append(pos)
                        if pos in lawn_set:
                            visited.add(pos)
                    current = next_node
                else:
                    # Can't reach - skip
                    visited.add(next_node)
        else:
            # Find nearest unvisited
            unvisited = lawn_set - visited
            if not unvisited:
                break
            
            nearest = min(unvisited, 
                         key=lambda p: (p[0]-current[0])**2 + (p[1]-current[1])**2)
            
            subpath = _find_path_astar(env, current, nearest, lawn_set)
            if subpath and subpath[-1] == nearest:
                for pos in subpath[1:]:
                    path.append(pos)
                    if pos in lawn_set:
                        visited.add(pos)
                current = nearest
            else:
                visited.add(nearest)
    
    return path


def _find_path_astar(env: LawnmowerEnvironment,
                    start: Tuple[int, int],
                    goal: Tuple[int, int],
                    valid_cells: Set[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
    """A* pathfinding"""
    if start == goal:
        return [start]
    
    open_set = [(0, 0, start)]
    came_from = {}
    g_score = {start: 0}
    counter = 1
    
    while open_set:
        _, _, current = heapq.heappop(open_set)
        
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        
        for neighbor in env.get_neighbors(current):
            if neighbor not in valid_cells:
                continue
            
            tentative_g = g_score[current] + 1
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                
                h = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                f = tentative_g + h
                
                heapq.heappush(open_set, (f, counter, neighbor))
                counter += 1
    
    return None
