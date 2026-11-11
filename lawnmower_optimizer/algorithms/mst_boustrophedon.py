"""
MST-Boustrophedon Hybrid
Uses MST structure but applies boustrophedon principles during traversal

Key ideas:
1. Build MST with strong straight-line preference (like current MST)
2. Traverse MST but identify "sweep segments" (long straight runs)
3. Execute sweeps efficiently within segments
"""

import heapq
from typing import Tuple, List, Generator, Set, Dict, Optional
from utils.environment import LawnmowerEnvironment


def mst_boustrophedon_hybrid(env: LawnmowerEnvironment) -> Generator[Tuple[int, int], None, None]:
    """
    MST-Boustrophedon hybrid coverage
    """
    print("Building MST-Boustrophedon hybrid...")
    
    # Build MST over mower positions with strong line preference
    lawn_positions = list(env.lawn_cells)
    if not lawn_positions:
        return
    
    print(f"Building MST over {len(lawn_positions)} positions...")
    
    # Build adjacency with very strong diagonal penalty
    adjacency = _build_adjacency_sweep_focused(env, lawn_positions)
    
    # Build MST
    edges, parent, children, edge_directions = _build_mst(
        env.start_pos, lawn_positions, adjacency)
    
    print(f"MST built with {len(edges)} edges")
    
    # Identify sweep segments in the tree
    sweep_segments = _identify_sweep_segments(parent, children, edge_directions, lawn_positions)
    print(f"Identified {len(sweep_segments)} sweep segments")
    
    # Traverse with sweep optimization
    path = _traverse_with_sweeps(env, env.start_pos, parent, children, 
                                 edge_directions, sweep_segments, lawn_positions)
    
    print(f"MST-Boustrophedon completed: {len(set(path))}/{len(lawn_positions)} positions")
    print(f"Total steps: {len(path)}")
    
    for pos in path:
        yield pos


def _build_adjacency_sweep_focused(env: LawnmowerEnvironment,
                                   lawn_cells: List[Tuple[int, int]]) -> Dict:
    """Build adjacency with VERY strong preference for straight lines"""
    lawn_set = set(lawn_cells)
    adjacency = {}
    
    for cell in lawn_cells:
        neighbors = []
        
        for neighbor in env.get_neighbors(cell):
            if neighbor in lawn_set:
                dx = abs(neighbor[0] - cell[0])
                dy = abs(neighbor[1] - cell[1])
                
                # Orthogonal moves strongly preferred
                if dx == 0 or dy == 0:
                    weight = 1.0  # Very low weight
                else:
                    # Diagonal move - HUGE penalty
                    weight = 100.0  # Massive penalty to create straight sweeps
                
                neighbors.append((neighbor, weight))
        
        adjacency[cell] = neighbors
    
    return adjacency


def _build_mst(start, lawn_cells, adjacency):
    """Build MST with direction penalties"""
    lawn_set = set(lawn_cells)
    
    visited = set()
    parent = {}
    children = {cell: [] for cell in lawn_cells}
    edges = []
    edge_directions = {}
    
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
        
        incoming_direction = edge_directions.get((parent.get(current), current))
        
        for neighbor, edge_weight in adjacency.get(current, []):
            if neighbor not in visited:
                dx = neighbor[0] - current[0]
                dy = neighbor[1] - current[1]
                new_direction = (dx, dy)
                
                # MASSIVE direction change penalty for sweep formation
                penalty = 0
                if incoming_direction is not None and new_direction != incoming_direction:
                    penalty = 100.0
                
                total_weight = edge_weight + penalty
                heapq.heappush(pq, (total_weight, neighbor, current, new_direction))
    
    return edges, parent, children, edge_directions


def _identify_sweep_segments(parent, children, edge_directions, lawn_positions):
    """
    Identify long straight segments that can be swept efficiently
    A segment is a chain of nodes all in the same direction
    """
    segments = []
    visited = set()
    
    for pos in lawn_positions:
        if pos in visited or pos not in parent:
            continue
        
        # Try to build a segment starting from this position
        segment = [pos]
        current = pos
        
        # Get direction from parent
        parent_edge = (parent[current], current)
        if parent_edge not in edge_directions:
            continue
        
        segment_direction = edge_directions[parent_edge]
        
        # Extend segment forward (through children in same direction)
        while current in children:
            same_dir_children = []
            
            for child in children[current]:
                child_edge = (current, child)
                if child_edge in edge_directions:
                    child_dir = edge_directions[child_edge]
                    if child_dir == segment_direction and child not in visited:
                        same_dir_children.append(child)
            
            if len(same_dir_children) == 1:
                # Continue segment
                child = same_dir_children[0]
                segment.append(child)
                visited.add(child)
                current = child
            else:
                break
        
        if len(segment) >= 3:  # Only keep segments of 3+ nodes
            segments.append(segment)
            for node in segment:
                visited.add(node)
    
    return segments


def _traverse_with_sweeps(env, start, parent, children, edge_directions, 
                          sweep_segments, lawn_cells):
    """Traverse MST using sweep segments where possible"""
    path = [start]
    visited = {start}
    current = start
    
    lawn_set = set(lawn_cells)
    segment_dict = {node: seg for seg in sweep_segments for node in seg}
    
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
            
            # Check if we're entering a sweep segment
            if next_node in segment_dict and next_node == segment_dict[next_node][0]:
                # Execute entire segment
                segment = segment_dict[next_node]
                for seg_node in segment:
                    if seg_node not in visited:
                        if env.is_valid_move(current, seg_node):
                            path.append(seg_node)
                            visited.add(seg_node)
                            current = seg_node
                        else:
                            # Need pathfinding
                            subpath = _find_path_astar(env, current, seg_node, lawn_set)
                            if subpath and subpath[-1] == seg_node:
                                for pos in subpath[1:]:
                                    path.append(pos)
                                    if pos in lawn_set:
                                        visited.add(pos)
                                current = seg_node
                            else:
                                visited.add(seg_node)
            else:
                # Regular move
                if env.is_valid_move(current, next_node):
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
                else:
                    subpath = _find_path_astar(env, current, next_node, lawn_set)
                    if subpath and subpath[-1] == next_node:
                        for pos in subpath[1:]:
                            path.append(pos)
                            if pos in lawn_set:
                                visited.add(pos)
                        current = next_node
                    else:
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


def _find_path_astar(env, start, goal, valid_cells):
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
