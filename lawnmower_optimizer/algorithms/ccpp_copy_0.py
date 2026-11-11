"""
Complete Coverage Path Planning Algorithm for Lawn Mowing
FINAL WORKING VERSION
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Deque
from collections import deque, defaultdict
import heapq
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class Region:
    """A boustrophedon-friendly region"""
    id: int
    cells: Set[Tuple[int, int]]
    bounding_box: Tuple[int, int, int, int]  # x_min, y_min, x_max, y_max
    entry_points: List[Tuple[int, int]]
    exit_points: List[Tuple[int, int]]
    neighbors: Set[int] = None
    
    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = set()


class CompleteCoveragePlanner:
    """
    Complete coverage planner - FINAL FIXED VERSION
    """
    
    def __init__(self, env, blade_diameter: float = 70.0):
        self.env = env
        self.blade_diameter = blade_diameter
        self.grid_cell_size = blade_diameter / 2.0
        
        # State
        self.regions: List[Region] = []
        self.region_map: Dict[Tuple[int, int], int] = {}
        self.path: List[Tuple[int, int]] = []
        
        # For visualization and verification
        self.mowed_cells = set()
        self.visited_positions = set()
        self.invalid_moves = []
        
        print(f"Initializing CompleteCoveragePlanner")
        print(f"  Grid: {env.width}x{env.height}")
        print(f"  Reachable cells: {len(env.reachable_lawn_cells)}")
    
    def plan_full_coverage(self) -> List[Tuple[int, int]]:
        """
        Main algorithm - returns path covering all reachable cells
        """
        print("\n" + "="*60)
        print("COMPLETE COVERAGE PLANNING ALGORITHM")
        print("="*60)
        
        # Step 1: Find connected components
        print("\nStep 1: Finding connected components...")
        self._find_connected_regions()
        
        # Step 2: Generate coverage for each region
        print(f"\nStep 2: Generating coverage for {len(self.regions)} regions...")
        full_path = []
        current_pos = self.env.start_pos
        
        # Start path at start position
        full_path.append(current_pos)
        self.visited_positions.add(current_pos)
        self._update_mowed_cells(current_pos)
        
        for region_id in range(len(self.regions)):
            region = self.regions[region_id]
            
            # If not already in this region, move to it
            if current_pos not in region.cells:
                # Find closest valid entry point
                best_entry = None
                best_dist = float('inf')
                
                for entry in region.entry_points:
                    if not self.env.is_valid_position(entry):
                        continue
                    dist = self._manhattan_distance(current_pos, entry)
                    if dist < best_dist:
                        best_dist = dist
                        best_entry = entry
                
                if best_entry:
                    print(f"  Moving to region {region_id}...")
                    move_path = self._a_star_search_full_path(current_pos, best_entry)
                    if move_path:
                        # Add ENTIRE A* path to our path
                        for i in range(1, len(move_path)):
                            if self._is_valid_move_sequence(move_path[i-1], move_path[i]):
                                full_path.append(move_path[i])
                                self.visited_positions.add(move_path[i])
                                self._update_mowed_cells(move_path[i])
                            else:
                                print(f"    WARNING: Invalid move in A* path: {move_path[i-1]} -> {move_path[i]}")
                        current_pos = move_path[-1]
            
            # Generate coverage within region
            print(f"  Covering region {region_id} ({len(region.cells)} cells)...")
            region_path = self._cover_region_with_a_star(region, current_pos)
            if region_path:
                # Add entire region path
                for pos in region_path:
                    full_path.append(pos)
                    self.visited_positions.add(pos)
                    self._update_mowed_cells(pos)
                current_pos = region_path[-1]
        
        self.path = full_path
        
        # Step 3: Verify and display results
        print("\nStep 3: Final verification...")
        self._verify_full_path(full_path)
        
        return full_path
    
    def _find_connected_regions(self) -> None:
        """
        Find connected components of reachable cells that are also valid positions
        """
        visited = set()
        regions = []
        region_id = 0
        
        # Use valid mower positions that are also reachable lawn
        valid_cells = self.env.lawn_cells.intersection(self.env.reachable_lawn_cells)
        
        for cell in valid_cells:
            if cell in visited:
                continue
            
            # BFS to find connected region
            region_cells = set()
            queue = deque([cell])
            
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                
                visited.add(current)
                region_cells.add(current)
                
                # Check 4-directional connectivity
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    neighbor = (current[0] + dx, current[1] + dy)
                    if (neighbor in valid_cells and 
                        neighbor not in visited):
                        queue.append(neighbor)
            
            if region_cells:
                region = self._create_region_from_cells(region_id, region_cells)
                regions.append(region)
                region_id += 1
        
        self.regions = regions
        
        # Build region map
        for region in regions:
            for cell in region.cells:
                self.region_map[cell] = region.id
        
        print(f"  Found {len(regions)} connected regions")
    
    def _create_region_from_cells(self, region_id: int, 
                                 cells: Set[Tuple[int, int]]) -> Region:
        """Create a region from cells"""
        x_coords = [c[0] for c in cells]
        y_coords = [c[1] for c in cells]
        bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        
        # Find boundary cells that are valid positions
        entry_exit = []
        for cell in cells:
            x, y = cell
            if (x == bbox[0] or x == bbox[2] or 
                y == bbox[1] or y == bbox[3]):
                if cell in self.env.lawn_cells:
                    entry_exit.append(cell)
        
        if not entry_exit:
            entry_exit = list(cells)
        
        return Region(
            id=region_id,
            cells=cells,
            bounding_box=bbox,
            entry_points=entry_exit,
            exit_points=entry_exit.copy(),
            neighbors=set()
        )
    
    def _cover_region_with_a_star(self, region: Region, 
                                 start_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Generate coverage using A* between boustrophedon points
        """
        # Get valid cells in region
        valid_cells = [cell for cell in region.cells 
                      if cell in self.env.lawn_cells]
        
        if not valid_cells:
            return []
        
        # Group by row for boustrophedon
        rows = defaultdict(list)
        for x, y in valid_cells:
            rows[y].append((x, y))
        
        # Sort rows
        sorted_rows = sorted(rows.items())
        
        # Find start row
        start_row_idx = 0
        for i, (y, _) in enumerate(sorted_rows):
            if y == start_pos[1]:
                start_row_idx = i
                break
        
        # Generate boustrophedon target points
        target_points = []
        left_to_right = True
        
        # Downward rows
        for i in range(start_row_idx, len(sorted_rows)):
            y, row_cells = sorted_rows[i]
            row_cells.sort(key=lambda c: c[0])
            
            if left_to_right:
                target_points.extend(row_cells)
            else:
                target_points.extend(reversed(row_cells))
            
            left_to_right = not left_to_right
        
        # Upward rows (before start)
        if start_row_idx > 0:
            left_to_right = not left_to_right  # Reverse direction
            
            for i in range(start_row_idx - 1, -1, -1):
                y, row_cells = sorted_rows[i]
                row_cells.sort(key=lambda c: c[0])
                
                if left_to_right:
                    target_points.extend(row_cells)
                else:
                    target_points.extend(reversed(row_cells))
                
                left_to_right = not left_to_right
        
        # Now connect all points with A*
        full_path = []
        current = start_pos
        
        for target in target_points:
            if current == target:
                continue
            
            # Get A* path
            a_star_path = self._a_star_search_full_path(current, target)
            if a_star_path and len(a_star_path) > 1:
                # Add all intermediate points
                for i in range(1, len(a_star_path)):
                    next_pos = a_star_path[i]
                    if self._is_valid_move_sequence(current, next_pos):
                        full_path.append(next_pos)
                        current = next_pos
                    else:
                        # Skip invalid move
                        break
        
        return full_path
    
    def _a_star_search_full_path(self, start: Tuple[int, int], 
                                goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        A* search that returns full path and respects ALL movement constraints
        """
        if start == goal:
            return [start]
        
        # Check if positions are valid
        if not self.env.is_valid_position(start) or not self.env.is_valid_position(goal):
            return []
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct full path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            # Check 8-directional neighbors
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                          (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # CRITICAL: Use environment's move validation
                if not self.env.is_valid_move(current, neighbor):
                    continue
                
                # Cost (1 for cardinal, √2 for diagonal)
                cost = 1.0 if dx == 0 or dy == 0 else 1.414
                tentative_g = g_score[current] + cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []
    
    def _is_valid_move_sequence(self, from_pos: Tuple[int, int], 
                               to_pos: Tuple[int, int]) -> bool:
        """
        Validate a move using environment's validation
        Also track invalid moves for reporting
        """
        is_valid = self.env.is_valid_move(from_pos, to_pos)
        if not is_valid:
            self.invalid_moves.append((from_pos, to_pos))
        return is_valid
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic"""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _update_mowed_cells(self, pos: Tuple[int, int]):
        """Update mowed cells based on mower position"""
        blade_cells = self.env.get_blade_coverage(pos)
        for cell in blade_cells:
            if cell in self.env.reachable_lawn_cells:
                self.mowed_cells.add(cell)
    
    def _verify_full_path(self, path: List[Tuple[int, int]]):
        """
        Verify the entire path and report statistics
        """
        if not path:
            print("  Empty path!")
            return
        
        # Simulate the path
        state = self.env.create_state()
        valid_moves = 0
        
        for i in range(1, len(path)):
            if self.env.is_valid_move(state.position, path[i]):
                valid_moves += 1
                state = self.env.move(state, path[i])
            else:
                print(f"  INVALID MOVE {i}: {state.position} -> {path[i]}")
        
        # Calculate coverage
        total_reachable = len(self.env.reachable_lawn_cells)
        coverage = 100 * len(self.mowed_cells) / total_reachable if total_reachable > 0 else 0
        efficiency = 100 * len(self.mowed_cells) / len(path) if path else 0
        
        print(f"\nResults:")
        print(f"  Path length: {len(path)} steps")
        print(f"  Valid moves: {valid_moves}/{len(path)-1}")
        print(f"  Coverage: {coverage:.1f}% ({len(self.mowed_cells)}/{total_reachable})")
        print(f"  Efficiency: {efficiency:.1f}%")
        
        if len(self.invalid_moves) > 0:
            print(f"  ⚠️  Found {len(self.invalid_moves)} invalid moves!")
            for i, (frm, to) in enumerate(self.invalid_moves[:3]):  # Show first 3
                print(f"    {i+1}. {frm} -> {to}")
            if len(self.invalid_moves) > 3:
                print(f"    ... and {len(self.invalid_moves)-3} more")
    
    def visualize_side_by_side(self, path: List[Tuple[int, int]] = None):
        """
        Show side-by-side visualization: planned path vs mowing results
        """
        if path is None:
            path = self.path
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # --- LEFT PLOT: Planned Path ---
        # Draw obstacles
        for obs in self.env.obstacle_cells:
            x, y = obs
            ax1.add_patch(plt.Rectangle((x, y), 1, 1, color='black', alpha=0.7))
        
        # Draw reachable lawn (unmowed)
        for cell in self.env.reachable_lawn_cells:
            x, y = cell
            ax1.add_patch(plt.Rectangle((x, y), 1, 1, color='lightgreen', alpha=0.3))
        
        # Draw the path
        if path:
            path_x = [p[0] + 0.5 for p in path]
            path_y = [p[1] + 0.5 for p in path]
            ax1.plot(path_x, path_y, 'r-', linewidth=2, alpha=0.7, label='Planned Path')
            ax1.scatter(path_x[0], path_y[0], color='blue', s=100, 
                       marker='o', label='Start')
            ax1.scatter(path_x[-1], path_y[-1], color='red', s=100, 
                       marker='X', label='End')
            
            # Mark invalid moves if any
            for frm, to in self.invalid_moves:
                ax1.plot([frm[0] + 0.5, to[0] + 0.5], 
                        [frm[1] + 0.5, to[1] + 0.5], 
                        'y-', linewidth=3, alpha=0.5, label='Invalid Move' if frm == self.invalid_moves[0][0] else '')
        
        ax1.set_aspect('equal')
        ax1.set_xlim(-1, self.env.width + 1)
        ax1.set_ylim(-1, self.env.height + 1)
        ax1.set_title(f"CCPP Planned Path\nSteps: {len(path)}, Regions: {len(self.regions)}")
        ax1.invert_yaxis()
        ax1.legend(loc='upper right')
        
        # --- RIGHT PLOT: Mowing Results ---
        # Draw obstacles
        for obs in self.env.obstacle_cells:
            x, y = obs
            ax2.add_patch(plt.Rectangle((x, y), 1, 1, color='black', alpha=0.7))
        
        # Draw unmowed reachable cells
        unmowed = self.env.reachable_lawn_cells - self.mowed_cells
        for cell in unmowed:
            x, y = cell
            ax2.add_patch(plt.Rectangle((x, y), 1, 1, color='yellow', alpha=0.5))
        
        # Draw mowed cells
        for cell in self.mowed_cells:
            x, y = cell
            ax2.add_patch(plt.Rectangle((x, y), 1, 1, color='green', alpha=0.8))
        
        # Draw visited positions
        for pos in self.visited_positions:
            ax2.scatter(pos[0] + 0.5, pos[1] + 0.5, color='red', s=20, alpha=0.7)
        
        # Add start and end
        if path:
            ax2.scatter(path[0][0] + 0.5, path[0][1] + 0.5, 
                       color='blue', s=100, marker='o')
            ax2.scatter(path[-1][0] + 0.5, path[-1][1] + 0.5, 
                       color='red', s=100, marker='X')
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.8, label=f'Mowed ({len(self.mowed_cells)})'),
            Patch(facecolor='yellow', alpha=0.5, label=f'Unmowed ({len(unmowed)})'),
            Patch(facecolor='black', alpha=0.7, label='Obstacle'),
            Patch(facecolor='red', alpha=0.3, label='Mower Position')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        ax2.set_aspect('equal')
        ax2.set_xlim(-1, self.env.width + 1)
        ax2.set_ylim(-1, self.env.height + 1)
        coverage = 100 * len(self.mowed_cells) / len(self.env.reachable_lawn_cells) if self.env.reachable_lawn_cells else 0
        ax2.set_title(f"Mowing Results\nCoverage: {coverage:.1f}%")
        ax2.invert_yaxis()
        
        plt.tight_layout()
        plt.show()


# Generator interface for main.py
def ccpp_algorithm(env, show_visualization):
    """
    Generator interface for main.py
    """
    planner = CompleteCoveragePlanner(env)
    path = planner.plan_full_coverage()
    
    # Show visualization only if requested
    if show_visualization:
        planner.visualize_side_by_side(path)

    # Yield each position
    for pos in path:
        yield pos