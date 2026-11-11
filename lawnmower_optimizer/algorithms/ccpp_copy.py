"""
Complete Coverage Path Planning Algorithm - ULTIMATE FIXED VERSION
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
    bounding_box: Tuple[int, int, int, int]
    entry_points: List[Tuple[int, int]]
    exit_points: List[Tuple[int, int]]
    neighbors: Set[int] = None
    
    def __post_init__(self):
        if self.neighbors is None:
            self.neighbors = set()


class CompleteCoveragePlanner:
    """
    Complete coverage planner - ULTIMATE FIXED VERSION
    """
    
    def __init__(self, env, blade_diameter: float = 70.0, show_visualization: bool = True):
        print("\n" + "="*60)
        print("CCPP PLANNER INITIALIZING")
        print("="*60)
        
        self.env = env
        self.show_visualization = show_visualization
        
        # Mower characteristics
        self.mower_radius = 2  # Blade radius in cells
        
        # State
        self.regions: List[Region] = []
        self.region_map: Dict[Tuple[int, int], int] = {}
        self.path: List[Tuple[int, int]] = []
        
        # For tracking
        self.mowed_cells = set()
        self.visited_positions = set()
        self.unsafe_moves = []
        self.missing_a_star_paths = []  # Track where A* paths are incomplete
        
        print(f"Grid: {env.width}x{env.height}")
        print(f"Reachable cells: {len(env.reachable_lawn_cells)}")
        print(f"Mower radius: {self.mower_radius} cells")
        print("="*60 + "\n")
    
    def plan_full_coverage(self) -> List[Tuple[int, int]]:
        """
        Main algorithm - FOLLOWS COMPLETE A* PATHS
        """
        print("\n" + "="*60)
        print("RUNNING CCPP - FOLLOWING FULL A* PATHS")
        print("="*60)
        
        # Step 1: Find connected components
        print("\nStep 1: Finding connected components...")
        self._find_connected_regions()
        
        if not self.regions:
            print("ERROR: No regions found!")
            return [self.env.start_pos]
        
        # Step 2: Generate coverage for each region
        print(f"\nStep 2: Generating coverage for {len(self.regions)} regions...")
        full_path = []
        current_pos = self.env.start_pos
        
        # Start path
        full_path.append(current_pos)
        self.visited_positions.add(current_pos)
        self._update_mowed_cells_along_move(current_pos, current_pos)  # Mow start position
        
        for region_id in range(len(self.regions)):
            region = self.regions[region_id]
            
            print(f"\nProcessing region {region_id} ({len(region.cells)} cells)")
            
            # If not already in this region, move to it WITH FULL A* PATH
            if current_pos not in region.cells:
                print(f"  Moving to region {region_id}...")
                entry = self._find_best_entry(current_pos, region)
                if entry:
                    print(f"  Found entry point: {entry}")
                    
                    # Get COMPLETE A* path
                    a_star_path = self._safe_a_star_with_full_path(current_pos, entry)
                    
                    if a_star_path and len(a_star_path) > 1:
                        print(f"  A* path has {len(a_star_path)} steps")
                        
                        # FOLLOW EVERY STEP of the A* path
                        for i in range(1, len(a_star_path)):
                            from_pos = a_star_path[i-1]
                            next_pos = a_star_path[i]
                            
                            # Check safety
                            if not self._is_safe_diagonal(from_pos, next_pos):
                                print(f"    ⚠️  Unsafe diagonal in A*: {from_pos} -> {next_pos}")
                                # Try to find alternative
                                alt_path = self._find_safe_alternative(from_pos, next_pos)
                                if alt_path:
                                    for j in range(1, len(alt_path)):
                                        alt_pos = alt_path[j]
                                        full_path.append(alt_pos)
                                        self.visited_positions.add(alt_pos)
                                        self._update_mowed_cells_along_move(from_pos, alt_pos)
                                        from_pos = alt_pos
                                    current_pos = alt_pos
                                else:
                                    print(f"    ❌ No safe alternative found, skipping move")
                                continue
                            
                            # Add step to path
                            full_path.append(next_pos)
                            self.visited_positions.add(next_pos)
                            self._update_mowed_cells_along_move(from_pos, next_pos)
                            current_pos = next_pos
                        
                        print(f"  Now at: {current_pos}")
                    else:
                        print(f"  ❌ No A* path found to region {region_id}")
                        continue
            
            # Generate coverage within region
            print(f"  Covering region {region_id}...")
            region_path = self._cover_region_with_full_paths(region, current_pos)
            
            if region_path:
                print(f"  Region path: {len(region_path)} steps")
                
                # Add entire region path
                for i in range(1, len(region_path)):
                    from_pos = region_path[i-1]
                    to_pos = region_path[i]
                    full_path.append(to_pos)
                    self.visited_positions.add(to_pos)
                    self._update_mowed_cells_along_move(from_pos, to_pos)
                
                current_pos = region_path[-1] if region_path else current_pos
            else:
                print(f"  ⚠️  No coverage path for region {region_id}")
        
        self.path = full_path
        
        # Step 3: Verify
        print("\n" + "="*60)
        print("FINAL VERIFICATION")
        print("="*60)
        self._verify_full_path(full_path)
        
        return full_path
    
    def _find_connected_regions(self):
        """Find connected components"""
        visited = set()
        regions = []
        region_id = 0
        
        valid_cells = self.env.lawn_cells.intersection(self.env.reachable_lawn_cells)
        
        for cell in valid_cells:
            if cell in visited:
                continue
            
            # BFS to find region
            region_cells = set()
            queue = deque([cell])
            
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                
                visited.add(current)
                region_cells.add(current)
                
                # 4-directional connectivity
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    neighbor = (current[0] + dx, current[1] + dy)
                    if (neighbor in valid_cells and 
                        neighbor not in visited):
                        queue.append(neighbor)
            
            if region_cells:
                region = self._create_region(region_id, region_cells)
                regions.append(region)
                region_id += 1
        
        self.regions = regions
        print(f"  Found {len(regions)} regions")
    
    def _create_region(self, region_id, cells):
        """Create region object"""
        x_coords = [c[0] for c in cells]
        y_coords = [c[1] for c in cells]
        bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        
        # Find boundary cells
        entry_exit = []
        for cell in cells:
            x, y = cell
            if (x == bbox[0] or x == bbox[2] or 
                y == bbox[1] or y == bbox[3]):
                entry_exit.append(cell)
        
        if not entry_exit:
            entry_exit = list(cells)
        
        return Region(
            id=region_id,
            cells=cells,
            bounding_box=bbox,
            entry_points=entry_exit,
            exit_points=entry_exit.copy()
        )
    
    def _cover_region_with_full_paths(self, region, start_pos):
        """
        Generate coverage with COMPLETE paths between points
        """
        # Get all cells in region
        cells = [cell for cell in region.cells if cell in self.env.lawn_cells]
        
        if not cells:
            return []
        
        # Group by row
        rows = defaultdict(list)
        for x, y in cells:
            rows[y].append((x, y))
        
        # Sort rows
        sorted_rows = sorted(rows.items())
        
        # Select optimal rows with 2-cell spacing - FIXED VERSION
        optimal_rows = self._select_optimal_rows_with_overlap(sorted_rows, region)
        
        # Generate boustrophedon order
        target_points = []
        left_to_right = True
        
        for y, row_cells in optimal_rows:
            row_cells.sort(key=lambda c: c[0])
            
            # Add extra points to ensure coverage at row boundaries
            if left_to_right:
                target_points.extend(row_cells)
                # Add one extra point at the end to ensure coverage
                if row_cells:
                    last_cell = row_cells[-1]
                    if last_cell[0] + 1 < self.env.width:
                        extra_cell = (last_cell[0] + 1, last_cell[1])
                        if extra_cell in self.env.lawn_cells:
                            target_points.append(extra_cell)
            else:
                target_points.extend(reversed(row_cells))
                # Add one extra point at the beginning to ensure coverage
                if row_cells:
                    first_cell = row_cells[0]
                    if first_cell[0] - 1 >= 0:
                        extra_cell = (first_cell[0] - 1, first_cell[1])
                        if extra_cell in self.env.lawn_cells:
                            target_points.append(extra_cell)
            
            left_to_right = not left_to_right
        
        # Remove duplicates while preserving order
        target_points = list(dict.fromkeys(target_points))
        
        # Connect ALL points with COMPLETE A* paths
        full_path = []
        current = start_pos
        
        for target in target_points:
            if current == target:
                continue
            
            # Get COMPLETE A* path
            a_star_path = self._safe_a_star_with_full_path(current, target)
            
            if a_star_path and len(a_star_path) > 1:
                # Add EVERY step from A* path
                for i in range(1, len(a_star_path)):
                    from_pos = a_star_path[i-1]
                    next_pos = a_star_path[i]
                    
                    # Safety check
                    if not self._is_safe_diagonal(from_pos, next_pos):
                        print(f"    ⚠️  Unsafe diagonal: {from_pos} -> {next_pos}")
                        # Try to find alternative
                        alt_path = self._find_safe_alternative(from_pos, next_pos)
                        if alt_path:
                            for j in range(1, len(alt_path)):
                                alt_pos = alt_path[j]
                                full_path.append(alt_pos)
                                self._update_mowed_cells_along_move(from_pos, alt_pos)
                                from_pos = alt_pos
                            current = alt_pos
                        continue
                    
                    # Add step to path
                    full_path.append(next_pos)
                    self._update_mowed_cells_along_move(from_pos, next_pos)
                    current = next_pos
            else:
                print(f"    ⚠️  No A* path to target {target}")
        
        return full_path
    
    def _select_optimal_rows_with_overlap(self, sorted_rows, region):
        """
        Select rows with 2-cell spacing while ensuring complete coverage
        """
        if len(sorted_rows) <= 1:
            return sorted_rows
        
        optimal_rows = []
        i = 0
        
        while i < len(sorted_rows):
            current_y, current_cells = sorted_rows[i]
            
            # Always add current row
            optimal_rows.append(sorted_rows[i])
            
            # Try to skip next row if possible, but check if we'll miss coverage
            if i + 1 < len(sorted_rows):
                next_y, next_cells = sorted_rows[i + 1]
                
                # Check if we can skip this row with guaranteed complete coverage
                can_skip = self._can_skip_row_with_guaranteed_coverage(current_cells, next_cells, region)
                
                if can_skip:
                    # Check if there's another row after the skipped one
                    if i + 2 < len(sorted_rows):
                        next_next_y, next_next_cells = sorted_rows[i + 2]
                        # Verify we won't create a gap
                        gap_check = self._check_gap_coverage(next_cells, next_next_cells)
                        if gap_check:
                            i += 2  # Skip next row
                        else:
                            i += 1  # Can't skip due to gap risk
                    else:
                        i += 2  # No more rows after, safe to skip
                else:
                    i += 1  # Can't skip, use next row
            else:
                i += 1
        
        return optimal_rows
    
    def _can_skip_row_with_guaranteed_coverage(self, current_cells, next_cells, region):
        """
        Check if we can skip the next row with guaranteed 100% coverage
        """
        # Get Y coordinates
        current_y = current_cells[0][1] if current_cells else 0
        next_y = next_cells[0][1] if next_cells else 0
        
        # If rows are far apart, we can skip
        if next_y - current_y > 3:
            return True
        
        # Calculate guaranteed coverage from current row to next row
        current_set = set(current_cells)
        next_set = set(next_cells)
        
        # For each cell in next row, check if it's covered by current row
        covered_count = 0
        total_cells = len(next_cells)
        
        for next_cell in next_cells:
            x, y = next_cell
            covered = False
            
            # Check all positions that could cover this cell
            for dx in range(-self.mower_radius, self.mower_radius + 1):
                for dy in range(-self.mower_radius, self.mower_radius + 1):
                    if abs(dx) + abs(dy) <= self.mower_radius:
                        check_cell = (x + dx, y + dy)
                        if check_cell in current_set:
                            covered = True
                            break
                if covered:
                    break
            
            if covered:
                covered_count += 1
        
        # Only allow skipping if 100% coverage is guaranteed
        coverage_ratio = covered_count / max(total_cells, 1)
        return coverage_ratio >= 1.0
    
    def _check_gap_coverage(self, skipped_cells, following_cells):
        """
        Check if skipping a row creates a coverage gap
        """
        skipped_set = set(skipped_cells)
        following_set = set(following_cells)
        
        # Check if following row covers what was skipped
        for skipped_cell in skipped_cells:
            x, y = skipped_cell
            covered_by_following = False
            
            for dx in range(-self.mower_radius, self.mower_radius + 1):
                for dy in range(-self.mower_radius, self.mower_radius + 1):
                    if abs(dx) + abs(dy) <= self.mower_radius:
                        check_cell = (x + dx, y + dy)
                        if check_cell in following_set:
                            covered_by_following = True
                            break
                if covered_by_following:
                    break
            
            if not covered_by_following:
                return False
        
        return True
    
    def _safe_a_star_with_full_path(self, start, goal):
        """
        A* that returns COMPLETE path and avoids unsafe diagonal moves
        """
        if start == goal:
            return [start]
        
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
                # Reconstruct COMPLETE path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # Return complete path
            
            # Check neighbors - AVOID UNSAFE DIAGONALS
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                          (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Skip if not valid move
                if not self.env.is_valid_move(current, neighbor):
                    continue
                
                # CRITICAL: Check diagonal safety
                if dx != 0 and dy != 0:  # Diagonal move
                    if not self._is_safe_diagonal(current, neighbor):
                        continue  # Skip unsafe diagonal
                
                # Cost (penalize diagonals)
                cost = 1.0 if dx == 0 or dy == 0 else 1.5  # Penalty for diagonals
                
                tentative_g = g_score[current] + cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # No path found

    def _is_safe_diagonal(self, from_pos, to_pos):
        """
        STRICT diagonal safety check
        Returns False if diagonal move touches an obstacle corner
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        if dx == 0 or dy == 0:
            return True  # Cardinal moves are always safe
        
        # For diagonal moves, check the corner cells
        # The dangerous cells are the ones adjacent to BOTH from_pos and to_pos
        
        # Check ALL relevant corner cells
        corner_cells = [
            # Cells that the mower "cuts across"
            (from_pos[0] + dx, from_pos[1]),      # Corner 1
            (from_pos[0], from_pos[1] + dy),      # Corner 2
            
            # Cells that might be obstacles further along the path
            (from_pos[0] + 2*dx, from_pos[1]),    # Corner 3
            (from_pos[0], from_pos[1] + 2*dy),    # Corner 4
            
            # The diagonal cell itself
            (from_pos[0] + dx, from_pos[1] + dy), # to_pos
            
            # Additional cells for extended blade coverage
            (from_pos[0] + dx, from_pos[1] + 2*dy),
            (from_pos[0] + 2*dx, from_pos[1] + dy),
        ]
        
        for cell in corner_cells:
            if cell in self.env.obstacle_cells:
                return False
        
        # Also check if mower blade would hit obstacles at intermediate positions
        # Check blade coverage at from_pos and to_pos
        for pos in [from_pos, to_pos]:
            blade = self.env.get_blade_coverage(pos)
            for cell in blade:
                if cell in self.env.obstacle_cells:
                    # Check if it's a corner cell
                    dist_x = abs(cell[0] - pos[0])
                    dist_y = abs(cell[1] - pos[1])
                    if dist_x == 1 and dist_y == 1:
                        return False  # Corner cell of blade hits obstacle
        
        return True
    
    def _find_safe_alternative(self, start, goal):
        """
        Find alternative path when direct move is unsafe
        """
        # Try L-shaped path (two cardinal moves)
        alt_paths = []
        
        # Try horizontal then vertical
        intermediate1 = (start[0], goal[1])
        if (self.env.is_valid_move(start, intermediate1) and 
            self.env.is_valid_move(intermediate1, goal)):
            alt_paths.append([start, intermediate1, goal])
        
        # Try vertical then horizontal
        intermediate2 = (goal[0], start[1])
        if (self.env.is_valid_move(start, intermediate2) and 
            self.env.is_valid_move(intermediate2, goal)):
            alt_paths.append([start, intermediate2, goal])
        
        # Return shortest alternative
        if alt_paths:
            return min(alt_paths, key=len)
        
        return None
    
    def _find_best_entry(self, current_pos, region):
        """Find best entry point to region"""
        best_entry = None
        best_dist = float('inf')
        
        for entry in region.entry_points:
            if not self.env.is_valid_position(entry):
                continue
            
            # Prefer entries that don't require diagonal moves
            dx = abs(current_pos[0] - entry[0])
            dy = abs(current_pos[1] - entry[1])
            dist = dx + dy
            
            # Penalize if diagonal would be required
            if dx > 0 and dy > 0:
                dist += 1  # Penalty for diagonal
            
            if dist < best_dist:
                best_dist = dist
                best_entry = entry
        
        return best_entry
    
    def _heuristic(self, a, b):
        """Manhattan distance heuristic"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _update_mowed_cells_along_move(self, from_pos, to_pos):
        """
        Update mowed cells for ALL positions along a move
        This ensures diagonal moves properly cover intermediate cells
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        # Mow at starting position
        self._update_mowed_cells(from_pos)
        
        # For diagonal moves, mow the intermediate corner cells
        if dx != 0 and dy != 0:
            # Mow the two corner positions that are crossed during diagonal move
            corner1 = (from_pos[0] + dx, from_pos[1])  # Horizontal step
            corner2 = (from_pos[0], from_pos[1] + dy)  # Vertical step
            
            # These cells are physically crossed during a diagonal move
            self._update_mowed_cells(corner1)
            self._update_mowed_cells(corner2)
            
            # Also mow cells adjacent to the diagonal path for blade coverage
            adjacent_cells = [
                (from_pos[0] + dx, from_pos[1] + dy),  # to_pos
                (from_pos[0], from_pos[1]),  # from_pos (already done, but keep for clarity)
            ]
            
            for cell in adjacent_cells:
                self._update_mowed_cells(cell)
        
        # Mow at ending position
        self._update_mowed_cells(to_pos)
    
    def _update_mowed_cells(self, pos):
        """Update mowed cells at a single position"""
        blade = self.env.get_blade_coverage(pos)
        for cell in blade:
            if cell in self.env.reachable_lawn_cells:
                self.mowed_cells.add(cell)
    
    def _verify_full_path(self, path):
        """Verify path thoroughly"""
        if not path:
            return
        
        print(f"Verifying {len(path)}-step path...")
        
        # Simulate the entire path to ensure all cells are mowed
        simulation_mowed = set()
        current_pos = path[0] if path else None
        
        for i in range(1, len(path)):
            from_pos = path[i-1]
            to_pos = path[i]
            
            # Simulate the move
            blade_start = self.env.get_blade_coverage(from_pos)
            blade_end = self.env.get_blade_coverage(to_pos)
            
            for cell in blade_start:
                if cell in self.env.reachable_lawn_cells:
                    simulation_mowed.add(cell)
            
            for cell in blade_end:
                if cell in self.env.reachable_lawn_cells:
                    simulation_mowed.add(cell)
            
            # For diagonal moves, add intermediate coverage
            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]
            
            if dx != 0 and dy != 0:
                # Add the corners crossed during diagonal move
                corner1 = (from_pos[0] + dx, from_pos[1])
                corner2 = (from_pos[0], from_pos[1] + dy)
                
                for corner in [corner1, corner2]:
                    blade_corner = self.env.get_blade_coverage(corner)
                    for cell in blade_corner:
                        if cell in self.env.reachable_lawn_cells:
                            simulation_mowed.add(cell)
            
            current_pos = to_pos
        
        # Check for A* path jumping
        print(f"\nChecking for A* path jumps...")
        jumps = 0
        for i in range(1, len(path)):
            from_pos = path[i-1]
            to_pos = path[i]
            dx = abs(to_pos[0] - from_pos[0])
            dy = abs(to_pos[1] - from_pos[1])
            
            # If move is more than 1 cell in any direction, it's a jump
            if dx > 1 or dy > 1:
                jumps += 1
                print(f"  ⚠️  A* PATH JUMP at step {i}: {from_pos} -> {to_pos} (distance: {dx},{dy})")
        
        # Calculate coverage
        total = len(self.env.reachable_lawn_cells)
        coverage = 100 * len(self.mowed_cells) / total if total > 0 else 0
        simulation_coverage = 100 * len(simulation_mowed) / total if total > 0 else 0
        
        print(f"\nRESULTS:")
        print(f"  Path length: {len(path)} steps")
        print(f"  A* path jumps: {jumps}")
        print(f"  Tracked coverage: {coverage:.1f}% ({len(self.mowed_cells)}/{total})")
        print(f"  Simulated coverage: {simulation_coverage:.1f}% ({len(simulation_mowed)}/{total})")
        print(f"  Efficiency: {100 * len(self.mowed_cells) / len(path):.1f}%" if len(path) > 0 else "  Efficiency: 0.0%")
        
        # Check for missing coverage
        missing_cells = self.env.reachable_lawn_cells - self.mowed_cells
        if missing_cells:
            print(f"\n⚠️  WARNING: {len(missing_cells)} cells not covered!")
            # Show first few missing cells
            for i, cell in enumerate(list(missing_cells)[:10]):
                print(f"    {i+1}. {cell}")
            if len(missing_cells) > 10:
                print(f"    ... and {len(missing_cells) - 10} more")
        
        if jumps > 0:
            print(f"\n⚠️  WARNING: Found {jumps} A* path jumps!")
            print("   The algorithm is not following complete A* paths.")
        
        if len(self.unsafe_moves) > 0:
            print(f"\n⚠️  WARNING: Found {len(self.unsafe_moves)} unsafe diagonal moves!")
    
    def visualize_side_by_side(self, path=None):
        """Show visualization"""
        if not self.show_visualization:
            return
        
        if path is None:
            path = self.path
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Path with move analysis
        for obs in self.env.obstacle_cells:
            x, y = obs
            ax1.add_patch(plt.Rectangle((x, y), 1, 1, color='black', alpha=0.7))
        
        # Draw path with safety coloring
        for i in range(1, len(path)):
            from_pos = path[i-1]
            to_pos = path[i]
            
            # Check move type and safety
            dx = abs(to_pos[0] - from_pos[0])
            dy = abs(to_pos[1] - from_pos[1])
            
            if dx > 1 or dy > 1:
                color = 'purple'  # A* jump
                width = 3
            elif dx == 1 and dy == 1:
                # Diagonal move
                if (from_pos, to_pos) in self.unsafe_moves:
                    color = 'red'  # Unsafe diagonal
                    width = 3
                else:
                    color = 'orange'  # Safe diagonal
                    width = 2
            else:
                color = 'green'  # Cardinal move
                width = 1
            
            ax1.plot([from_pos[0] + 0.5, to_pos[0] + 0.5],
                    [from_pos[1] + 0.5, to_pos[1] + 0.5],
                    color=color, linewidth=width, alpha=0.7)
        
        ax1.set_title(f"Path Analysis\nGreen=Cardinal, Orange=Safe Diagonal\nRed=Unsafe, Purple=A* Jump")
        ax1.set_aspect('equal')
        ax1.set_xlim(-1, self.env.width + 1)
        ax1.set_ylim(-1, self.env.height + 1)
        ax1.invert_yaxis()
        
        # Plot 2: Coverage results
        for obs in self.env.obstacle_cells:
            x, y = obs
            ax2.add_patch(plt.Rectangle((x, y), 1, 1, color='black', alpha=0.7))
        
        unmowed = self.env.reachable_lawn_cells - self.mowed_cells
        for cell in unmowed:
            x, y = cell
            ax2.add_patch(plt.Rectangle((x, y), 1, 1, color='yellow', alpha=0.5))
        
        for cell in self.mowed_cells:
            x, y = cell
            ax2.add_patch(plt.Rectangle((x, y), 1, 1, color='green', alpha=0.8))
        
        coverage = 100 * len(self.mowed_cells) / len(self.env.reachable_lawn_cells)
        ax2.set_title(f"Coverage: {coverage:.1f}%")
        ax2.set_aspect('equal')
        ax2.set_xlim(-1, self.env.width + 1)
        ax2.set_ylim(-1, self.env.height + 1)
        ax2.invert_yaxis()
        
        # Plot 3: Row spacing visualization
        for obs in self.env.obstacle_cells:
            x, y = obs
            ax3.add_patch(plt.Rectangle((x, y), 1, 1, color='black', alpha=0.3))
        
        # Show which rows were selected
        for region in self.regions:
            rows_used = set()
            for cell in self.visited_positions:
                if cell in region.cells:
                    rows_used.add(cell[1])
            
            for y in rows_used:
                ax3.add_patch(plt.Rectangle((0, y), self.env.width, 1, 
                                          color='blue', alpha=0.1))
        
        ax3.set_title(f"Row Spacing\nBlue=Rows used for coverage")
        ax3.set_aspect('equal')
        ax3.set_xlim(-1, self.env.width + 1)
        ax3.set_ylim(-1, self.env.height + 1)
        ax3.invert_yaxis()
        
        plt.tight_layout()
        plt.show()


# Generator function
def ccpp_algorithm(env, show_visualization=True):
    """Generator interface"""
    planner = CompleteCoveragePlanner(env, show_visualization=show_visualization)
    path = planner.plan_full_coverage()
    
    if show_visualization:
        planner.visualize_side_by_side(path)
    
    for pos in path:
        yield pos