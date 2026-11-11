"""
Complete Coverage Path Planning - WORKING VERSION
"""
import numpy as np
from typing import List, Tuple, Set, Dict
from collections import defaultdict, deque
import heapq


class WorkingCoveragePlanner:
    """
    Simple but effective coverage planner that actually works
    """
    
    def __init__(self, env):
        self.env = env
        self.width = env.width
        self.height = env.height
        self.reachable = env.reachable_lawn_cells
        
        print(f"WorkingCoveragePlanner initialized")
        print(f"  Reachable cells: {len(self.reachable)}")
        print(f"  Grid: {self.width}x{self.height}")
    
    def plan_coverage(self) -> List[Tuple[int, int]]:
        """
        Generate complete coverage path using simple boustrophedon
        """
        print("\nGenerating coverage path...")
        
        # Step 1: Group cells by row for boustrophedon
        rows = defaultdict(list)
        for cell in self.reachable:
            x, y = cell
            rows[y].append(cell)
        
        # Step 2: Sort rows and cells within rows
        sorted_rows = sorted(rows.keys())
        print(f"  Found {len(sorted_rows)} rows with reachable cells")
        
        # Step 3: Generate boustrophedon path
        path = []
        direction = 1  # 1 = left to right, -1 = right to left
        
        for i, y in enumerate(sorted_rows):
            # Get cells in this row, sorted by x
            cells_in_row = sorted(rows[y], key=lambda c: c[0])
            
            if direction == 1:
                # Left to right
                path.extend(cells_in_row)
            else:
                # Right to left
                path.extend(reversed(cells_in_row))
            
            # Change direction for next row
            direction *= -1
            
            # Add transition to next row if not last row
            if i < len(sorted_rows) - 1:
                next_y = sorted_rows[i + 1]
                
                # Find best transition point
                if direction == 1:  # Next row will be L→R
                    # Current row end → next row start
                    current_end = cells_in_row[-1] if direction == 1 else cells_in_row[0]
                    next_cells = sorted(rows[next_y], key=lambda c: c[0])
                    next_start = next_cells[0]
                    
                    # Find path between rows
                    transition = self._find_transition_path(current_end, next_start)
                    path.extend(transition[1:])  # Skip first (already in path)
                else:  # Next row will be R→L
                    # Current row end → next row start (from right)
                    current_end = cells_in_row[0] if direction == -1 else cells_in_row[-1]
                    next_cells = sorted(rows[next_y], key=lambda c: c[0])
                    next_start = next_cells[-1]
                    
                    # Find path between rows
                    transition = self._find_transition_path(current_end, next_start)
                    path.extend(transition[1:])
        
        # Remove duplicates while preserving order
        unique_path = []
        seen = set()
        for cell in path:
            if cell not in seen:
                unique_path.append(cell)
                seen.add(cell)
        
        # Verify all reachable cells are covered
        covered = set(unique_path) & self.reachable
        coverage = 100 * len(covered) / len(self.reachable) if self.reachable else 0
        
        print(f"\nPath generated:")
        print(f"  Path length: {len(unique_path)}")
        print(f"  Coverage: {coverage:.1f}%")
        print(f"  Missing cells: {len(self.reachable) - len(covered)}")
        
        return unique_path
    
    def _find_transition_path(self, start: Tuple[int, int], 
                              end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Find shortest path between two points using BFS
        """
        if start == end:
            return [start]
        
        queue = deque([start])
        came_from = {start: None}
        
        while queue:
            current = queue.popleft()
            
            if current == end:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            
            # Check all 8 directions
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                          (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if (neighbor not in came_from and 
                    self._is_valid_cell(neighbor)):
                    came_from[neighbor] = current
                    queue.append(neighbor)
        
        # If no path found, return direct line (might have obstacles)
        return [start, end]
    
    def _is_valid_cell(self, cell: Tuple[int, int]) -> bool:
        """Check if cell is valid for movement"""
        x, y = cell
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        
        # Check if it's a lawn cell or at least not an obstacle
        return cell in self.env.lawn_cells or cell not in self.env.obstacle_cells
    
    def verify_coverage(self, path: List[Tuple[int, int]]) -> Dict:
        """Verify path coverage"""
        covered = set(path) & self.reachable
        coverage = 100 * len(covered) / len(self.reachable) if self.reachable else 0
        efficiency = 100 * len(covered) / len(path) if path else 0
        
        return {
            'coverage': coverage,
            'efficiency': efficiency,
            'covered_cells': len(covered),
            'total_reachable': len(self.reachable),
            'path_length': len(path)
        }


# SIMPLE VERSION - Even simpler but guaranteed to work
def simple_boustrophedon_coverage(env):
    """
    Ultra-simple boustrophedon coverage that just works
    """
    # Get all reachable cells
    reachable = list(env.reachable_lawn_cells)
    
    if not reachable:
        return []
    
    # Sort by y (row), then x (column)
    reachable.sort(key=lambda c: (c[1], c[0]))
    
    # Group by row
    rows = {}
    for x, y in reachable:
        if y not in rows:
            rows[y] = []
        rows[y].append((x, y))
    
    # Sort rows
    sorted_rows = sorted(rows.keys())
    path = []
    
    # Alternate direction each row
    left_to_right = True
    
    for y in sorted_rows:
        row_cells = sorted(rows[y], key=lambda c: c[0])  # Sort by x
        
        if left_to_right:
            path.extend(row_cells)
        else:
            path.extend(reversed(row_cells))
        
        left_to_right = not left_to_right
    
    return path


# EVEN SIMPLER - Grid-based boustrophedon
def grid_boustrophedon(env):
    """
    Grid-based boustrophedon that covers every cell systematically
    """
    path = []
    
    # Get min/max coordinates of reachable area
    xs = [c[0] for c in env.reachable_lawn_cells]
    ys = [c[1] for c in env.reachable_lawn_cells]
    
    if not xs or not ys:
        return []
    
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    
    print(f"Coverage area: x={x_min}-{x_max}, y={y_min}-{y_max}")
    
    # Simple row-by-row boustrophedon
    left_to_right = True
    
    for y in range(y_min, y_max + 1):
        # Collect reachable cells in this row
        row_cells = []
        for x in range(x_min, x_max + 1):
            cell = (x, y)
            if cell in env.reachable_lawn_cells:
                row_cells.append(cell)
        
        if not row_cells:
            continue
        
        # Add cells in appropriate direction
        if left_to_right:
            path.extend(row_cells)
        else:
            path.extend(reversed(row_cells))
        
        left_to_right = not left_to_right
    
    return path


# Try this in your main.py instead:
def run_simple_coverage(env, visualize=True):
    """
    Run the simple boustrophedon algorithm
    """
    print("\n" + "="*60)
    print("SIMPLE BOUSTROPHEDON COVERAGE")
    print("="*60)
    
    # Use the simple version
    path = grid_boustrophedon(env)
    
    # Calculate metrics
    covered = set(path) & env.reachable_lawn_cells
    total_reachable = len(env.reachable_lawn_cells)
    
    metrics = {
        'steps': len(path),
        'coverage': 100 * len(covered) / total_reachable if total_reachable else 0,
        'efficiency': 100 * len(covered) / len(path) if path else 0,
    }
    
    print(f"\nResults:")
    print(f"  Steps: {metrics['steps']}")
    print(f"  Coverage: {metrics['coverage']:.1f}%")
    print(f"  Efficiency: {metrics['efficiency']:.1f}%")
    print(f"  Covered: {len(covered)}/{total_reachable} cells")
    
    if metrics['coverage'] < 100:
        missing = total_reachable - len(covered)
        print(f"  WARNING: {missing} cells not covered!")
    
    return path, metrics