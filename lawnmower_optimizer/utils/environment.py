"""
Updated Environment with proper reachable lawn calculation
"""
import numpy as np
from typing import Tuple, List, Set, Optional
from dataclasses import dataclass


@dataclass
class MowerState:
    position: Tuple[int, int]
    heading: int
    mowed_cells: Set[Tuple[int, int]]
    total_steps: int = 0
    direction_changes: int = 0


class LawnmowerEnvironment:
    """
    Environment that properly tracks reachable lawn cells
    """
    
    def _get_directions(self):
        return {
            0: (1, 0),      # E
            45: (1, -1),    # NE  
            90: (0, -1),    # N
            135: (-1, -1),  # NW
            180: (-1, 0),   # W
            225: (-1, 1),   # SW
            270: (0, 1),    # S
            315: (1, 1)     # SE
        }
    
    @property
    def DIRECTIONS(self):
        return self._get_directions()
    
    def __init__(self, grid_file: str, blade_diameter: float = 70.0):
        self.blade_diameter = blade_diameter
        self.grid_square_size = blade_diameter / 2.0
        
        self.grid = self._load_and_convert_to_fine_grid(grid_file)
        self.height, self.width = self.grid.shape
        
        # All lawn cells from the grid
        self.all_lawn_cells = self._find_cells('L')
        self.obstacle_cells = self._find_cells('+')
        
        # Valid mower positions (where mower can be placed)
        self.lawn_cells = self._calculate_valid_mower_positions()
        
        # REACHABLE lawn cells (cells that can actually be mowed)
        self.reachable_lawn_cells = self._calculate_reachable_lawn_cells()
        
        self.start_pos = self._find_start()
        
        print(f"Grid size: {self.width}x{self.height}")
        print(f"Total lawn cells in map: {len(self.all_lawn_cells)}")
        print(f"Reachable lawn cells (mowable): {len(self.reachable_lawn_cells)}")
        print(f"Valid mower positions: {len(self.lawn_cells)}")
        print(f"Start position: {self.start_pos}")
        
        if len(self.reachable_lawn_cells) < len(self.all_lawn_cells):
            unreachable = len(self.all_lawn_cells) - len(self.reachable_lawn_cells)
            print(f"Note: {unreachable} lawn cells are NOT reachable due to mower shape constraints")
    
    def _calculate_reachable_lawn_cells(self) -> Set[Tuple[int, int]]:
        """
        Calculate which lawn cells can actually be mowed by the mower.
        A lawn cell is reachable if it can be covered by the mower's blade
        from at least one valid mower position.
        """
        reachable = set()
        
        for mower_pos in self.lawn_cells:
            blade_cells = self.get_blade_coverage(mower_pos)
            for cell in blade_cells:
                if cell in self.all_lawn_cells:
                    reachable.add(cell)
        
        return reachable
    
    def _calculate_valid_mower_positions(self) -> Set[Tuple[int, int]]:
        """Calculate where the mower can be positioned"""
        valid = set()
        excluded_count = 0
        
        for x in range(self.width):
            for y in range(self.height):
                pos = (x, y)
                
                # Skip obstacle cells
                if pos in self.obstacle_cells:
                    continue
                
                # Check blade coverage doesn't hit obstacles (including diagonal corners)
                blade = self.get_blade_partial_coverage(pos)  # Changed from get_blade_coverage
                valid_position = True
                
                for cell in blade:
                    # Check bounds
                    cx, cy = cell
                    if cx < 0 or cx >= self.width or cy < 0 or cy >= self.height:
                        valid_position = False
                        break
                    
                    # Check if cell is obstacle
                    if cell in self.obstacle_cells:
                        valid_position = False
                        excluded_count += 1
                        break
                
                if valid_position:
                    valid.add(pos)
        
        print(f"  _calculate_valid_mower_positions: excluded {excluded_count} positions due to blade hitting obstacles")
        return valid
    
    def _load_grid(self, filename: str) -> np.ndarray:
        with open(filename, 'r') as f:
            lines = [line.rstrip('\n') for line in f.readlines()]
        
        max_width = max(len(line) for line in lines)
        grid = np.array([list(line.ljust(max_width)) for line in lines])
        return grid
    
    def _load_and_convert_to_fine_grid(self, grid_file: str) -> np.ndarray:
        coarse_grid = self._load_grid(grid_file)
        coarse_height, coarse_width = coarse_grid.shape
        
        fine_height = coarse_height * 2
        fine_width = coarse_width * 2
        fine_grid = np.full((fine_height, fine_width), ' ', dtype=str)
        
        for i in range(coarse_height):
            for j in range(coarse_width):
                cell_type = coarse_grid[i, j]
                fine_i = i * 2
                fine_j = j * 2
                
                for di in range(2):
                    for dj in range(2):
                        fine_grid[fine_i + di, fine_j + dj] = cell_type
        
        return fine_grid
    
    def _find_cells(self, symbol: str) -> Set[Tuple[int, int]]:
        positions = np.where(self.grid == symbol)
        cells = set(zip(positions[1].tolist(), positions[0].tolist()))
        return cells
    
    def _find_start(self) -> Tuple[int, int]:
        start_cells = self._find_cells('S')
        if start_cells:
            pos = list(start_cells)[0]
            if pos in self.lawn_cells:
                return pos
        
        # Find center of valid positions
        if self.lawn_cells:
            # Find position closest to center of reachable area
            if self.reachable_lawn_cells:
                center_x = sum(c[0] for c in self.reachable_lawn_cells) / len(self.reachable_lawn_cells)
                center_y = sum(c[1] for c in self.reachable_lawn_cells) / len(self.reachable_lawn_cells)
                best_pos = min(self.lawn_cells, key=lambda p: (p[0] - center_x)**2 + (p[1] - center_y)**2)
                return best_pos
            return list(self.lawn_cells)[0]
        
        return (0, 0)
    
    def create_state(self) -> MowerState:
        return MowerState(
            position=self.start_pos,
            heading=0,
            mowed_cells=set(),
            total_steps=0,
            direction_changes=0
        )
    
    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        x, y = pos
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return (x, y) not in self.obstacle_cells
    
    def get_blade_coverage(self, center_pos: Tuple[int, int]) -> Set[Tuple[int, int]]:
        x, y = center_pos
        covered = set()
        
        # Blade covers: center, left, right, up, down
        for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            cell = (x + dx, y + dy)
            covered.add(cell)
        
        return covered
    
    def get_blade_partial_coverage(self, center_pos: Tuple[int, int]) -> Set[Tuple[int, int]]:
        x, y = center_pos
        covered = set()
        
        # Blade fully covers: center, left, right, up, down
        # Blade partially covers: diagonals (right, up), (right, down), (left, up), (left, down)
        for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            cell = (x + dx, y + dy)
            covered.add(cell)
        
        return covered
    
    def is_valid_move(self, from_pos: Tuple[int, int], 
                     to_pos: Tuple[int, int]) -> bool:
        # Check basic position validity
        if not self.is_valid_position(from_pos):
            return False
        
        if not self.is_valid_position(to_pos):
            return False
        
        # Check move distance
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        if abs(dx) > 1 or abs(dy) > 1:
            return False
        
        if dx == 0 and dy == 0:
            return False
        
        # Check blade coverage at destination
        blade = self.get_blade_partial_coverage(to_pos)
        
        for cell in blade:
            # Check bounds
            cx, cy = cell
            if cx < 0 or cx >= self.width or cy < 0 or cy >= self.height:
                return False
            
            # Check obstacle
            if cell in self.obstacle_cells:
                return False
        
        # For diagonal moves: check adjacent cells
        if dx != 0 and dy != 0:
            cell1 = (from_pos[0] + dx, from_pos[1])
            cell2 = (from_pos[0], from_pos[1] + dy)
            
            if not self.is_valid_position(cell1):
                return False
            
            if not self.is_valid_position(cell2):
                return False
        
        return True
    
    def get_neighbors(self, pos: Tuple[int, int], 
                     valid_only: bool = True) -> List[Tuple[int, int]]:
        x, y = pos
        neighbors = []
        
        for dx, dy in self.DIRECTIONS.values():
            neighbor = (x + dx, y + dy)
            
            if not valid_only:
                neighbors.append(neighbor)
            elif self.is_valid_move(pos, neighbor):
                neighbors.append(neighbor)
        
        return neighbors
    
    def move(self, state: MowerState, 
            new_pos: Tuple[int, int]) -> MowerState:
        
        if not self.is_valid_move(state.position, new_pos):
            raise ValueError(f"Invalid move: {state.position} → {new_pos}")
        
        # Calculate heading
        dx = new_pos[0] - state.position[0]
        dy = new_pos[1] - state.position[1]
        
        new_heading = state.heading
        for heading, (dir_dx, dir_dy) in self.DIRECTIONS.items():
            if dir_dx == dx and dir_dy == dy:
                new_heading = heading
                break
        
        # Update mowed cells
        new_mowed = state.mowed_cells.copy()
        
        # Cover both positions
        for pos in [state.position, new_pos]:
            blade = self.get_blade_coverage(pos)
            for cell in blade:
                if cell in self.all_lawn_cells:
                    new_mowed.add(cell)
        
        # For diagonal moves
        if dx != 0 and dy != 0:
            intermediate_positions = [
                (state.position[0] + dx, state.position[1]),
                (state.position[0], state.position[1] + dy)
            ]
            
            for pos in intermediate_positions:
                blade = self.get_blade_coverage(pos)
                for cell in blade:
                    if cell in self.all_lawn_cells:
                        new_mowed.add(cell)
        
        # Count direction changes
        direction_changes = state.direction_changes
        if new_heading != state.heading:
            direction_changes += 1
        
        return MowerState(
            position=new_pos,
            heading=new_heading,
            mowed_cells=new_mowed,
            total_steps=state.total_steps + 1,
            direction_changes=direction_changes
        )
    
    def calculate_efficiency(self, state: MowerState) -> float:
        if state.total_steps == 0:
            return 100.0
        # Efficiency based on reachable cells, not all lawn cells
        mowed_reachable = len(state.mowed_cells.intersection(self.reachable_lawn_cells))
        return 100.0 * mowed_reachable / max(state.total_steps, 1)
    
    def is_complete(self, state: MowerState) -> bool:
        """Task is complete when all REACHABLE lawn cells are mowed"""
        # Only check reachable cells
        mowed_reachable = len(state.mowed_cells.intersection(self.reachable_lawn_cells))
        return mowed_reachable >= len(self.reachable_lawn_cells)
    
    def get_unmowed_cells(self, state: MowerState) -> Set[Tuple[int, int]]:
        """Get all REACHABLE lawn cells that haven't been mowed yet"""
        return self.reachable_lawn_cells - state.mowed_cells
    
    def get_mowable_cells(self) -> Set[Tuple[int, int]]:
        """Public method to get reachable lawn cells"""
        return self.reachable_lawn_cells
    
    def calculate_progress(self, state: MowerState) -> float:
        """Calculate progress based on reachable cells"""
        mowed_reachable = len(state.mowed_cells.intersection(self.reachable_lawn_cells))
        return mowed_reachable / max(len(self.reachable_lawn_cells), 1)