"""
Map Converter - Convert images or real maps to mowing grids

Supports:
- Image files (PNG, JPG) -> grid conversion
- Manual grid creation
- Future: Integration with mapping APIs
"""
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from typing import Tuple, List


def image_to_grid(image_path: str, 
                 output_path: str,
                 grid_size: Tuple[int, int] = (30, 30),
                 obstacle_threshold: int = 128,
                 invert: bool = False):
    """
    Convert an image to a lawn mowing grid
    
    Args:
        image_path: Path to input image
        output_path: Path to save grid file
        grid_size: Target grid dimensions (width, height)
        obstacle_threshold: Pixel brightness threshold (0-255)
                          Darker pixels -> obstacles
                          Brighter pixels -> lawn
        invert: Invert the threshold (bright=obstacle, dark=lawn)
    
    Process:
    1. Load image and convert to grayscale
    2. Resize to grid dimensions
    3. Threshold: bright pixels = lawn, dark = obstacles
    4. Save as text grid file
    """
    # Load and process image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize(grid_size, Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    arr = np.array(img)
    
    # Create grid based on threshold
    grid = np.zeros(grid_size, dtype=str)
    
    for y in range(grid_size[1]):
        for x in range(grid_size[0]):
            pixel_value = arr[y, x]
            
            # Determine cell type based on brightness
            if invert:
                is_obstacle = pixel_value > obstacle_threshold
            else:
                is_obstacle = pixel_value < obstacle_threshold
            
            if is_obstacle:
                grid[y, x] = '+'
            else:
                grid[y, x] = 'L'
    
    # Add border of obstacles
    grid = _add_border(grid)
    
    # Save to file
    _save_grid(grid, output_path)
    
    print(f"Grid created: {output_path}")
    print(f"Dimensions: {grid.shape[1]} x {grid.shape[0]}")
    print(f"Lawn cells: {np.sum(grid == 'L')}")
    print(f"Obstacles: {np.sum(grid == '+')}")


def create_grid_from_polygon(vertices: List[Tuple[int, int]],
                             grid_size: Tuple[int, int],
                             output_path: str):
    """
    Create a grid where lawn is defined by a polygon
    
    Args:
        vertices: List of (x, y) vertices defining lawn boundary
        grid_size: Grid dimensions (width, height)
        output_path: Where to save grid file
    """
    grid = np.full(grid_size[::-1], '+', dtype=str)  # All obstacles initially
    
    # Fill polygon with lawn
    from matplotlib.path import Path
    
    # Create all grid points
    y_coords, x_coords = np.meshgrid(np.arange(grid_size[1]), 
                                     np.arange(grid_size[0]), 
                                     indexing='ij')
    points = np.column_stack([x_coords.ravel(), y_coords.ravel()])
    
    # Check which points are inside polygon
    path = Path(vertices)
    mask = path.contains_points(points)
    mask = mask.reshape(grid_size[1], grid_size[0])
    
    # Set lawn cells
    grid[mask] = 'L'
    
    # Add border
    grid = _add_border(grid)
    
    # Save
    _save_grid(grid, output_path)
    print(f"Polygon grid created: {output_path}")


def create_random_obstacles(grid_size: Tuple[int, int],
                           obstacle_density: float = 0.2,
                           output_path: str = None) -> np.ndarray:
    """
    Create a random grid with obstacles
    
    Args:
        grid_size: Grid dimensions (width, height)
        obstacle_density: Fraction of cells that are obstacles (0-1)
        output_path: If provided, save to file
    
    Returns:
        Grid array
    """
    grid = np.full(grid_size[::-1], 'L', dtype=str)
    
    # Add random obstacles
    n_obstacles = int(grid_size[0] * grid_size[1] * obstacle_density)
    for _ in range(n_obstacles):
        x = np.random.randint(1, grid_size[0] - 1)
        y = np.random.randint(1, grid_size[1] - 1)
        grid[y, x] = '+'
    
    # Add border
    grid = _add_border(grid)
    
    if output_path:
        _save_grid(grid, output_path)
        print(f"Random grid created: {output_path}")
    
    return grid


def download_satellite_map(latitude: float, 
                          longitude: float,
                          zoom: int = 18,
                          grid_size: Tuple[int, int] = (50, 50),
                          output_path: str = None):
    """
    Download satellite imagery and convert to grid
    
    NOTE: This is a placeholder. In production, you would:
    1. Use Google Maps API, Mapbox, or OpenStreetMap
    2. Authenticate with API keys
    3. Download satellite/terrain imagery
    4. Apply image segmentation to identify lawn areas
    
    Args:
        latitude: Center latitude
        longitude: Center longitude  
        zoom: Zoom level (higher = more detail)
        grid_size: Target grid size
        output_path: Where to save grid
    """
    print("Satellite map download requires API integration.")
    print("Placeholder implementation:")
    print(f"  Location: ({latitude}, {longitude})")
    print(f"  Zoom: {zoom}")
    print(f"  Would download satellite imagery and segment lawn areas")
    print()
    print("For now, creating a sample grid...")
    
    # Create sample grid instead
    create_random_obstacles(grid_size, obstacle_density=0.15, output_path=output_path)


def _add_border(grid: np.ndarray) -> np.ndarray:
    """Add border of obstacles around grid"""
    bordered = np.full((grid.shape[0] + 2, grid.shape[1] + 2), '+', dtype=str)
    bordered[1:-1, 1:-1] = grid
    return bordered


def _save_grid(grid: np.ndarray, filepath: str):
    """Save grid array to text file"""
    with open(filepath, 'w') as f:
        for row in grid:
            f.write(''.join(row) + '\n')


# Example usage
if __name__ == '__main__':
    from typing import List, Tuple
    
    # Example 1: Random grid
    create_random_obstacles(
        grid_size=(30, 30),
        obstacle_density=0.2,
        output_path='maps/random_map.txt'
    )
    
    # Example 2: Polygon lawn
    # Define a lawn shaped like a house yard
    vertices = [
        (5, 5), (25, 5), (25, 25), (15, 25), (15, 15), (5, 15)
    ]
    create_grid_from_polygon(
        vertices=vertices,
        grid_size=(30, 30),
        output_path='maps/polygon_map.txt'
    )
    
    print("Sample maps created!")
