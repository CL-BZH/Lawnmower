#!/usr/bin/env python3
"""
Simple map creator - Generate custom lawn mowing maps
"""
import argparse
from utils.map_converter import (
    create_random_obstacles,
    create_grid_from_polygon,
    image_to_grid
)


def main():
    parser = argparse.ArgumentParser(
        description='Create custom lawn mowing maps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create random 30x30 map with 20% obstacles
  python create_map.py random --size 30 30 --density 0.2 --output maps/random.txt
  
  # Create rectangular lawn
  python create_map.py rectangle --size 40 30 --output maps/rectangle.txt
  
  # Convert image to grid
  python create_map.py image --input yard.png --size 50 50 --output maps/yard.txt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Map creation mode')
    
    # Random map
    random_parser = subparsers.add_parser('random', help='Create random obstacle map')
    random_parser.add_argument('--size', type=int, nargs=2, required=True,
                              metavar=('WIDTH', 'HEIGHT'),
                              help='Grid size (width height)')
    random_parser.add_argument('--density', type=float, default=0.2,
                              help='Obstacle density (0.0-1.0)')
    random_parser.add_argument('--output', '-o', type=str, required=True,
                              help='Output file path')
    
    # Rectangle map
    rect_parser = subparsers.add_parser('rectangle', help='Create rectangular lawn')
    rect_parser.add_argument('--size', type=int, nargs=2, required=True,
                            metavar=('WIDTH', 'HEIGHT'),
                            help='Grid size (width height)')
    rect_parser.add_argument('--output', '-o', type=str, required=True,
                            help='Output file path')
    
    # Image conversion
    image_parser = subparsers.add_parser('image', help='Convert image to grid')
    image_parser.add_argument('--input', '-i', type=str, required=True,
                             help='Input image file')
    image_parser.add_argument('--size', type=int, nargs=2, required=True,
                             metavar=('WIDTH', 'HEIGHT'),
                             help='Grid size (width height)')
    image_parser.add_argument('--threshold', type=int, default=128,
                             help='Brightness threshold (0-255)')
    image_parser.add_argument('--invert', action='store_true',
                             help='Invert threshold (bright=obstacle)')
    image_parser.add_argument('--output', '-o', type=str, required=True,
                             help='Output file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'random':
        create_random_obstacles(
            grid_size=tuple(args.size),
            obstacle_density=args.density,
            output_path=args.output
        )
    
    elif args.command == 'rectangle':
        width, height = args.size
        # Create simple rectangular lawn
        vertices = [
            (2, 2),
            (width - 2, 2),
            (width - 2, height - 2),
            (2, height - 2)
        ]
        create_grid_from_polygon(
            vertices=vertices,
            grid_size=tuple(args.size),
            output_path=args.output
        )
    
    elif args.command == 'image':
        image_to_grid(
            image_path=args.input,
            output_path=args.output,
            grid_size=tuple(args.size),
            obstacle_threshold=args.threshold,
            invert=args.invert
        )
    
    print(f"\n✓ Map created successfully: {args.output}")
    print(f"  Run with: python main.py {args.output} --algorithm greedy")


if __name__ == '__main__':
    main()
