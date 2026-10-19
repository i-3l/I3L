#!/usr/bin/env python3
"""
Visualize URDF file using available visualization tools.

Supports:
- PyBullet (if installed)
- Meshcat (if installed)
- ROS check_urdf (if ROS is available)
"""

import argparse
import sys
from pathlib import Path

def visualize_with_pybullet(urdf_path: Path):
    """Visualize URDF using PyBullet."""
    try:
        import pybullet as p
        import pybullet_data
    except ImportError:
        print("PyBullet not installed. Install with: pip install pybullet")
        return False
    
    print(f"Loading URDF: {urdf_path}")
    
    # Start PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load URDF
    robot_id = p.loadURDF(str(urdf_path), useFixedBase=True)
    
    print("URDF loaded successfully!")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Shift+Mouse: Pan view")
    print("  - Ctrl+Mouse: Zoom")
    print("  - Close window to exit")
    
    # Keep window open
    try:
        while True:
            p.stepSimulation()
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()
    
    return True


def visualize_with_meshcat(urdf_path: Path):
    """Visualize URDF using Meshcat."""
    try:
        import meshcat
        import meshcat.geometry as g
        import meshcat.transformations as tf
        from urdf_parser_py.urdf import URDF
    except ImportError:
        print("Meshcat or urdf_parser_py not installed.")
        print("Install with: pip install meshcat urdf-parser-py")
        return False
    
    print(f"Loading URDF: {urdf_path}")
    
    # Parse URDF
    robot = URDF.from_xml_file(str(urdf_path))
    
    # Create visualizer
    vis = meshcat.Visualizer()
    vis.open()
    
    print("Meshcat viewer opened in browser!")
    print("Close the browser window or press Ctrl+C to exit")
    
    # Add links (simplified - would need full URDF parsing for meshes)
    for link in robot.links:
        if link.visuals:
            # Add a simple box for each link
            vis[link.name].set_object(g.Box([0.1, 0.1, 0.1]))
    
    try:
        input("Press Enter to close...")
    except KeyboardInterrupt:
        pass
    
    return True


def check_urdf_syntax(urdf_path: Path):
    """Check URDF syntax using ROS check_urdf if available."""
    import subprocess
    
    try:
        result = subprocess.run(
            ['check_urdf', str(urdf_path)],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return result.returncode == 0
    except FileNotFoundError:
        print("check_urdf not found. Install ROS to use this tool.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Visualize URDF file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Try PyBullet visualization
  python scripts/URDF/visualize_urdf.py scripts/robot.urdf
  
  # Try Meshcat visualization
  python scripts/URDF/visualize_urdf.py scripts/robot.urdf --method meshcat
  
  # Check URDF syntax only
  python scripts/URDF/visualize_urdf.py scripts/robot.urdf --check-only
        """
    )
    
    parser.add_argument(
        'urdf_file',
        type=str,
        help='Path to URDF file'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['pybullet', 'meshcat', 'auto'],
        default='auto',
        help='Visualization method (default: auto - tries pybullet first)'
    )
    
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check URDF syntax, do not visualize'
    )
    
    args = parser.parse_args()
    
    urdf_path = Path(args.urdf_file)
    if not urdf_path.exists():
        print(f"Error: URDF file not found: {urdf_path}")
        sys.exit(1)
    
    if args.check_only:
        check_urdf_syntax(urdf_path)
        return
    
    # Try visualization methods
    if args.method == 'auto':
        # Try PyBullet first, then Meshcat
        if visualize_with_pybullet(urdf_path):
            return
        if visualize_with_meshcat(urdf_path):
            return
        print("\nNo visualization method available.")
        print("Install one of:")
        print("  - PyBullet: pip install pybullet")
        print("  - Meshcat: pip install meshcat urdf-parser-py")
    elif args.method == 'pybullet':
        if not visualize_with_pybullet(urdf_path):
            sys.exit(1)
    elif args.method == 'meshcat':
        if not visualize_with_meshcat(urdf_path):
            sys.exit(1)


if __name__ == '__main__':
    main()








