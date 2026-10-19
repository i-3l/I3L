#!/usr/bin/env python3
"""
Export Onshape CAD model to URDF format using onshape-to-robot.

This script reads configuration from config.json and environment variables
from .env file to export your robot model from Onshape to URDF format.

Usage:
    python scripts/URDF/onshape_to_urdf.py [--output OUTPUT_DIR] [--config CONFIG_FILE]

Requirements:
    - pip install onshape-to-robot python-dotenv
    - Set ONSHAPE_ACCESS_KEY and ONSHAPE_SECRET_KEY in .env file
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    print("Error: python-dotenv not installed. Install with: pip install python-dotenv")
    sys.exit(1)

# Check if onshape-to-robot is available
def check_onshape_to_robot():
    """Check if onshape-to-robot CLI is available."""
    import shutil
    if shutil.which('onshape-to-robot') is None:
        # Try as Python module
        try:
            import onshape_to_robot
            return True
        except ImportError:
            return False
    return True


def load_config(config_path: Path) -> dict:
    """Load configuration from JSON file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Clean up empty optional fields that might cause issues
    # Remove fields with empty strings (onshape-to-robot expects them omitted or null)
    optional_fields = ['package_name', 'additional_xml']
    for field in optional_fields:
        if field in config and config[field] == "":
            del config[field]
    
    return config


def validate_env_vars() -> tuple[str, str]:
    """Validate that required environment variables are set."""
    access_key = os.getenv('ONSHAPE_ACCESS_KEY')
    secret_key = os.getenv('ONSHAPE_SECRET_KEY')
    
    if not access_key or access_key == 'your_access_key_here':
        raise ValueError(
            "ONSHAPE_ACCESS_KEY not set. Please set it in .env file or environment.\n"
            "Get your API keys from: https://dev.onshape.com/keys"
        )
    
    if not secret_key or secret_key == 'your_secret_key_here':
        raise ValueError(
            "ONSHAPE_SECRET_KEY not set. Please set it in .env file or environment.\n"
            "Get your API keys from: https://dev.onshape.com/keys"
        )
    
    return access_key, secret_key


def export_urdf(config: dict, output_dir: Path, access_key: str, secret_key: str, config_file: Path):
    """Export Onshape model to URDF using onshape-to-robot."""
    # Get the Onshape URL from config
    url = config.get('url')
    if not url:
        raise ValueError("'url' not found in config.json")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Exporting URDF from Onshape...")
    print(f"  URL: {url}")
    print(f"  Output directory: {output_dir}")
    print(f"  Format: {config.get('output_format', 'urdf')}")
    print(f"\nNote: For bilateral systems, export only ONE arm (6 or 7 DOF).")
    print(f"      The system will automatically mirror it for the second arm.")
    
    # Call onshape-to-robot CLI
    # The tool can be called in two ways:
    # 1. onshape-to-robot <url> --output <dir> (URL as first arg, no config)
    # 2. onshape-to-robot <config_file> --output <dir> (config file as first arg, URL in config)
    import subprocess
    
    # Prepare environment variables for subprocess
    # Create a copy of current environment and add Onshape API keys
    env = os.environ.copy()
    env['ONSHAPE_ACCESS_KEY'] = access_key
    env['ONSHAPE_SECRET_KEY'] = secret_key
    # Also set ONSHAPE_API if not already set (some versions might need this)
    if 'ONSHAPE_API' not in env:
        env['ONSHAPE_API'] = 'https://cad.onshape.com'
    
    # When using a config file, pass the config file as the first argument
    # The URL should already be in the config file
    if config_file.exists():
        # Use config file as first argument (URL is in the config)
        cmd = [
            'onshape-to-robot',
            str(config_file),
            '--output', str(output_dir)
        ]
    else:
        # No config file, pass URL directly
        cmd = [
            'onshape-to-robot',
            url,
            '--output', str(output_dir)
        ]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, check=False, env=env)
    
    if result.returncode != 0:
        print(f"\nError: Export failed with return code {result.returncode}")
        print("\nTroubleshooting:")
        print("  1. Verify onshape-to-robot is installed: pip install onshape-to-robot")
        print("  2. Check your API keys are correct in .env file")
        print("  3. Verify the Onshape URL is accessible")
        sys.exit(1)
    
    print(f"\n✓ Successfully exported URDF to {output_dir}")
    
    # Check for output files
    urdf_file = output_dir / 'robot.urdf'
    meshes_dir = output_dir / 'meshes'
    
    if urdf_file.exists():
        print(f"  URDF file: {urdf_file}")
    else:
        print(f"  Warning: Expected URDF file not found at {urdf_file}")
    
    if meshes_dir.exists():
        mesh_count = len(list(meshes_dir.glob('*.stl')))
        print(f"  Meshes: {meshes_dir} ({mesh_count} STL files)")
    else:
        print(f"  Warning: Meshes directory not found at {meshes_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Export Onshape CAD model to URDF format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config.json and output to ./urdf_output
  python scripts/URDF/onshape_to_urdf.py
  
  # Specify custom output directory
  python scripts/URDF/onshape_to_urdf.py --output ./my_robot_urdf
  
  # Use custom config file
  python scripts/URDF/onshape_to_urdf.py --config ./my_config.json
        """
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./urdf_output',
        help='Output directory for URDF files (default: ./urdf_output)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config.json file (default: scripts/config.json)'
    )
    
    args = parser.parse_args()
    
    # Check if onshape-to-robot is installed
    if not check_onshape_to_robot():
        print("Error: onshape-to-robot not found.")
        print("Install it with: pip install onshape-to-robot")
        sys.exit(1)
    
    # Determine script directory
    script_dir = Path(__file__).parent.resolve()
    
    # Load environment variables from .env file
    env_file = script_dir / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded environment variables from {env_file}")
    else:
        print(f"Warning: .env file not found at {env_file}")
        print("  Attempting to use environment variables from system...")
        print("  To create .env file, copy .env.example and fill in your credentials")
    
    # Load configuration
    if args.config:
        config_path = Path(args.config).resolve()
    else:
        config_path = script_dir / 'config.json'
    
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Validate environment variables
    try:
        access_key, secret_key = validate_env_vars()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Determine output directory
    output_dir = Path(args.output).resolve()
    
    # Export URDF
    try:
        export_urdf(config, output_dir, access_key, secret_key, config_path)
    except Exception as e:
        print(f"Error during export: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

