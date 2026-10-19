"""
Script to chunk HDF5 dataset files to retain only the first N demonstrations.

This script takes an HDF5 dataset file and creates a new file containing only
the first N demonstrations, reducing the dataset size.

Usage:
    python chunk_dataset.py --input /path/to/dataset.hdf5 --output /path/to/chunked_dataset.hdf5 --num_demos 10
    
    Or to process a directory of HDF5 files:
    python chunk_dataset.py --input /path/to/data_dir --output /path/to/output_dir --num_demos 10 --process_dir
"""

import argparse
import h5py
import os
from pathlib import Path
from tqdm import tqdm


def chunk_single_file(input_file, output_file, num_demos):
    """
    Chunk a single HDF5 file to retain only the first N demonstrations.
    
    Args:
        input_file (str): Path to input HDF5 file
        output_file (str): Path to output HDF5 file
        num_demos (int): Number of demonstrations to retain
    """
    with h5py.File(input_file, "r") as in_f:
        # Get all demo keys
        demo_keys = sorted(in_f["data"].keys())
        
        # Check if we have enough demos
        num_available_demos = len(demo_keys)
        if num_demos > num_available_demos:
            print(f"Warning: Requested {num_demos} demos, but only {num_available_demos} available. "
                  f"Using all {num_available_demos} demos.")
            num_demos = num_available_demos
        
        # Select first N demos
        selected_demos = demo_keys[:num_demos]
        print(f"Processing {len(selected_demos)} demos out of {num_available_demos} total demos")
        
        # Create output file and copy selected demos
        with h5py.File(output_file, "w") as out_f:
            # Create data group in output file
            out_data_group = out_f.create_group("data")
            
            # Copy selected demonstrations
            for demo_name in tqdm(selected_demos, desc="Copying demos"):
                in_demo = in_f["data"][demo_name]
                out_demo = out_data_group.create_group(demo_name)
                
                # Copy all datasets within the demonstration
                for key in in_demo.keys():
                    in_subgroup = in_demo[key]
                    if isinstance(in_subgroup, h5py.Dataset):
                        # Copy dataset
                        out_demo.create_dataset(key, data=in_subgroup[:])
                    elif isinstance(in_subgroup, h5py.Group):
                        # Copy group and its contents
                        out_subgroup = out_demo.create_group(key)
                        for subkey in in_subgroup.keys():
                            if isinstance(in_subgroup[subkey], h5py.Dataset):
                                out_subgroup.create_dataset(subkey, data=in_subgroup[subkey][:])


def chunk_directory(input_dir, output_dir, num_demos):
    """
    Chunk all HDF5 files in a directory into multiple files with num_demos each.
    Demos are sorted by filename and redistributed across output files.
    
    Args:
        input_dir (str): Path to input directory containing HDF5 files
        output_dir (str): Path to output directory where chunked files will be saved
        num_demos (int): Number of demonstrations per output file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all HDF5 files (sorted)
    hdf5_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".hdf5")])
    
    if not hdf5_files:
        print(f"No HDF5 files found in {input_dir}")
        return
    
    print(f"Found {len(hdf5_files)} HDF5 files to process")
    
    # First pass: collect all demo metadata (file, demo_key)
    all_demos = []
    for hdf5_file in tqdm(hdf5_files, desc="Collecting demos"):
        input_file = os.path.join(input_dir, hdf5_file)
        with h5py.File(input_file, "r") as f:
            demo_keys = sorted(f["data"].keys())
            for demo_key in demo_keys:
                all_demos.append((input_file, demo_key))
    
    total_demos = len(all_demos)
    num_output_files = (total_demos + num_demos - 1) // num_demos  # ceiling division
    
    print(f"\nTotal demos across all files: {total_demos}")
    print(f"Will create {num_output_files} output files with {num_demos} demos each")
    
    # Extract base name from first file for output naming
    base_name = hdf5_files[0].replace(".hdf5", "")
    
    # Second pass: redistribute demos into output files
    demo_idx = 0
    for file_idx in tqdm(range(num_output_files), desc="Creating chunked files"):
        output_file = os.path.join(output_dir, f"{base_name}_chunk_{file_idx}.hdf5")
        
        # Determine how many demos go in this file
        demos_in_this_file = min(num_demos, total_demos - demo_idx)
        
        with h5py.File(output_file, "w") as out_f:
            out_data_group = out_f.create_group("data")
            
            # Copy demos for this output file
            for i in range(demos_in_this_file):
                input_file, demo_key = all_demos[demo_idx]
                
                # Open input file and copy demo
                with h5py.File(input_file, "r") as in_f:
                    in_demo = in_f["data"][demo_key]
                    out_demo = out_data_group.create_group(f"demo_{i}")
                    
                    # Copy all datasets within the demonstration
                    for key in in_demo.keys():
                        in_subgroup = in_demo[key]
                        if isinstance(in_subgroup, h5py.Dataset):
                            out_demo.create_dataset(key, data=in_subgroup[:])
                        elif isinstance(in_subgroup, h5py.Group):
                            out_subgroup = out_demo.create_group(key)
                            for subkey in in_subgroup.keys():
                                if isinstance(in_subgroup[subkey], h5py.Dataset):
                                    out_subgroup.create_dataset(subkey, data=in_subgroup[subkey][:])
                
                demo_idx += 1
        
        print(f"Created {output_file} with {demos_in_this_file} demos")


def main():
    parser = argparse.ArgumentParser(
        description="Chunk HDF5 dataset files to retain only the first N demonstrations"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to input HDF5 file or directory containing HDF5 files"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Path to output HDF5 file or directory where chunked files will be saved"
    )
    parser.add_argument(
        "-n", "--num_demos",
        type=int,
        required=True,
        help="Number of demonstrations to retain"
    )
    parser.add_argument(
        "-d", "--process_dir",
        action="store_true",
        help="Process all HDF5 files in the input directory"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input path '{args.input}' does not exist")
        return
    
    if args.num_demos <= 0:
        print("Error: num_demos must be a positive integer")
        return
    
    # Process single file or directory
    if args.process_dir:
        if not os.path.isdir(args.input):
            print("Error: Input path must be a directory when using --process_dir")
            return
        chunk_directory(args.input, args.output, args.num_demos)
    else:
        if not os.path.isfile(args.input):
            print("Error: Input path must be a file when not using --process_dir")
            return
        chunk_single_file(args.input, args.output, args.num_demos)
        print(f"\nSuccessfully saved chunked dataset to: {args.output}")


if __name__ == "__main__":
    main()
