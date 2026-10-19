import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import pyvista as pv
from typing import List, Optional


def visualize_state_distribution(
    hdf5_files: List[str],
    max_demos: Optional[int] = None,
    alpha: float = 0.5,
):
    """
    Visualize the 3D distribution of object positions from multiple HDF5 demonstration files.
    
    Args:
        hdf5_files: List of paths to HDF5 files containing demonstrations
        max_demos: Maximum number of demos to process per file (None = all)
        subsample: Subsample factor for points (1 = all points, 10 = every 10th point)
        alpha: Transparency for scatter points
        figsize: Figure size (width, height)
    """
    # Collect all xyz positions
    object_positions, target_positions = [], []
    
    for hdf5_file in hdf5_files:
        print(f"Processing file: {hdf5_file}")
        try:
            with h5py.File(hdf5_file, 'r') as f:
                demo_idx = 0
                # Iterate through all demos in the file
                while True:
                    demo_key = f"demo_{demo_idx}"
                    if demo_key not in f["data"]:
                        break
                    if max_demos is not None and demo_idx >= max_demos:
                        break
                    try:
                        # Extract state from obs - GET ONLY THE INITIAL STATE (first timestep)
                        state = f["data"][demo_key]["obs"]["task::object_pose"][:]
                        
                        # Get xyz positions of initial state only (first 3 dimensions)
                        initial_object_pos = state[0, :3]  # Only first timestep
                        initial_target_pos = state[0, 7:10]  # Only first timestep
                        
                        object_positions.append(initial_object_pos)
                        target_positions.append(initial_target_pos)
                        
                    except Exception as e:
                        print(f"  - Error reading {demo_key}: {e}")
                    
                    demo_idx += 1
                    
        except Exception as e:
            print(f"Error opening file {hdf5_file}: {e}")
            continue
    
    if not object_positions:
        print("No data found!")
        return
    
    # Stack all initial positions
    all_object_positions = np.array(object_positions)  # Now just one point per demo
    all_target_positions = np.array(target_positions)
    
    print(f"\nTotal demos: {len(all_object_positions)}")
    print(f"Initial object position range:")
    print(f"  X: [{all_object_positions[:, 0].min():.3f}, {all_object_positions[:, 0].max():.3f}]")
    print(f"  Y: [{all_object_positions[:, 1].min():.3f}, {all_object_positions[:, 1].max():.3f}]")
    print(f"  Z: [{all_object_positions[:, 2].min():.3f}, {all_object_positions[:, 2].max():.3f}]")
    print(f"Initial target position range:")
    print(f"  X: [{all_target_positions[:, 0].min():.3f}, {all_target_positions[:, 0].max():.3f}]")
    print(f"  Y: [{all_target_positions[:, 1].min():.3f}, {all_target_positions[:, 1].max():.3f}]")
    print(f"  Z: [{all_target_positions[:, 2].min():.3f}, {all_target_positions[:, 2].max():.3f}]")
    
    # Create visualization - only 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 3D scatter plot of initial states
    scatter = ax.scatter(
        all_object_positions[:, 0],
        all_object_positions[:, 1],
        all_object_positions[:, 2],
        c='blue',
        alpha=alpha,
        s=50,  # Larger markers since we have fewer points
        label='Object Initial Position'
    )
    
    # Also plot target positions if useful
    ax.scatter(
        all_target_positions[:, 0],
        all_target_positions[:, 1],
        all_target_positions[:, 2],
        c='red',
        alpha=alpha,
        s=50,
        marker='x',
        label='Target Position'
    )
    
    ax.set_xlabel('X Position (Forward)')
    ax.set_ylabel('Y Position (Left)')
    ax.set_zlabel('Z Position (Up)')
    ax.set_title(f'Initial State Distribution ({len(all_object_positions)} demos)')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return all_object_positions, all_target_positions


def visualize_trajectory(
    hdf5_files: List[str],
    max_demos: Optional[int] = None,
    subsample: int = 5,
    alpha: float = 0.7,
    linewidth: float = 2.0,
    segmentation_mode: str = 'grasp',
):
    """
    Visualize robot end effector trajectories segmented by grasp state or intervention mode using PyVista.
    
    Args:
        hdf5_files: List of paths to HDF5 files containing demonstrations
        max_demos: Maximum number of demos to process per file (None = all)
        subsample: Subsample factor for points (e.g., 5 = every 5th point)
        alpha: Transparency for trajectory lines (not used in PyVista version)
        linewidth: Width of trajectory lines
        segmentation_mode: 'grasp' or 'intervention' - how to segment trajectories
    """    
    # Create PyVista plotter with larger window
    plotter = pv.Plotter(window_size=[2500, 2000])
    plotter.set_background('white')
    
    # Track meshes for toggling visibility
    pre_grasp_meshes = []
    post_grasp_meshes = []
    start_markers = []
    end_markers = []
    grasp_markers = []
    
    # Mesh sets for color scheme toggling
    grasp_meshes = []  # Red->magenta->blue gradient meshes
    intervention_meshes = []  # Intervention state colored meshes
    
    # View state: 0=full, 1=pre-grasp only, 2=post-grasp only
    view_state = {'current': 0}
    # Color scheme: 0=grasp, 1=intervention
    color_scheme = {'current': 0 if segmentation_mode == 'grasp' else 1}
    
    # Title text actor for dynamic updates
    title_actor = None
    
    def update_title():
        """Update title based on current color scheme"""
        nonlocal title_actor
        if color_scheme['current'] == 0:
            title_text = (
                f'Robot End Effector Trajectories ({demo_count} demos)\n'
                'Red→Magenta: Approach & Pick | Magenta→Blue: Transport & Place'
            )
        else:
            title_text = (
                f'Robot End Effector Trajectories ({demo_count} demos)\n'
                'Green: Human Demo | Cyan: Pre-Intervention | Yellow: Intervention | Pink: On-Policy'
            )
        
        if title_actor is not None:
            plotter.remove_actor(title_actor)
        title_actor = plotter.add_text(
            title_text,
            position='upper_left',
            font_size=12,
            color='black'
        )
    
    def toggle_view():
        """Toggle between full, pre-grasp only, and post-grasp only views"""
        view_state['current'] = (view_state['current'] + 1) % 3
        
        # Determine which meshes to control based on current color scheme
        if color_scheme['current'] == 0:
            active_pre_meshes = [m for m in pre_grasp_meshes if m in grasp_meshes]
            active_post_meshes = [m for m in post_grasp_meshes if m in grasp_meshes]
        else:
            active_pre_meshes = [m for m in pre_grasp_meshes if m in intervention_meshes]
            active_post_meshes = [m for m in post_grasp_meshes if m in intervention_meshes]
        
        if view_state['current'] == 0:
            # Full trajectory
            for mesh in active_pre_meshes + active_post_meshes:
                mesh.SetVisibility(True)
            for mesh in start_markers + end_markers + grasp_markers:
                mesh.SetVisibility(True)
            print("View: Full trajectory")
        elif view_state['current'] == 1:
            # Pre-grasp only - hide end markers
            for mesh in active_pre_meshes:
                mesh.SetVisibility(True)
            for mesh in active_post_meshes:
                mesh.SetVisibility(False)
            for mesh in start_markers + grasp_markers:
                mesh.SetVisibility(True)
            for mesh in end_markers:
                mesh.SetVisibility(False)
            print("View: Pre-grasp only (approach & pick)")
        elif view_state['current'] == 2:
            # Post-grasp only - hide start markers
            for mesh in active_pre_meshes:
                mesh.SetVisibility(False)
            for mesh in active_post_meshes:
                mesh.SetVisibility(True)
            for mesh in start_markers:
                mesh.SetVisibility(False)
            for mesh in end_markers + grasp_markers:
                mesh.SetVisibility(True)
            print("View: Post-grasp only (transport & place)")
        
        plotter.render()
    
    def toggle_color_scheme():
        """Toggle between grasp-based and intervention-based color schemes"""
        color_scheme['current'] = (color_scheme['current'] + 1) % 2
        
        # Hide the previous color scheme's meshes
        if color_scheme['current'] == 0:
            # Switching to grasp coloring
            for mesh in intervention_meshes:
                mesh.SetVisibility(False)
            print("Color scheme: Grasp-based (Red→Magenta→Blue)")
        else:
            # Switching to intervention coloring
            for mesh in grasp_meshes:
                mesh.SetVisibility(False)
            print("Color scheme: Intervention-based (Cyan/Yellow/Pink)")
        
        # Update title
        update_title()
        
        # Re-apply current view state to set correct visibility for new color scheme
        current_view = view_state['current']
        
        # Determine which meshes to show based on view state
        if color_scheme['current'] == 0:
            active_pre_meshes = [m for m in pre_grasp_meshes if m in grasp_meshes]
            active_post_meshes = [m for m in post_grasp_meshes if m in grasp_meshes]
        else:
            active_pre_meshes = [m for m in pre_grasp_meshes if m in intervention_meshes]
            active_post_meshes = [m for m in post_grasp_meshes if m in intervention_meshes]
        
        if current_view == 0:
            # Full trajectory
            for mesh in active_pre_meshes + active_post_meshes:
                mesh.SetVisibility(True)
        elif current_view == 1:
            # Pre-grasp only
            for mesh in active_pre_meshes:
                mesh.SetVisibility(True)
            for mesh in active_post_meshes:
                mesh.SetVisibility(False)
        elif current_view == 2:
            # Post-grasp only
            for mesh in active_pre_meshes:
                mesh.SetVisibility(False)
            for mesh in active_post_meshes:
                mesh.SetVisibility(True)
        
        plotter.render()
    
    demo_count = 0
    
    for hdf5_file in hdf5_files:
        print(f"Processing file: {hdf5_file}")
        try:
            with h5py.File(hdf5_file, 'r') as f:
                demo_idx = 0
                while True:
                    demo_key = f"demo_{demo_idx}"
                    if demo_key not in f["data"]:
                        break
                    if max_demos is not None and demo_count >= max_demos:
                        break
                    try:
                        # Extract robot proprioception
                        proprio = f["data"][demo_key]["obs"]["robot::proprio"][:]
                        # Get end effector position (xyz)
                        eef_pos = proprio[:, 40:43]
                        
                        # Apply subsampling
                        eef_pos = eef_pos[::subsample]
                        
                        # Get grasp indicator and int_state for both modes
                        grasp_indicator = proprio[:, 47]
                        grasp_indicator = grasp_indicator[::subsample]
                        int_state = f["data"][demo_key]["policy"]["int_state"][:].flatten()
                        int_state = int_state[::subsample]
                        
                        # Skip human demos in intervention mode
                        if segmentation_mode == 'intervention' and np.all(int_state == 0):
                            print(f"  - {demo_key}: Skipping human demo")
                            demo_idx += 1
                            continue
                        
                        # Find grasp transition point
                        grasp_indices = np.where(grasp_indicator == 1)[0]
                        if len(grasp_indices) > 0:
                            first_grasp_idx = grasp_indices[0]
                        else:
                            first_grasp_idx = len(eef_pos)
                        
                        # ============== RENDER GRASP-BASED COLORING ==============
                        # Segment 1: Move to object and pick (before grasp)
                        pre_grasp_traj = eef_pos[:first_grasp_idx + 1]
                        # Segment 2: Move to target and place (after grasp)
                        post_grasp_traj = eef_pos[first_grasp_idx:]
                        
                        # Pre-grasp: red to magenta
                        if len(pre_grasp_traj) > 1:
                            # Create color array for pre-grasp segment
                            red = np.array([255, 0, 0])
                            magenta = np.array([255, 0, 255])
                            n_pre = len(pre_grasp_traj)
                            
                            colors_pre = np.array([
                                red * (1 - i / (n_pre - 1)) + magenta * (i / (n_pre - 1))
                                for i in range(n_pre)
                            ], dtype=np.uint8)
                            
                            # Create line from trajectory points
                            line = pv.lines_from_points(pre_grasp_traj)
                            line['colors'] = colors_pre
                            
                            actor = plotter.add_mesh(
                                line,
                                scalars='colors',
                                rgb=True,
                                line_width=linewidth,
                                render_lines_as_tubes=True
                            )
                            actor.SetVisibility(segmentation_mode == 'grasp')
                            pre_grasp_meshes.append(actor)
                            grasp_meshes.append(actor)
                            pre_grasp_meshes.append(actor)
                            grasp_meshes.append(actor)
                        
                        # Post-grasp: magenta to blue
                        if len(post_grasp_traj) > 1:
                            # Create color array for post-grasp segment
                            magenta = np.array([255, 0, 255])
                            blue = np.array([0, 0, 255])
                            n_post = len(post_grasp_traj)
                            
                            colors_post = np.array([
                                magenta * (1 - i / (n_post - 1)) + blue * (i / (n_post - 1))
                                for i in range(n_post)
                            ], dtype=np.uint8)
                            
                            # Create line from trajectory points
                            line = pv.lines_from_points(post_grasp_traj)
                            line['colors'] = colors_post
                            
                            actor = plotter.add_mesh(
                                line,
                                scalars='colors',
                                rgb=True,
                                line_width=linewidth,
                                render_lines_as_tubes=True
                            )
                            actor.SetVisibility(segmentation_mode == 'grasp')
                            post_grasp_meshes.append(actor)
                            grasp_meshes.append(actor)
                        
                        # ============== RENDER INTERVENTION-BASED COLORING ==============
                        # Define colors for each intervention state
                        # 0: human demo (green), 1: pre-intervention (cyan), 
                        # 2: intervention (yellow), 3: on-policy rollout (pink)
                        state_colors = {
                            0: np.array([0, 255, 0]),      # green
                            1: np.array([0, 255, 255]),    # cyan
                            2: np.array([255, 255, 0]),    # yellow
                            3: np.array([255, 192, 203])     # pink
                        }
                        
                        # Find segments where intervention state changes OR at grasp point
                        change_indices = [0] + (np.where(np.diff(int_state) != 0)[0] + 1).tolist() + [len(int_state)]
                        # Add grasp transition point to split intervention segments at grasp
                        if first_grasp_idx not in change_indices and first_grasp_idx < len(int_state):
                            change_indices.append(first_grasp_idx)
                        change_indices = sorted(set(change_indices))  # Remove duplicates and sort
                        
                        # Render each segment with its corresponding color
                        for i in range(len(change_indices) - 1):
                            start_idx = change_indices[i]
                            end_idx = change_indices[i + 1]
                            segment_traj = eef_pos[start_idx:end_idx + 1]
                            
                            if len(segment_traj) > 1:
                                state_val = int(int_state[start_idx])
                                color = state_colors.get(state_val, np.array([128, 128, 128]))  # default gray
                                
                                # Create line from trajectory points
                                line = pv.lines_from_points(segment_traj)
                                
                                # Assign uniform color to segment
                                segment_colors = np.tile(color, (len(segment_traj), 1)).astype(np.uint8)
                                line['colors'] = segment_colors
                                
                                actor = plotter.add_mesh(
                                    line,
                                    scalars='colors',
                                    rgb=True,
                                    line_width=linewidth,
                                    render_lines_as_tubes=True
                                )
                                actor.SetVisibility(segmentation_mode == 'intervention')
                                intervention_meshes.append(actor)
                                
                                # Categorize as pre or post grasp for view toggling
                                # Use the midpoint of the segment to determine category
                                segment_midpoint = (start_idx + end_idx) / 2
                                if segment_midpoint < first_grasp_idx:
                                    pre_grasp_meshes.append(actor)
                                else:
                                    post_grasp_meshes.append(actor)
                        
                        # ============== ADD MARKERS ==============
                        # Mark start with red sphere
                        sphere_start = pv.Sphere(radius=0.001, center=eef_pos[0])
                        actor = plotter.add_mesh(sphere_start, color='red')
                        start_markers.append(actor)
                        
                        # Mark end with blue sphere
                        sphere_end = pv.Sphere(radius=0.001, center=eef_pos[-1])
                        actor = plotter.add_mesh(sphere_end, color='blue')
                        end_markers.append(actor)
                        
                        # Mark grasp transition point with magenta sphere
                        if len(grasp_indices) > 0:
                            sphere_grasp = pv.Sphere(radius=0.001, center=eef_pos[first_grasp_idx])
                            actor = plotter.add_mesh(sphere_grasp, color='magenta')
                            grasp_markers.append(actor)
                        
                        # Print summary
                        print(f"  - {demo_key}: {len(eef_pos)} points, grasp at idx {first_grasp_idx}")
                        demo_count += 1
                        
                    except Exception as e:
                        print(f"  - Error reading {demo_key}: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    demo_idx += 1
                    
        except Exception as e:
            print(f"Error opening file {hdf5_file}: {e}")
            continue
    
    # Initialize title
    update_title()
    
    # Add grid with ticks (similar to matplotlib)
    plotter.show_grid(
        xlabel='X (Forward)',
        ylabel='Y (Left)',
        zlabel='Z (Up)',
        color='gray',
        show_xlabels=True,
        show_ylabels=True,
        show_zlabels=True,
        font_size=10,
        padding=0.05
    )
    
    # Add keyboard callback for toggling view
    plotter.add_key_event('s', toggle_view)
    plotter.add_key_event('c', toggle_color_scheme)
    print("\nPress 's' to toggle between Full / Pre-grasp / Post-grasp views")
    print("Press 'c' to toggle between Grasp and Intervention color schemes")
    
    plotter.reset_camera()
    plotter.camera.zoom(1.25) 
    plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize state distributions from HDF5 demo files.")
    parser.add_argument("-d", "--demos", help='Path to the folder containing HDF5 demo files.', required=True)
    parser.add_argument("-m", "--mode", help='Visualization mode: "distribution" or "trajectory"', 
                        default="distribution", choices=["distribution", "trajectory"])
    parser.add_argument("--max-demos", type=int, help='Maximum number of demos to visualize', default=None)
    parser.add_argument("--subsample", type=int, help='Subsample factor for trajectory visualization', default=1)
    parser.add_argument("--segment-by", help='Segmentation mode for trajectories: "grasp" or "intervention"',
                        default='grasp', choices=['grasp', 'intervention'])
    
    args = parser.parse_args()
    
    # get all files ending with .hdf5 in the provided folder
    hdf5_files = []
    for filename in os.listdir(args.demos):
        if filename.endswith(".hdf5"):
            hdf5_files.append(os.path.join(args.demos, filename))
    
    print("=" * 50)
    print(f"Found {len(hdf5_files)} HDF5 files.")
    
    if args.mode == "distribution":
        print("Visualizing state distribution")
        print("=" * 50)
        visualize_state_distribution(hdf5_files, max_demos=args.max_demos)
    elif args.mode == "trajectory":
        print("Visualizing robot trajectories")
        print("=" * 50)
        visualize_trajectory(hdf5_files, max_demos=args.max_demos, subsample=args.subsample, 
                           segmentation_mode=args.segment_by)
    