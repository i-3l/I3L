#!/bin/bash

# IIIL Run Script - Data Collection, Policy, and Playback
# Usage: ./run.sh

# echo pwd
echo 

# if [[ -f /opt/ros/humble/setup.bash ]]; then
#     source /opt/ros/humble/setup.bash
# fi
# Do NOT source /opt/ros/humble/setup.bash here.
# The iiil conda env has its own ROS Humble (RoboStack) that activates automatically.
# Sourcing system ROS prepends Python 3.10 paths to PYTHONPATH, which breaks
# the conda env's Python 3.11 rclpy.

# Source conda (activation deferred until after mode selection)
if [[ -f ~/miniforge3/etc/profile.d/conda.sh ]]; then
    source ~/miniforge3/etc/profile.d/conda.sh
elif [[ -f ~/miniconda3/etc/profile.d/conda.sh ]]; then
    source ~/miniconda3/etc/profile.d/conda.sh
else
    echo "Error: conda installation not found"
    exit 1
fi


# ROS2 + FastDDS Discovery Server client settings
# REQUIRES BOTH: XML config file AND environment variable
export FASTRTPS_DEFAULT_PROFILES_FILE=$PWD/scripts/real_robot/super_client_configuration_file.xml
export ROS_DISCOVERY_SERVER="${ROS_DISCOVERY_SERVER:-10.0.0.30:11811}"  # R1 Pro onboard discovery server
export ROS_DOMAIN_ID=0
unset ROS_LOCALHOST_ONLY LOCAL_HOST_ONLY 2>/dev/null

DATA_DIR="${HOME}/data"
IIIL_DIR=$PWD
IL_LIB_DIR=$(dirname "$IIIL_DIR")/il_lib

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --------------- camera pre-check (real R1Pro only) ---------------
check_cameras() {
    echo ""
    echo -e "${BLUE}Checking camera feeds...${NC}"
    python "$IIIL_DIR/scripts/check_cameras.py" --timeout 3
    if [[ $? -ne 0 ]]; then
        read -p "Continue anyway? [y/N]: " cam_choice
        case $cam_choice in
            [Yy]*) echo -e "${YELLOW}Proceeding without all cameras.${NC}" ;;
            *)     echo -e "${RED}Aborted.${NC}"; exit 1 ;;
        esac
    fi
    echo ""
}

# --------------- Real Franka guided startup ---------------
prepare_real_franka() {
    local setup_script="$IIIL_DIR/setup_franka_robot.sh"

    echo ""
    echo -e "${BLUE}======================================================================${NC}"
    echo -e "${BLUE}  Franka Robot Guided Startup${NC}"
    echo -e "${BLUE}======================================================================${NC}"
    echo ""
    echo -e "  Franka Desk: ${BLUE}https://172.16.0.2${NC}"
    echo -e "  Use Desk to unlock the joints and activate FCI before control starts."
    echo ""

    if [[ ! -x "$setup_script" ]]; then
        echo -e "${RED}ERROR: Franka robot setup is missing or not executable:${NC}"
        echo -e "  ${RED}${setup_script}${NC}"
        exit 1
    fi

    if ! "$setup_script"; then
        echo ""
        echo -e "${RED}Franka robot setup did not pass. Real control will not start.${NC}"
        exit 1
    fi
    echo ""
}

# --------------- Franka scene snapshot (align.sh) ---------------
snapshot_franka_scene() {
    # Record how the scene looks at session start so it can be re-created
    # later with ./align.sh --last (or ./align.sh --list for older sessions).
    # Must run before run_task.py starts, while the cameras are still free.
    local hdf5_path="$1"
    local base
    base="$(basename "$hdf5_path")"
    base="${base%.hdf5}"

    echo ""
    echo -e "${BLUE}Capturing scene snapshot for align.sh...${NC}"
    if "$IIIL_DIR/align.sh" --snapshot "$base"; then
        echo -e "${GREEN}Scene snapshot saved (re-create later with: ./align.sh ${base}).${NC}"
    else
        echo -e "${YELLOW}WARNING: Scene snapshot failed; continuing without it.${NC}"
    fi
    echo ""
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}         IIIL Run Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Select mode:"
echo ""
echo -e "  ${BLUE}--- R1Pro ---${NC}"
echo "  1) Data Collection (Gello + Data Collection)"
echo "  2) Policy Rollout (IL + Gello)"
echo "  3) TrajectoryPlayback (Replay)"
echo "  4) HDF5 to Video"
echo "  14) Plot Joint Traces"
echo "  5) Policy Evaluation (IL)"
echo "  9) PolicyCorrection (IL + Gello + Data Collection)"
echo ""
echo -e "  ${BLUE}--- Franka ---${NC}"
echo "  10) Franka Data Collection (Gello Teleop)"
echo "  11) Franka PolicyCorrection (IL + Gello + Touch Takeover)"
echo "  12) Franka Sim Teleop (MuJoCo + Gello)"
echo "  13) Franka Sim Correction Test (Sine Policy + Admittance)"
echo ""
echo -e "  ${BLUE}--- Simulation ---${NC}"
echo "  6) Simulation Teleoperation"
echo "  7) Simulation Rollout (CuRobo + Gello)"
echo "  8) Simulation Rollout (IL)"
echo ""
mode_choice="${1:-}"
if [[ -z "$mode_choice" ]]; then
    read -p "Enter choice: " mode_choice
fi

case $mode_choice in
    1|dc|DC)
        # ==================== DATA COLLECTION ====================
        conda activate iiil
        check_cameras
        echo ""
        echo -e "${GREEN}=== Data Collection Mode ===${NC}"
        echo ""
        echo "Enter save file name (relative to ${DATA_DIR}):"
        echo "  Examples: mydata.hdf5, test/filename.hdf5"
        read -p "Filename: " filename

        # Add .hdf5 extension if not present
        if [[ ! "$filename" == *.hdf5 ]]; then
            filename="${filename}.hdf5"
        fi

        # Construct full path
        if [[ "$filename" == /* ]]; then
            full_path="$filename"
        else
            full_path="${DATA_DIR}/${filename}"
        fi

        # Create parent directory if it doesn't exist
        parent_dir=$(dirname "$full_path")
        if [[ ! -d "$parent_dir" ]]; then
            echo -e "${YELLOW}Directory $parent_dir does not exist. Creating...${NC}"
            mkdir -p "$parent_dir"
        fi

        # Check if file exists
        overwrite_flag="false"
        if [[ -f "$full_path" ]]; then
            echo ""
            echo -e "${YELLOW}WARNING: File already exists: ${full_path}${NC}"
            read -p "Do you want to overwrite? [y/N]: " overwrite_choice
            case $overwrite_choice in
                [Yy]*)
                    overwrite_flag="true"
                    echo -e "${YELLOW}Will overwrite existing file.${NC}"
                    ;;
                *)
                    echo -e "${RED}Aborted. Please choose a different filename.${NC}"
                    exit 1
                    ;;
            esac
        fi

        echo ""
        echo -e "${GREEN}Starting data collection...${NC}"
        echo -e "Saving to: ${BLUE}${full_path}${NC}"
        echo ""

        cmd="python run_task.py env=real_r1pro_joint task=real/dc policy=gello policy.base_policy=dummy policy.control_direction=gello_to_env policy.require_joycon=true policy.takeover_enable=false hdf_path=${full_path} overwrite=${overwrite_flag}"
        echo -e "${YELLOW}Running: ${cmd}${NC}"
        echo ""
        exec $cmd
        ;;

    2|policy|Policy)
        # ==================== POLICY ====================
        conda activate iiil
        check_cameras
        echo ""
        echo -e "${GREEN}=== Policy Mode ===${NC}"
        echo ""
        echo "This will start the policy server and run the task."
        echo ""

        # Start run_task.py (stays in iiil env)
        echo ""
        echo -e "${GREEN}Starting task runner (run_task.py) in 'iiil' conda env...${NC}"
        cd "$IIIL_DIR"
        task_cmd="python run_task.py env=real_r1pro_joint task=real/base policy=gello policy.base_policy=il policy.require_joycon=true policy.takeover_enable=false"
        echo -e "${YELLOW}Running: ${task_cmd}${NC}"
        echo ""

        # Run task and wait for it to complete
        $task_cmd
        TASK_EXIT=$?

        # Cleanup: kill the server
        echo ""
        echo -e "${YELLOW}Stopping policy server (PID: ${SERVE_PID})...${NC}"
        kill $SERVE_PID 2>/dev/null
        wait $SERVE_PID 2>/dev/null

        echo -e "${GREEN}Done.${NC}"
        exit $TASK_EXIT
        ;;

    3|playback|Playback)
        # ==================== PLAYBACK ====================
        conda activate iiil
        check_cameras
        echo ""
        echo -e "${GREEN}=== Playback Mode ===${NC}"
        echo ""
        echo "Enter file name to playback (relative to ${DATA_DIR}):"
        echo "  Examples: mydata.hdf5, test/filename.hdf5"
        read -p "Filename: " filename

        # Add .hdf5 extension if not present
        if [[ ! "$filename" == *.hdf5 ]]; then
            filename="${filename}.hdf5"
        fi

        # Construct full path
        if [[ "$filename" == /* ]]; then
            full_path="$filename"
        else
            full_path="${DATA_DIR}/${filename}"
        fi

        # Check if file exists
        if [[ ! -f "$full_path" ]]; then
            echo -e "${RED}ERROR: File does not exist: ${full_path}${NC}"
            exit 1
        fi

        # Show available demos
        echo ""
        echo -e "${BLUE}Checking available trajectories in file...${NC}"
        num_demos=$(python3 -c "
import h5py
with h5py.File('${full_path}', 'r') as f:
    # Check both root level and inside 'data' group
    if 'data' in f:
        demos = [k for k in f['data'].keys() if k.startswith('demo_')]
    else:
        demos = [k for k in f.keys() if k.startswith('demo_')]
    print(len(demos))
" 2>/dev/null)

        if [[ -z "$num_demos" || "$num_demos" == "0" ]]; then
            echo -e "${RED}ERROR: No demos found in file or unable to read file.${NC}"
            exit 1
        fi

        echo -e "${GREEN}Found ${num_demos} trajectory/trajectories (demo_0 to demo_$((num_demos-1)))${NC}"
        echo ""

        # Ask which trajectories
        echo "Which trajectories do you want to play?"
        echo "  - Enter 'all' for all trajectories"
        echo "  - Enter comma-separated indices (e.g., 0,1,2)"
        echo "  - Enter a range (e.g., 0-5)"
        read -p "Trajectories: " traj_input

        # Parse trajectory input
        if [[ "$traj_input" == "all" || "$traj_input" == "ALL" || -z "$traj_input" ]]; then
            demo_ids="null"
            echo -e "${BLUE}Will play all trajectories.${NC}"
        else
            # Convert range format (e.g., 0-5) to list
            if [[ "$traj_input" == *-* ]]; then
                start=$(echo "$traj_input" | cut -d'-' -f1)
                end=$(echo "$traj_input" | cut -d'-' -f2)
                demo_ids="["
                for ((i=start; i<=end; i++)); do
                    if [[ $i -gt $start ]]; then
                        demo_ids="${demo_ids},"
                    fi
                    demo_ids="${demo_ids}${i}"
                done
                demo_ids="${demo_ids}]"
            else
                # Already comma-separated, just wrap in brackets
                demo_ids="[${traj_input}]"
            fi
            echo -e "${BLUE}Will play trajectories: ${demo_ids}${NC}"
        fi

        echo ""
        read -p "How many times to repeat? [1]: " repeat_count
        if [[ -z "$repeat_count" ]]; then
            repeat_count=1
        fi

        echo ""
        echo -e "${GREEN}Starting playback...${NC}"
        echo -e "File: ${BLUE}${full_path}${NC}"
        echo -e "Trajectories: ${BLUE}${demo_ids}${NC}"
        echo -e "Repeat: ${BLUE}${repeat_count}${NC}"
        echo ""

        cd "$IIIL_DIR"
        if [[ "$demo_ids" == "null" ]]; then
            cmd="python run_task.py env=real_r1pro_joint task=real/playback policy=playback hdf_path=${full_path} policy.model.repeat=${repeat_count}"
        else
            cmd="python run_task.py env=real_r1pro_joint task=real/playback policy=playback hdf_path=${full_path} policy.model.demo_ids=${demo_ids} policy.model.repeat=${repeat_count}"
        fi
        echo -e "${YELLOW}Running: ${cmd}${NC}"
        echo ""
        exec $cmd
        ;;

    4|video|Video)
        # ==================== HDF5 TO VIDEO ====================
        conda activate iiil
        echo ""
        echo -e "${GREEN}=== HDF5 to Video Mode ===${NC}"
        echo ""
        echo "Enter file name to convert (relative to ${DATA_DIR}):"
        echo "  Examples: mydata.hdf5, test/filename.hdf5"
        read -p "Filename: " filename

        # Add .hdf5 extension if not present
        if [[ ! "$filename" == *.hdf5 ]]; then
            filename="${filename}.hdf5"
        fi

        # Construct full path
        if [[ "$filename" == /* ]]; then
            full_path="$filename"
        else
            full_path="${DATA_DIR}/${filename}"
        fi

        # Check if file exists
        if [[ ! -f "$full_path" ]]; then
            echo -e "${RED}ERROR: File does not exist: ${full_path}${NC}"
            exit 1
        fi

        # Show available demos
        echo ""
        echo -e "${BLUE}Checking available demos in file...${NC}"
        num_demos=$(python3 -c "
import h5py
with h5py.File('${full_path}', 'r') as f:
    if 'data' in f:
        demos = [k for k in f['data'].keys() if k.startswith('demo_')]
    else:
        demos = [k for k in f.keys() if k.startswith('demo_')]
    print(len(demos))
" 2>/dev/null)

        if [[ -z "$num_demos" || "$num_demos" == "0" ]]; then
            echo -e "${RED}ERROR: No demos found in file or unable to read file.${NC}"
            exit 1
        fi

        echo -e "${GREEN}Found ${num_demos} demo(s) (demo_0 to demo_$((num_demos-1)))${NC}"
        echo ""

        # Ask which demo to convert
        echo "Which demo do you want to convert to video?"
        echo "  - Enter 'all' for all demos"
        echo "  - Enter a specific demo (e.g., demo_0)"
        read -p "Demo [all]: " demo_input

        # Build command
        cd "$IIIL_DIR"
        if [[ -z "$demo_input" || "$demo_input" == "all" || "$demo_input" == "ALL" ]]; then
            cmd="python scripts/data/hdf5_to_video.py ${full_path}"
            echo -e "${BLUE}Will convert all demos.${NC}"
        else
            cmd="python scripts/data/hdf5_to_video.py ${full_path} --demo ${demo_input}"
            echo -e "${BLUE}Will convert ${demo_input}.${NC}"
        fi

        echo ""
        echo -e "${GREEN}Converting HDF5 to video...${NC}"
        echo -e "File: ${BLUE}${full_path}${NC}"
        echo ""
        echo -e "${YELLOW}Running: ${cmd}${NC}"
        echo ""
        exec $cmd
        ;;

    14|joints|"joint traces")
        # ==================== PLOT JOINT TRACES ====================
        echo ""
        echo -e "${GREEN}=== Plot Joint Traces ===${NC}"
        echo ""
        echo "Enter file name (relative to ${DATA_DIR}):"
        echo "  Examples: mydata.hdf5, test/filename.hdf5"
        read -p "Filename: " filename

        # Add .hdf5 extension if not present
        if [[ ! "$filename" == *.hdf5 ]]; then
            filename="${filename}.hdf5"
        fi

        # Construct full path
        if [[ "$filename" == /* ]]; then
            full_path="$filename"
        else
            full_path="${DATA_DIR}/${filename}"
        fi

        # Check if file exists
        if [[ ! -f "$full_path" ]]; then
            echo -e "${RED}ERROR: File does not exist: ${full_path}${NC}"
            exit 1
        fi

        # Show available demos
        echo ""
        echo -e "${BLUE}Checking available demos in file...${NC}"
        num_demos=$(python3 -c "
import h5py
with h5py.File('${full_path}', 'r') as f:
    if 'data' in f:
        demos = [k for k in f['data'].keys() if k.startswith('demo_')]
    else:
        demos = [k for k in f.keys() if k.startswith('demo_')]
    print(len(demos))
" 2>/dev/null)

        if [[ -z "$num_demos" || "$num_demos" == "0" ]]; then
            echo -e "${RED}ERROR: No demos found in file or unable to read file.${NC}"
            exit 1
        fi

        echo -e "${GREEN}Found ${num_demos} demo(s) (demo_0 to demo_$((num_demos-1)))${NC}"
        echo ""

        echo "Which demo to plot?"
        echo "  - Enter 'all' for all demos"
        echo "  - Enter a specific demo (e.g., demo_0)"
        read -p "Demo [all]: " demo_input

        cd "$IIIL_DIR"
        if [[ -z "$demo_input" || "$demo_input" == "all" || "$demo_input" == "ALL" ]]; then
            cmd="python scripts/data/plot_joint_traces.py ${full_path}"
            echo -e "${BLUE}Will plot all demos.${NC}"
        else
            cmd="python scripts/data/plot_joint_traces.py ${full_path} --demo ${demo_input}"
            echo -e "${BLUE}Will plot ${demo_input}.${NC}"
        fi

        echo ""
        echo -e "${GREEN}Plotting joint traces...${NC}"
        echo -e "File: ${BLUE}${full_path}${NC}"
        echo ""
        echo -e "${YELLOW}Running: ${cmd}${NC}"
        echo ""
        exec $cmd
        ;;

    5|eval|Evaluation)
        # ==================== EVALUATION ====================
        conda activate iiil
        check_cameras
        echo ""
        echo -e "${GREEN}=== Evaluation Mode ===${NC}"
        echo ""
        cmd="python run_task.py env=real_r1pro_joint task=real/base policy=il reset_robot_on_env_reset=true"
        echo -e "${YELLOW}Running: ${cmd}${NC}"
        echo ""
        exec $cmd
        ;;

    6)
        # ==================== SIMULATION TELEOPERATION ====================
        conda deactivate 2>/dev/null
        conda activate behavior
        echo ""
        echo -e "${GREEN}=== Simulation Teleoperation Mode ===${NC}"
        echo ""
        echo "Enter save file name (relative to ${DATA_DIR}):"
        echo "  Examples: mydata.hdf5, test/filename.hdf5"
        read -p "Filename: " filename

        # Add .hdf5 extension if not present
        if [[ ! "$filename" == *.hdf5 ]]; then
            filename="${filename}.hdf5"
        fi

        # Construct full path
        if [[ "$filename" == /* ]]; then
            full_path="$filename"
        else
            full_path="${DATA_DIR}/${filename}"
        fi

        # Create parent directory if it doesn't exist
        parent_dir=$(dirname "$full_path")
        if [[ ! -d "$parent_dir" ]]; then
            echo -e "${YELLOW}Directory $parent_dir does not exist. Creating...${NC}"
            mkdir -p "$parent_dir"
        fi

        # Check if file exists
        overwrite_flag="false"
        if [[ -f "$full_path" ]]; then
            echo ""
            echo -e "${YELLOW}WARNING: File already exists: ${full_path}${NC}"
            read -p "Do you want to overwrite? [y/N]: " overwrite_choice
            case $overwrite_choice in
                [Yy]*)
                    overwrite_flag="true"
                    echo -e "${YELLOW}Will overwrite existing file.${NC}"
                    ;;
                *)
                    echo -e "${RED}Aborted. Please choose a different filename.${NC}"
                    exit 1
                    ;;
            esac
        fi

        echo ""
        echo -e "${GREEN}Starting simulation teleoperation...${NC}"
        echo -e "Saving to: ${BLUE}${full_path}${NC}"
        echo ""

        cmd="python run_task.py env=sim_r1pro_joint task=open/dc policy=gello policy.base_policy=dummy policy.control_direction=gello_to_env policy.require_joycon=true hdf_path=${full_path} overwrite=${overwrite_flag}"
        echo -e "${YELLOW}Running: ${cmd}${NC}"
        echo ""
        exec $cmd
        ;;

    7)
        # ==================== SIMULATION ROLLOUT ====================
        conda deactivate 2>/dev/null
        conda activate behavior
        echo ""
        echo -e "${GREEN}=== Simulation Rollout Mode ===${NC}"
        echo ""
        echo "Running Gello with CuRobo motion planning in simulation..."
        echo ""

        cmd="python run_task.py env=sim_r1pro_joint task=open/base policy=gello policy.base_policy=curobo policy.require_joycon=false policy.takeover_enable=true"
        echo -e "${YELLOW}Running: ${cmd}${NC}"
        echo ""
        exec $cmd
        ;;

    8)
        # ==================== SIMULATION ROLLOUT (IL) ====================
        conda deactivate 2>/dev/null
        conda activate behavior
        echo ""
        echo -e "${GREEN}=== Simulation Rollout (IL) ===${NC}"
        echo ""

        # Start serve.py in background (IL policy server on port 8000)
        echo -e "${GREEN}Starting policy server (serve.py) in background...${NC}"
        cd "$IL_LIB_DIR"
        serve_cmd="python serve.py arch=wbvima_r1pro_state robot=r1pro task=open/base ckpt_path=${IL_LIB_DIR}/open_state_150k-100.pth"
        echo -e "${YELLOW}Running: ${serve_cmd}${NC}"
        $serve_cmd &
        SERVE_PID=$!
        echo -e "${BLUE}Policy server PID: ${SERVE_PID}${NC}"
        echo ""

        # Wait for server to start
        echo -e "${YELLOW}Waiting for policy server to start...${NC}"
        sleep 5

        echo -e "${GREEN}Starting simulation rollout with IL policy...${NC}"
        cd "$IIIL_DIR"
        cmd="python run_task.py env=sim_r1pro_joint task=open/base policy=gello policy.base_policy=il policy.require_joycon=false policy.takeover_enable=true"
        echo -e "${YELLOW}Running: ${cmd}${NC}"
        echo ""

        $cmd
        TASK_EXIT=$?

        # Cleanup: kill the server
        echo ""
        echo -e "${YELLOW}Stopping policy server (PID: ${SERVE_PID})...${NC}"
        kill $SERVE_PID 2>/dev/null
        wait $SERVE_PID 2>/dev/null

        echo -e "${GREEN}Done.${NC}"
        exit $TASK_EXIT
        ;;

    9|correction|Correction)
        # ==================== CORRECTION (IL + Gello + Data Collection) ====================
        conda activate iiil
        check_cameras
        echo ""
        echo -e "${GREEN}=== Policy Correction Mode (IL + Gello + Data Collection) ===${NC}"
        echo ""
        echo "Enter save file name (relative to ${DATA_DIR}):"
        echo "  Examples: mydata.hdf5, test/filename.hdf5"
        read -p "Filename: " filename

        # Add .hdf5 extension if not present
        if [[ ! "$filename" == *.hdf5 ]]; then
            filename="${filename}.hdf5"
        fi

        # Construct full path
        if [[ "$filename" == /* ]]; then
            full_path="$filename"
        else
            full_path="${DATA_DIR}/${filename}"
        fi

        # Create parent directory if it doesn't exist
        parent_dir=$(dirname "$full_path")
        if [[ ! -d "$parent_dir" ]]; then
            echo -e "${YELLOW}Directory $parent_dir does not exist. Creating...${NC}"
            mkdir -p "$parent_dir"
        fi

        # Check if file exists
        overwrite_flag="false"
        if [[ -f "$full_path" ]]; then
            echo ""
            echo -e "${YELLOW}WARNING: File already exists: ${full_path}${NC}"
            read -p "Do you want to overwrite? [y/N]: " overwrite_choice
            case $overwrite_choice in
                [Yy]*)
                    overwrite_flag="true"
                    echo -e "${YELLOW}Will overwrite existing file.${NC}"
                    ;;
                *)
                    echo -e "${RED}Aborted. Please choose a different filename.${NC}"
                    exit 1
                    ;;
            esac
        fi

        echo ""
        echo -e "${GREEN}Starting correction data collection (IL + admittance + takeover)...${NC}"
        echo -e "Saving to: ${BLUE}${full_path}${NC}"
        echo ""

        cmd="python run_task.py env=real_r1pro_joint task=real/base policy=gello policy.base_policy=il policy.require_joycon=true policy.takeover_enable=true mode=collection enforce_frequency=true hdf_path=${full_path} overwrite=${overwrite_flag}"
        echo -e "${YELLOW}Running: ${cmd}${NC}"
        echo ""
        exec $cmd
        ;;

    10|franka-dc|franka_dc)
        # ==================== FRANKA DATA COLLECTION ====================
        # Uses behavior env (has omnigibson + IIIL). Franka doesn't need ROS2.
        conda deactivate 2>/dev/null
        conda activate behavior

        # Per-session hardware startup: dependencies, Joylo/cameras, NUC
        # services, Franka Desk/FCI, and an end-to-end robot state check.
        prepare_real_franka

        echo ""
        echo -e "${GREEN}=== Franka Data Collection Mode ===${NC}"
        echo ""
        echo "Enter save file name (relative to ${DATA_DIR}):"
        echo "  Examples: franka_demos.hdf5, franka/test.hdf5"
        read -p "Filename: " filename

        # Add .hdf5 extension if not present
        if [[ ! "$filename" == *.hdf5 ]]; then
            filename="${filename}.hdf5"
        fi

        # Construct full path
        if [[ "$filename" == /* ]]; then
            full_path="$filename"
        else
            full_path="${DATA_DIR}/${filename}"
        fi

        # Create parent directory if it doesn't exist
        parent_dir=$(dirname "$full_path")
        if [[ ! -d "$parent_dir" ]]; then
            echo -e "${YELLOW}Directory $parent_dir does not exist. Creating...${NC}"
            mkdir -p "$parent_dir"
        fi

        # Check if file exists
        overwrite_flag="false"
        if [[ -f "$full_path" ]]; then
            echo ""
            echo -e "${YELLOW}WARNING: File already exists: ${full_path}${NC}"
            read -p "Do you want to overwrite? [y/N]: " overwrite_choice
            case $overwrite_choice in
                [Yy]*)
                    overwrite_flag="true"
                    echo -e "${YELLOW}Will overwrite existing file.${NC}"
                    ;;
                *)
                    echo -e "${RED}Aborted. Please choose a different filename.${NC}"
                    exit 1
                    ;;
            esac
        fi

        snapshot_franka_scene "$full_path"

        echo -e "${GREEN}Starting Franka data collection (gello teleop)...${NC}"
        echo -e "Saving to: ${BLUE}${full_path}${NC}"
        echo ""

        cmd="python run_task.py env=real_franka_joint task=real/dc policy=gello policy.gello_config=gello_franka policy.base_policy=dummy policy.control_direction=gello_to_env policy.require_joycon=true policy.takeover_enable=false mode=collection env.cfg.camera_viewer.enabled=true hdf_path=${full_path} overwrite=${overwrite_flag}"
        echo -e "${YELLOW}Running: ${cmd}${NC}"
        echo ""
        exec $cmd
        ;;

    11|franka-correction|franka_correction)
        # ==================== FRANKA POLICY CORRECTION ====================
        # Uses behavior env (has omnigibson + IIIL). Franka doesn't need ROS2.
        conda deactivate 2>/dev/null
        conda activate behavior

        # Per-session hardware startup: dependencies, Joylo/cameras, NUC
        # services, Franka Desk/FCI, and an end-to-end robot state check.
        prepare_real_franka

        echo ""
        echo -e "${GREEN}=== Franka Policy Correction Mode (IL + Admittance + Touch Takeover) ===${NC}"
        echo ""
        echo "Enter save file name (relative to ${DATA_DIR}):"
        echo "  Examples: franka_corrections.hdf5, franka/corrections.hdf5"
        read -p "Filename: " filename

        # Add .hdf5 extension if not present
        if [[ ! "$filename" == *.hdf5 ]]; then
            filename="${filename}.hdf5"
        fi

        # Construct full path
        if [[ "$filename" == /* ]]; then
            full_path="$filename"
        else
            full_path="${DATA_DIR}/${filename}"
        fi

        # Create parent directory if it doesn't exist
        parent_dir=$(dirname "$full_path")
        if [[ ! -d "$parent_dir" ]]; then
            echo -e "${YELLOW}Directory $parent_dir does not exist. Creating...${NC}"
            mkdir -p "$parent_dir"
        fi

        # Check if file exists
        overwrite_flag="false"
        if [[ -f "$full_path" ]]; then
            echo ""
            echo -e "${YELLOW}WARNING: File already exists: ${full_path}${NC}"
            read -p "Do you want to overwrite? [y/N]: " overwrite_choice
            case $overwrite_choice in
                [Yy]*)
                    overwrite_flag="true"
                    echo -e "${YELLOW}Will overwrite existing file.${NC}"
                    ;;
                *)
                    echo -e "${RED}Aborted. Please choose a different filename.${NC}"
                    exit 1
                    ;;
            esac
        fi

        snapshot_franka_scene "$full_path"

        echo -e "${GREEN}Starting Franka correction data collection (IL + admittance + touch takeover)...${NC}"
        echo -e "Saving to: ${BLUE}${full_path}${NC}"
        echo ""

        cmd="python run_task.py env=real_franka_joint task=real/base policy=gello policy.gello_config=gello_franka policy.base_policy=il policy.require_joycon=true policy.takeover_enable=true mode=collection enforce_frequency=true env.cfg.camera_viewer.enabled=true env.cfg.camera_viewer.show_takeover_count=true hdf_path=${full_path} overwrite=${overwrite_flag}"
        echo -e "${YELLOW}Running: ${cmd}${NC}"
        echo ""
        exec $cmd
        ;;

    12|franka-sim|franka_sim)
        # ==================== FRANKA SIM TELEOP ====================
        # MuJoCo sim Franka driven by Gello through the full I3L pipeline.
        # No deoxys or real robot needed — just the Joylo hardware.
        conda deactivate 2>/dev/null
        conda activate behavior

        # Ensure franka_joylo is importable
        FRANKA_JOYLO_DIR=$(dirname "$IIIL_DIR")/FrankaJoylo
        if ! python -c "import franka_joylo" 2>/dev/null; then
            echo -e "${YELLOW}Installing franka_joylo...${NC}"
            if [[ -d "$FRANKA_JOYLO_DIR" ]]; then
                python -m pip install -e "$FRANKA_JOYLO_DIR" -q
            else
                echo -e "${RED}ERROR: FrankaJoylo not found at ${FRANKA_JOYLO_DIR}${NC}"
                exit 1
            fi
        fi

        echo ""
        echo -e "${GREEN}=== Franka Sim Teleop (MuJoCo + Gello) ===${NC}"
        echo ""
        echo "Launching MuJoCo sim Franka with Gello teleop through the I3L pipeline."
        echo "Move the Joylo to control the simulated Franka."
        echo ""

        cmd="python run_task.py env=sim_franka_joint task=real/base policy=gello policy.gello_config=gello_franka policy.base_policy=sine policy.control_direction=gello_to_env policy.require_joycon=false policy.takeover_enable=false"
        echo -e "${YELLOW}Running: ${cmd}${NC}"
        echo ""
        exec $cmd
        ;;

    13|franka-sim-correction|franka_sim_correction)
        # ==================== FRANKA SIM CORRECTION TEST ====================
        # Simulates mode 11 (PolicyCorrection) entirely in sim.
        # A sine-wave policy drives the MuJoCo Franka, Gello follows in
        # admittance mode. Push on the Joylo to test takeover detection.
        conda deactivate 2>/dev/null
        conda activate behavior

        # Ensure franka_joylo is importable
        FRANKA_JOYLO_DIR=$(dirname "$IIIL_DIR")/FrankaJoylo
        if ! python -c "import franka_joylo" 2>/dev/null; then
            echo -e "${YELLOW}Installing franka_joylo...${NC}"
            if [[ -d "$FRANKA_JOYLO_DIR" ]]; then
                python -m pip install -e "$FRANKA_JOYLO_DIR" -q
            else
                echo -e "${RED}ERROR: FrankaJoylo not found at ${FRANKA_JOYLO_DIR}${NC}"
                exit 1
            fi
        fi

        echo ""
        echo -e "${GREEN}=== Franka Sim Correction Test (Sine Policy + Admittance + Gello) ===${NC}"
        echo ""
        echo "Sine-wave policy drives sim Franka. Gello follows in admittance mode."
        echo "Push the Joylo to test takeover detection."
        echo ""

        cmd="python run_task.py env=sim_franka_joint task=real/base policy=gello policy.gello_config=gello_franka policy.base_policy=sine policy.control_direction=admittance policy.require_joycon=false policy.takeover_enable=true"
        echo -e "${YELLOW}Running: ${cmd}${NC}"
        echo ""
        exec $cmd
        ;;

    *)
        echo -e "${RED}Invalid choice.${NC}"
        exit 1
        ;;
esac
