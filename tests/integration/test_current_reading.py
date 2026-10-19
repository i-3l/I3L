import math
import numpy as np
import time
import traceback
from iiil.gello.drivers.current_control_driver import DXLCurrentControlDriver
from iiil.gello.drivers.velocity_control_driver import DXLVelocityControlDriver
from iiil.gello.drivers.position_control_driver import DXLPositionControlDriver
from iiil.gello.motor_type import MotorType
from typing import Literal


def radians_to_dxl_encoder(radians: float, initial_encoder_ticks: int = 2048) -> int:
    """
    Convert radians to DXL encoder ticks.
    
    Args:
        radians: Angle in radians
        initial_encoder_ticks: Initial encoder position (default 2048 for center)
    
    Returns:
        DXL encoder ticks
    """
    # DXL encoder has 4096 ticks per revolution
    # 2048 ticks = 0 radians (center position)
    encoder_ticks = int(radians * 2048 / math.pi + initial_encoder_ticks)
    return encoder_ticks


def dxl_encoder_to_radians(encoder_ticks: int, initial_encoder_ticks: int = 2048) -> float:
    """
    Convert DXL encoder ticks to radians.
    
    Args:
        encoder_ticks: DXL encoder ticks
        initial_encoder_ticks: Initial encoder position (default 2048 for center)
    
    Returns:
        Angle in radians
    """
    radians = (encoder_ticks - initial_encoder_ticks) * math.pi / 2048
    return radians


def main(
    motor_id: int,
    motor_type: MotorType,
    port: str,
    output_file: str,
    control_type: Literal["current", "velocity", "position", "current_based_position"],
    constant_command: float,
    baudrate: int = 3000000,
):
    """
    Test function that initializes a single motor and applies a constant command
    while recording readings.
    
    Args:
        motor_id: ID of the motor to control
        motor_type: Type of motor (from MotorType enum)
        port: Serial port to connect to
        output_file: Path to HDF5 file to save data
        control_type: Type of control ("current", "velocity", "position", or "current_based_position")
        constant_command: Constant command value to apply 
            (current in Nm, velocity in rad/s, or position in rad)
        baudrate: Baudrate for serial communication (default 3000000)
    """
    # Lists to store data
    timestamps = []
    current_readings = []
    position_readings = []
    velocity_readings = []
    sent_commands = []
    
    print(f"Initializing motor at ID {motor_id}")
    print(f"Motor type: {motor_type.value.motor_name}")
    print(f"Port: {port}")
    print(f"Control type: {control_type}")
    print(f"Constant command: {constant_command}")
    
    try:
        # Initialize the appropriate control driver
        if control_type == "current":
            driver = DXLCurrentControlDriver(
                ids=[motor_id],
                port=port,
                motor_type=motor_type,
                baudrate=baudrate,
                multithread_read_joints=False,
            )
            command_units = "Nm"
        elif control_type == "velocity":
            driver = DXLVelocityControlDriver(
                ids=[motor_id],
                port=port,
                motor_type=motor_type,
                baudrate=baudrate,
                multithread_read_joints=False,
            )
            command_units = "rad/s"
        elif control_type == "position":
            driver = DXLPositionControlDriver(
                ids=[motor_id],
                port=port,
                motor_type=motor_type,
                baudrate=baudrate,
                multithread_read_joints=False,
            )
            command_units = "rad"

            # Convert joint angles from radians to DXL encoder ticks
            # Assuming 2048 ticks per revolution (typical for Dynamixel X series)
            constant_command = np.array(constant_command, dtype=np.int32) * 2048 / (2 * np.pi)

        elif control_type == "current_based_position":
            driver = DXLPositionControlDriver(
                ids=[motor_id],
                port=port,
                motor_type=motor_type,
                baudrate=baudrate,
                multithread_read_joints=False,
                current_limit=10,
            )
            command_units = "rad"
            
        print("Motor driver initialized successfully")
        
        # Enable torque
        driver.set_torque_mode(True)
        print("Torque enabled")
        
        # Get initial position
        initial_state = driver.get_motor_state(block=True)
        initial_position = dxl_encoder_to_radians(initial_state.joint_angles[0])
        print(f"Initial position: {initial_position:.3f} rad ({math.degrees(initial_position):.1f}°)")
        
        # Start control
        print("\nStarting control...")
        print(f"Constant command: {constant_command} {command_units}")
        print("Press Ctrl+C to stop")
        
        start_time = time.time()
        
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Get current state
            state = driver.get_motor_state(block=False)
            current_position = dxl_encoder_to_radians(state.joint_angles[0])
            current_velocity = state.joint_velocities[0] * 0.229 * 2 * math.pi / 60  # Convert to rad/s
            current = state.joint_currents[0]  # Get current reading in mA
            
            # Send command
            driver.set_controls([constant_command])
            
            # Store readings
            timestamps.append(elapsed_time)
            current_readings.append(current)
            position_readings.append(current_position)
            velocity_readings.append(current_velocity)
            sent_commands.append(constant_command)
            
            # Print status every 0.5 seconds
            if int(current_time * 2) % 2 == 0:  # Every 0.5 seconds
                print(f"Time: {elapsed_time:.1f}s | Position: {current_position:.3f} rad | Velocity: {current_velocity:.3f} rad/s | Current: {current:.1f} mA")
            
            # Small delay to avoid overwhelming the motor
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nStopping control...")
    except Exception as e:
        print(f"Error: {e}")
        print("Traceback:")
        print(traceback.format_exc())
    finally:
        # Save data to HDF5 file
        import h5py
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('timestamps', data=timestamps)
            f.create_dataset('current_readings', data=current_readings)
            f.create_dataset('position_readings', data=position_readings)
            f.create_dataset('velocity_readings', data=velocity_readings)
            f.create_dataset('sent_commands', data=sent_commands)
            f.attrs['control_type'] = control_type
            f.attrs['constant_command'] = constant_command
            f.attrs['command_units'] = command_units
        print(f"Data saved to {output_file}")

        # Disable torque and close driver
        driver.set_torque_mode(False)
        print("Torque disabled")
        driver.close()
        print("Motor driver closed")

if __name__ == "__main__":
    main(
        motor_id=1,
        motor_type=MotorType.XM430_W350_T,
        port="/dev/tty_large_motor",
        output_file="position_grav_load.h5",
        control_type="position",
        constant_command=0.0,
        # constant_command=np.pi,  # 0.01 Nm torque command
    )

