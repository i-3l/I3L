import numpy as np
import torch
from iiil.policies.policy_base import BasePolicy
from omegaconf import DictConfig
from typing import List


class RandomPositionPolicy(BasePolicy):
    """
    Policy that continuously samples random 3D positions within a cube and moves the robot to them.
    Optionally samples random velocities for each target.
    """
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        # Cube bounds: [x_min, x_max], [y_min, y_max], [z_min, z_max]
        self.cube_bounds = config.cube_bounds  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        self.velocity_lower_bound = config.velocity_lower_bound
        self.velocity_upper_bound = config.velocity_upper_bound
        self.random_velocity = getattr(config, 'random_velocity', False)
        self.target_pos = None
        self.current_velocity = None
        self.current_rotation_velocity = None
        self.waypoint_reached_threshold = config.waypoint_reached_threshold
        self.steps_since_rotation_update = 0
        self.rotation_update_freq = 5
        self._sample_new_target()

    def predict(self, obs: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        current_pos = obs['task']['eef_pos'].cpu().numpy()
        delta_pos = self.target_pos - current_pos
        distance = np.linalg.norm(delta_pos)
        if distance < self.waypoint_reached_threshold:
            self._sample_new_target()
            delta_pos = self.target_pos - current_pos
            distance = np.linalg.norm(delta_pos)
        # Scale delta by velocity
        if distance > 0:
            delta_pos = delta_pos * (self.current_velocity / distance)

        # Update rotation velocity periodically
        self.steps_since_rotation_update += 1
        if self.steps_since_rotation_update >= self.rotation_update_freq:
            if self.random_velocity:
                self.current_rotation_velocity = np.random.uniform(-0.1, 0.1, size=3)
            self.steps_since_rotation_update = 0

        delta_rot = self.current_rotation_velocity
        delta_pose = np.concatenate([delta_pos, delta_rot, np.zeros(1)])
        return torch.tensor(delta_pose, dtype=torch.float32, device=self.device)

    def _sample_new_target(self):
        self.target_pos = np.array([
            np.random.uniform(*self.cube_bounds[0]),
            np.random.uniform(*self.cube_bounds[1]),
            np.random.uniform(*self.cube_bounds[2])
        ])
        print(f"New target position: {self.target_pos}")
        if self.random_velocity:
            self.current_velocity = np.random.uniform(self.velocity_lower_bound, self.velocity_upper_bound)
            self.current_rotation_velocity = np.random.uniform(-0.1, 0.1, size=3)
        else:
            self.current_velocity = self.velocity_upper_bound
            self.current_rotation_velocity = np.zeros(3)
        self.steps_since_rotation_update = 0

    def load(self, **kwargs):
        self._sample_new_target()

    def reset(self) -> None:
        self._sample_new_target()

    def close(self) -> None:
        pass

    def check_episode_ended(self, success: bool = True) -> bool:
        return False  # Never ends by itself

    def get_extra_info(self) -> np.ndarray:
        return np.array([
            *self.target_pos,
            self.current_velocity
        ]) 
    

class WaypointPolicy(BasePolicy):
    """
    Waypoint policy that follows a list of 3D waypoints.
    Outputs delta end-effector poses (3D position + 3 euler angles).
    """
    
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        
        # Load waypoint configuration directly from config
        self.waypoints = config.waypoints  # List of [x, y, z] waypoints
        self.loop = config.loop  # Whether to loop the trajectory
        self.current_waypoint_idx = 0
        self.waypoint_reached_threshold = config.waypoint_reached_threshold
        
        # Movement parameters
        self.velocity_lower_bound = config.velocity_lower_bound
        self.velocity_upper_bound = config.velocity_upper_bound
        
        # State tracking
        self.current_pos = None
        self.target_pos = None
        self.current_velocity = None
        self.steps_since_velocity_update = 0
        self.velocity_update_freq = 20  # Update velocity every 20 steps
        
        # Initialize target position from first waypoint
        if self.waypoints:
            self.target_pos = np.array(self.waypoints[0])
            self._sample_new_velocity()
    
    def predict(self, obs: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Predict delta end-effector pose to move towards current waypoint.
        
        Args:
            obs (dict[str, torch.Tensor]): observation dictionary containing EEF pose
            **kwargs: additional arguments
            
        Returns:
            torch.Tensor: delta pose [dx, dy, dz, droll, dpitch, dyaw]
        """
        # Extract current EEF position from observation
        current_pos = obs['robot::proprio'][8:11].cpu().numpy()
        self.current_pos = current_pos
        
        # Check if we need to update target waypoint
        self._update_target_waypoint()
        
        # Update velocity periodically
        self.steps_since_velocity_update += 1
        if self.steps_since_velocity_update >= self.velocity_update_freq:
            self._sample_new_velocity()
            self.steps_since_velocity_update = 0
        
        # Calculate delta position to target
        if self.target_pos is not None:
            delta_pos = self.target_pos - current_pos
            distance = np.linalg.norm(delta_pos)
            
            # Scale based on sampled velocity
            if distance > 0:
                delta_pos = delta_pos * (self.current_velocity / distance)
            
            # No rotation changes (keep current orientation)
            delta_rot = np.zeros(3)  # [droll, dpitch, dyaw]
            
            # Combine position and rotation deltas
            delta_pose = np.concatenate([delta_pos, delta_rot, np.zeros(1)])
        else:
            # No waypoints or all waypoints reached
            delta_pose = np.zeros(7)
        
        return torch.tensor(delta_pose, dtype=torch.float32, device=self.device)
    
    def _sample_new_velocity(self):
        """Sample a new velocity between lower and upper bounds."""
        self.current_velocity = np.random.uniform(
            self.velocity_lower_bound,
            self.velocity_upper_bound
        )
    
    def _update_target_waypoint(self):
        """Update the target waypoint based on current position and loop setting."""
        if not self.waypoints or self.target_pos is None:
            return
        
        # Check if current waypoint is reached
        distance_to_target = np.linalg.norm(self.current_pos - self.target_pos)
        
        if distance_to_target < self.waypoint_reached_threshold:
            # Move to next waypoint
            if self.current_waypoint_idx < len(self.waypoints) - 1:
                self.current_waypoint_idx += 1
            elif self.loop:
                # Loop back to first waypoint
                self.current_waypoint_idx = 0
            else:
                # Stay at last waypoint if not looping
                return
            
            # Update target position
            self.target_pos = np.array(self.waypoints[self.current_waypoint_idx])
    
    def load(self, **kwargs):
        """
        Load waypoints from configuration or YAML file.
        
        Args:
            **kwargs: additional arguments
        """        
        # Reset state
        self.current_waypoint_idx = 0
        if self.waypoints:
            self.target_pos = np.array(self.waypoints[0])
            
    def reset(self) -> None:
        """Reset the policy state."""
        self.current_waypoint_idx = 0
        self.current_pos = None
        self.steps_since_velocity_update = 0
        if self.waypoints:
            self.target_pos = np.array(self.waypoints[0])
            self._sample_new_velocity()
    
    def close(self) -> None:
        """Clean up resources."""
        pass

    def check_episode_ended(self, success: bool = True) -> bool:
        """
        Check if the episode has ended.
        For waypoint policy, episode ends when all waypoints are reached (if not looping).
        """
        if not self.waypoints:
            return True
        
        if self.loop:
            # Never end if looping
            return False
        
        # End if we've reached the last waypoint
        return (self.current_waypoint_idx == len(self.waypoints) - 1 and
                self.target_pos is not None and
                np.linalg.norm(self.current_pos - self.target_pos) < self.waypoint_reached_threshold)
    
    def get_extra_info(self) -> np.ndarray:
        """Get extra information for logging/visualization."""
        if self.waypoints:
            return np.array([
                self.current_waypoint_idx,
                len(self.waypoints),
                np.linalg.norm(self.current_pos - self.target_pos) if self.target_pos is not None else 0.0
            ])
        return np.array([])
    
    def set_waypoints(self, waypoints: List[List[float]]):
        """Set the waypoints programmatically."""
        self.waypoints = waypoints
        self.reset()
    
    def get_progress(self) -> float:
        """Get progress through waypoints (0.0 to 1.0)."""
        if not self.waypoints:
            return 0.0
        
        if self.loop:
            # For looping, progress is based on current waypoint index
            return self.current_waypoint_idx / len(self.waypoints)
        # For non-looping, progress is linear
        return self.current_waypoint_idx / (len(self.waypoints) - 1)