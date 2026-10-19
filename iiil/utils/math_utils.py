import omnigibson.utils.transform_utils as T
import torch


def add_position_noise(pos: torch.Tensor, noise_std: float = 0.01) -> torch.Tensor:
    """
    Add Gaussian noise to a position.
    
    Args:
        pos: Input position
        noise_std: Standard deviation of position noise
    
    Returns:
        Noisy position
    """
    noise = torch.randn_like(pos) * noise_std
    return pos + noise


def add_quaternion_noise(quat: torch.Tensor, noise_std: float = 0.05) -> torch.Tensor:
    """
    Add noise to a quaternion by applying a small random rotation.
    
    Args:
        quat: Input quaternion [w, x, y, z]
        noise_std: Standard deviation of rotation noise in radians
    
    Returns:
        Noisy quaternion (normalized)
    """
    # Generate small random Euler angles
    angle_noise = torch.randn(3) * noise_std
    
    # Convert to quaternion (axis-angle to quaternion)
    noise_quat = T.axisangle2quat(angle_noise)
    return T.quat_multiply(noise_quat, quat)
