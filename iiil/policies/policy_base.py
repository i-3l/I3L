import os
import torch
from enum import Enum
from iiil.utils.misc_utils import BASE_PATHS
from omegaconf import DictConfig
from typing import Dict, Optional


class PolicyMode(Enum):
	"""
	Enum for policy modes
	"""
	LEADING = "leading"
	FOLLOWING = "following"

	@classmethod
	def from_string(cls, mode: str) -> "PolicyMode":
		"""
		Returns the appropriate enum value for a given string.

		Args:
			mode (str): The string representation of the mode.

		Returns:
			PolicyMode: The corresponding enum value.
		"""
		if mode == cls.LEADING.value:
			return cls.LEADING
		elif mode == cls.FOLLOWING.value:
			return cls.FOLLOWING
		else:
			raise ValueError(f"Invalid policy mode: {mode}")


class BasePolicy():
	"""
	Base policy
	"""
	# Override to True in policies that support X/A/Y button trial control
	supports_trial_control: bool = False

	def __init__(self, config: DictConfig) -> None:
		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.base_path = None
		self._debug_info = dict()
		for path in BASE_PATHS:
			if os.path.exists(path):
				self.base_path = path
				break
		
	def predict(self, obs: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
		"""
		Given the neural signal, predict the intented action
		Args:
			obs (dict[str, torch.Tensor]): observation dictionary from the environment
			**kwargs: additional arguments
		Returns:
			torch.Tensor: the predicted action
		"""
		raise NotImplementedError('Subclasses need to implement this')
		
	def load(self, **kwargs):
		raise NotImplementedError('Subclasses need to implement this')

	def reset(self) -> None:
		raise NotImplementedError('Subclasses need to implement this')

	def close(self) -> None:
		"""
		Function to be called when the policy is exiting
		"""
		raise NotImplementedError('Subclasses need to implement this')
	
	def check_block_ended(self) -> bool:
		"""
		Check if the block has ended
		Returns:
			bool: True if the block has ended, False otherwise
		"""
		return False
	
	def check_episode_ended(self, success: bool=True) -> bool:
		"""
		Check if the episode has ended
		Args:
			success (bool): True if the episode was successful, False otherwise
		Returns:
			bool: True if the episode has ended, False otherwise
		"""
		return False
	
	def get_current_state(self) -> Optional[Dict[str, torch.Tensor]]:
		"""
		Generate state information for logging/visualization
		"""
		return None
	
	def set_mode(self, mode: PolicyMode) -> None:
		"""
		Notifies the policy that it is now the leading or following policy.

		By default, this does nothing.
		Args:
			mode (PolicyMode): The mode to set the policy to
		"""
		pass

	def on_env_reset(self, obs: dict[str, torch.Tensor]) -> None:
		"""
		Hook invoked when the environment resets. Policies can override to
		reinitialize hardware or internal state using the first observation.
		"""
		pass
