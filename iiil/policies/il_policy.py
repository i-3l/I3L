import torch
from copy import deepcopy
from iiil.policies.policy_base import BasePolicy
from omegaconf import DictConfig, OmegaConf
from iiil.utils.obs_utils import crop_camera_obs
from omnigibson.learning.policies import LocalPolicy, WebsocketPolicy
from omnigibson.learning.utils.network_utils import WebsocketClientPolicy, unpackb
import websockets


class EmbeddingWebsocketClientPolicy(WebsocketClientPolicy):
	"""WebsocketClientPolicy that also captures obs_embedding from server."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._last_obs_embedding = None

	def act(self, obs):
		if self._ws is None:
			self._ws, self._server_metadata = self._wait_for_server()

		data = self._packer.pack(obs)
		max_retries = 2

		for attempt in range(max_retries + 1):
			try:
				self._ws.send(data)
				response = self._ws.recv()
				if isinstance(response, str):
					raise RuntimeError(f"Error in inference server:\n{response}")
				action_dict = unpackb(response)
				if "action" not in action_dict:
					if attempt < max_retries:
						continue
					raise RuntimeError(f"Server response missing 'action' key: {action_dict}")
				self._last_obs_embedding = action_dict.get("obs_embedding", None)
				return torch.from_numpy(deepcopy(action_dict["action"])).to(torch.float32)
			except websockets.exceptions.ConnectionClosedError as e:
				if self._allow_reconnect and attempt < max_retries:
					self._ws, self._server_metadata = self._wait_for_server()
					continue
				raise RuntimeError(f"Websocket connection error: {e}")


class EmbeddingWebsocketPolicy(WebsocketPolicy):
	"""WebsocketPolicy that uses EmbeddingWebsocketClientPolicy."""

	def __init__(self, *args, host=None, port=None, allow_reconnect=False, **kwargs):
		super().__init__(*args, host=host, port=port, allow_reconnect=allow_reconnect, **kwargs)
		if host is not None or port is not None:
			self.policy = EmbeddingWebsocketClientPolicy(host=host, port=port, allow_reconnect=allow_reconnect)

	@property
	def last_obs_embedding(self):
		if self.policy is not None and hasattr(self.policy, '_last_obs_embedding'):
			return self.policy._last_obs_embedding
		return None


class ImitationLearningPolicy(BasePolicy, LocalPolicy, EmbeddingWebsocketPolicy):
	"""
	Vision-based imitation learning policy
	"""
	def __init__(self, config: DictConfig) -> None:
		BasePolicy.__init__(self, config)
		self.ema_alpha = config.ema_alpha
		self.prev_action = None
		cfg = OmegaConf.to_container(config)
		if config.use_websocket:
			EmbeddingWebsocketPolicy.__init__(self, **cfg)
		else:
			LocalPolicy.__init__(self, **cfg)

	def predict(self, obs: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
		# Crop camera obs before sending to model — raw obs are saved
		# independently by the data collection wrapper.
		# Skip crop for websocket mode: the server's process_obs handles resizing.
		obs = {k: v.clone() for k, v in obs.items()}
		if not self.config.use_websocket:
			for k, v in obs.items():
				if k.endswith("rgb"):
					camera_name = k.split("::")[-2]
					obs[k] = crop_camera_obs(v, camera_name)
		if self.config.use_websocket:
			action = EmbeddingWebsocketPolicy.forward(self, obs, **kwargs)
		else:
			action = LocalPolicy.forward(self, obs, **kwargs)
		# optionally apply exponential moving average smoothing
		if self.prev_action is None:
			self.prev_action = action
		if self.ema_alpha is not None:
			action = self.ema_alpha * action + (1 - self.ema_alpha) * self.prev_action
		self.prev_action = action
		return action

	def load(self, **kwargs):
		pass

	def reset(self) -> None:
		if self.config.use_websocket:
			EmbeddingWebsocketPolicy.reset(self)
		else:
			LocalPolicy.reset(self)
		self.prev_action = None

	def close(self) -> None:
		pass