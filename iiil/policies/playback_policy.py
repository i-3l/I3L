import h5py
import os
import time
import torch
from iiil.policies.policy_base import BasePolicy
from omnigibson.utils.ui_utils import create_module_logger

logger = create_module_logger("PlaybackPolicy")


class PlaybackPolicy(BasePolicy):
	def __init__(self, config):
		super().__init__(config)
		self.hdf5 = None
		self.demo_ids = []
		self.demo_index = -1
		self.step_counter = 0
		self.cur_episode_length = 0
		self.action_repeat_counter = 0
		self.repeat_count = 1
		self.current_repeat = 0
		self._all_demos_played = False
		self._last_step_time = None
		self._step_period = None  # seconds per step for fixed rate limiting
		self._timestamps = None   # timestamps from HDF5 (ms)
		self._use_timestamps = False
		self._episode_start_time = None

	@property
	def current_demo_id(self):
		return self.demo_ids[self.demo_index]

	def predict(self, obs: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
		# Rate limiting using recorded timestamps (preferred) or fixed frequency
		if self._use_timestamps and self._timestamps is not None:
			# Use recorded timestamps for exact replay timing
			if self._episode_start_time is None:
				self._episode_start_time = time.time()
			# Target time for current step (convert ms to seconds)
			target_elapsed = self._timestamps[min(self.step_counter, len(self._timestamps) - 1)] / 1000.0
			actual_elapsed = time.time() - self._episode_start_time
			sleep_time = target_elapsed - actual_elapsed
			if sleep_time > 0:
				time.sleep(sleep_time)
		elif self._step_period is not None and self._last_step_time is not None:
			# Fallback: fixed frequency rate limiting
			elapsed = time.time() - self._last_step_time
			sleep_time = self._step_period - elapsed
			if sleep_time > 0:
				time.sleep(sleep_time)
		self._last_step_time = time.time()

		action = self.hdf5["data"][f"demo_{self.current_demo_id}"]["action"][min(self.step_counter, self.cur_episode_length - 1)]
		self.action_repeat_counter += 1
		if self.action_repeat_counter == self.config.model.action_repeat:
			self.action_repeat_counter = 0
			self.step_counter += 1
			# Log progress every 50 steps
			if self.step_counter % 50 == 0:
				pct = 100 * self.step_counter / self.cur_episode_length
				print(f"  [Playback] Step {self.step_counter}/{self.cur_episode_length} ({pct:.0f}%)", flush=True)
		action = torch.from_numpy(action).float()
		return action.unsqueeze(0)

	def load(self, **kwargs):
		hdf_path = os.path.expanduser(self.config.model.hdf_path)
		if not os.path.isabs(hdf_path):
			hdf_path = os.path.join(self.base_path, hdf_path)
		self.hdf5 = h5py.File(hdf_path, "r")
		num_demos = len(self.hdf5['data'].keys())

		# Determine which demos to play
		if self.config.model.demo_ids is not None:
			self.demo_ids = list(self.config.model.demo_ids)
		else:
			self.demo_ids = list(range(num_demos))

		# Log length (number of timesteps) for each selected demo trajectory
		for demo_id in self.demo_ids:
			demo_key = f"demo_{demo_id}"
			if demo_key in self.hdf5["data"]:
				traj_len = self.hdf5["data"][demo_key]["action"].shape[0]
				logger.info(f"Trajectory length for {demo_key}: {traj_len} steps")

		# Repeat count
		self.repeat_count = getattr(self.config.model, 'repeat', 1) or 1

		# Check if we should use recorded timestamps for replay timing
		self._use_timestamps = getattr(self.config.model, 'use_timestamps', False)

		# Check if timestamps are available in the first demo (under time/ group)
		first_demo_key = f"demo_{self.demo_ids[0]}"
		has_timestamps = "timestamp_ms" in self.hdf5["data"][first_demo_key]["time"]

		if self._use_timestamps and has_timestamps:
			logger.info("Using recorded timestamps for replay timing")
		elif self._use_timestamps and not has_timestamps:
			logger.warning("use_timestamps=True but no timestamp_ms in data, falling back to playback_frequency")
			self._use_timestamps = False

		# Real-time rate limiting: playback_frequency in Hz (e.g., 30 for 30Hz)
		# Used as fallback if use_timestamps is False or timestamps unavailable
		playback_freq = getattr(self.config.model, 'playback_frequency', None)
		if playback_freq is not None and playback_freq > 0:
			self._step_period = 1.0 / playback_freq
			if not self._use_timestamps:
				logger.info(f"Rate limiting enabled: {playback_freq} Hz ({self._step_period*1000:.1f}ms per step)")
		else:
			self._step_period = None

		logger.info(f"Loaded {hdf_path}: playing demos {self.demo_ids} x{self.repeat_count}")

	def reset(self) -> None:
		self.step_counter = 0
		self._last_step_time = None  # Reset rate limiter on episode start
		self._episode_start_time = None  # Reset timestamp-based timing
		self._episode_end_printed = False  # Reset end message flag
		self.demo_index += 1
		if self.demo_index >= len(self.demo_ids):
			self.demo_index = 0
			self.current_repeat += 1
			if self.current_repeat >= self.repeat_count:
				self._all_demos_played = True
		self.cur_episode_length = self.hdf5["data"][f"demo_{self.current_demo_id}"]["action"].shape[0]

		# Load timestamps for current demo if using timestamp-based replay (from time/ group)
		if self._use_timestamps:
			demo_key = f"demo_{self.current_demo_id}"
			self._timestamps = self.hdf5["data"][demo_key]["time"]["timestamp_ms"][:]

		# Print replay header
		border = "=" * 70
		repeat_info = f" (repeat {self.current_repeat + 1}/{self.repeat_count})" if self.repeat_count > 1 else ""
		print(f"\n\033[1m\033[96m{border}\033[0m", flush=True)
		print(f"\033[1m\033[96m  ▶️  REPLAYING: demo_{self.current_demo_id}{repeat_info}\033[0m", flush=True)
		print(f"\033[1m\033[96m  📊 Trajectory length: {self.cur_episode_length} steps\033[0m", flush=True)
		print(f"\033[1m\033[96m{border}\033[0m\n", flush=True)

	def check_block_ended(self) -> bool:
		"""Returns True when all demos have been played for all repeats."""
		return self._all_demos_played

	def check_episode_ended(self, success: bool=True):
		ended = self.step_counter >= self.cur_episode_length
		if ended and not getattr(self, '_episode_end_printed', False):
			print(f"\n\033[1m\033[92m  ✅ Playback complete: demo_{self.current_demo_id} ({self.step_counter}/{self.cur_episode_length} steps)\033[0m\n", flush=True)
			self._episode_end_printed = True
		return ended

	def close(self) -> None:
		self.hdf5.close()