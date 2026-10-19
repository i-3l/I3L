"""
Generalized momentum observer for external torque estimation.

Estimates external joint torques from robot dynamics (mass matrix, Coriolis, gravity)
and measured state (position, velocity), without depending on the current control loop.

Theory:
    Generalized momentum: p = M(q) * v
    Observer integrates known terms and compares to measured momentum:
        sigma(k) = sigma(k-1) + dt * [tau_cmd(k-1) + C^T(k)*v(k) - g(k) + r(k-1)]
        p(k) = M(q(k)) * v(k)
        r(k) = K_O * (sigma(k) - p(k))

    r converges to tau_ext with time constant 1/K_O.
    No acceleration estimation needed.
"""

import numpy as np
import pinocchio as pin
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MomentumObserverConfig:
    """Configuration for the generalized momentum observer."""
    enable: bool = False
    gain: float = 30.0            # K_O: observer bandwidth (rad/s). Higher = faster, noisier.
    include_coriolis: bool = True  # Use full Coriolis matrix (vs simplified gravity-only)
    leak_rate: float = 0.01       # Anti-drift for integral (prevents unbounded growth)


class MomentumObserver:
    """
    Generalized momentum observer for estimating external joint torques.

    Shares the Pinocchio model from the gravity compensator but maintains
    its own data instance to avoid interfering with gravity compensation calls.

    For bilateral systems with a single-arm URDF, handles arm splitting
    and mirroring internally (same pattern as GravityCompensator).
    """

    def __init__(
        self,
        config: MomentumObserverConfig,
        pinocchio_model,
        dof: int,
        is_bilateral_single_arm: bool,
        dt: float = 0.01,
    ):
        """
        Initialize momentum observer.

        Args:
            config: Observer configuration
            pinocchio_model: Pinocchio model (shared read-only from gravity compensator)
            dof: Total degrees of freedom (e.g. 14 for bilateral)
            is_bilateral_single_arm: True if URDF is single-arm for bilateral system
            dt: Control loop period in seconds (default 0.01 = 100Hz)
        """
        self.config = config
        self.enabled = config.enable
        self.dof = dof
        self.dt = dt
        self._is_bilateral_single_arm = is_bilateral_single_arm

        if self.enabled:
            # Share model (read-only), create own data (mutated by Pinocchio calls)
            self.model = pinocchio_model
            self.data = pinocchio_model.createData()
            self._arm_dof = self.model.nv
        else:
            self.model = None
            self.data = None
            self._arm_dof = dof if not is_bilateral_single_arm else dof // 2

        self._gain = config.gain
        self._leak = config.leak_rate
        self._include_coriolis = config.include_coriolis

        # Observer state per arm
        if self._is_bilateral_single_arm:
            # Two arms, each with arm_dof joints
            self._sigma_right = np.zeros(self._arm_dof, dtype=np.float64)
            self._sigma_left = np.zeros(self._arm_dof, dtype=np.float64)
            self._r_right = np.zeros(self._arm_dof, dtype=np.float64)
            self._r_left = np.zeros(self._arm_dof, dtype=np.float64)
            self._prev_tau_cmd_right = np.zeros(self._arm_dof, dtype=np.float64)
            self._prev_tau_cmd_left = np.zeros(self._arm_dof, dtype=np.float64)
        else:
            self._sigma = np.zeros(dof, dtype=np.float64)
            self._r = np.zeros(dof, dtype=np.float64)
            self._prev_tau_cmd = np.zeros(dof, dtype=np.float64)

        self._first_update = True
        self._external_torque: Optional[np.ndarray] = None

    def update(
        self,
        positions_urdf: np.ndarray,
        velocities_urdf: np.ndarray,
        tau_cmd_motor: np.ndarray,
    ) -> np.ndarray:
        """
        Update observer with new measurements.

        Args:
            positions_urdf: Joint positions in URDF frame (no sign conversion), shape (dof,)
            velocities_urdf: Joint velocities in motor/URDF frame (no sign conversion), shape (dof,)
            tau_cmd_motor: Total commanded torque in motor frame (what was sent to motors), shape (dof,)

        Returns:
            Estimated external torque in Nm per joint, shape (dof,).
            Motor frame (same frame as tau_cmd_motor).
        """
        if not self.enabled:
            result = np.zeros(self.dof, dtype=np.float32)
            self._external_torque = result
            return result

        if self._is_bilateral_single_arm:
            arm_dof = self._arm_dof

            # Split into right arm (first half) and left arm (second half)
            q_right = positions_urdf[:arm_dof]
            q_left = positions_urdf[arm_dof:]
            v_right = velocities_urdf[:arm_dof]
            v_left = velocities_urdf[arm_dof:]
            tau_right = tau_cmd_motor[:arm_dof]
            tau_left = tau_cmd_motor[arm_dof:]

            # Right arm: mirror to left-arm URDF frame, compute, mirror back
            # (same convention as GravityCompensator)
            q_right_mirrored = -q_right
            v_right_mirrored = -v_right
            tau_right_mirrored = -tau_right

            r_right_mirrored = self._update_single_arm(
                q_right_mirrored,
                v_right_mirrored,
                tau_right_mirrored,
                arm='right',
            )
            # Mirror result back to right-arm frame
            tau_ext_right = -r_right_mirrored

            # Left arm: compute directly (matches URDF)
            tau_ext_left = self._update_single_arm(
                q_left,
                v_left,
                tau_left,
                arm='left',
            )

            result = np.concatenate([tau_ext_right, tau_ext_left]).astype(np.float32)
        else:
            result = self._update_single_arm(
                positions_urdf,
                velocities_urdf,
                tau_cmd_motor,
                arm='single',
            ).astype(np.float32)

        self._external_torque = result
        return result

    def _update_single_arm(
        self,
        q: np.ndarray,
        v: np.ndarray,
        tau_cmd: np.ndarray,
        arm: str,
    ) -> np.ndarray:
        """
        Run one observer step for a single arm.

        Args:
            q: Joint positions in URDF frame, shape (arm_dof,)
            v: Joint velocities in URDF frame, shape (arm_dof,)
            tau_cmd: Commanded torque, shape (arm_dof,)
            arm: 'right', 'left', or 'single' (selects state buffers)

        Returns:
            Observer output r (estimated external torque), shape (arm_dof,)
        """
        q = np.asarray(q, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        tau_cmd = np.asarray(tau_cmd, dtype=np.float64)

        # Select state buffers for this arm
        if arm == 'right':
            sigma = self._sigma_right
            r_prev = self._r_right
            prev_tau = self._prev_tau_cmd_right
        elif arm == 'left':
            sigma = self._sigma_left
            r_prev = self._r_left
            prev_tau = self._prev_tau_cmd_left
        else:
            sigma = self._sigma
            r_prev = self._r
            prev_tau = self._prev_tau_cmd

        # 1. Mass matrix: M(q)
        pin.crba(self.model, self.data, q)
        M = self.data.M.copy()
        # crba only fills upper triangle; symmetrize
        M = M + np.triu(M, 1).T

        # 2. Gravity torque: g(q)
        pin.computeGeneralizedGravity(self.model, self.data, q)
        g = self.data.g.copy()

        # 3. Coriolis matrix: C(q, v) (optional)
        if self._include_coriolis:
            pin.computeCoriolisMatrix(self.model, self.data, q, v)
            Ct_v = self.data.C.T @ v
        else:
            Ct_v = np.zeros_like(v)

        # 4. Measured generalized momentum
        p = M @ v

        # On first update, initialize sigma to match p so r starts at zero
        if self._first_update:
            sigma[:] = p
            prev_tau[:] = tau_cmd
            self._first_update = False

        # 5. Integrate: sigma += dt * (prev_tau_cmd + C^T*v - g + r_prev)
        #    With anti-drift leak: sigma -= dt * leak * sigma
        sigma += self.dt * (prev_tau + Ct_v - g + r_prev)
        sigma -= self.dt * self._leak * sigma

        # 6. Observer output: r = K_O * (sigma - p)
        r = self._gain * (sigma - p)

        # Store state back
        if arm == 'right':
            self._sigma_right = sigma
            self._r_right = r
            self._prev_tau_cmd_right = tau_cmd.copy()
        elif arm == 'left':
            self._sigma_left = sigma
            self._r_left = r
            self._prev_tau_cmd_left = tau_cmd.copy()
        else:
            self._sigma = sigma
            self._r = r
            self._prev_tau_cmd = tau_cmd.copy()

        return r

    def reset(self):
        """Zero all observer state. Called on mode change / episode reset."""
        if self._is_bilateral_single_arm:
            self._sigma_right.fill(0.0)
            self._sigma_left.fill(0.0)
            self._r_right.fill(0.0)
            self._r_left.fill(0.0)
            self._prev_tau_cmd_right.fill(0.0)
            self._prev_tau_cmd_left.fill(0.0)
        else:
            self._sigma.fill(0.0)
            self._r.fill(0.0)
            self._prev_tau_cmd.fill(0.0)

        self._first_update = True
        self._external_torque = None

    @property
    def external_torque(self) -> Optional[np.ndarray]:
        """Last computed external torque estimate (for reading from gate/logger)."""
        return self._external_torque
