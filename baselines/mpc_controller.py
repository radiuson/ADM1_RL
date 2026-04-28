#!/usr/bin/env python3
"""
MPC Controller for ADM1 Biogas
================================

Standard Model Predictive Control with the persistent-disturbance assumption:
future influent substrate concentrations are held constant at their current
measured value throughout the prediction horizon.  This is the realistic
closed-loop MPC variant — it does not exploit knowledge of future disturbances.

Contrast with ``nmpc_controller.NMPCController`` (labelled "NMPC (oracle)" in
the paper), which has perfect look-ahead over future substrate concentrations.

Paper label: "MPC"
"""

import numpy as np
from baselines.nmpc_controller import NMPCController


class MPCController(NMPCController):
    """
    MPC with persistent-disturbance assumption.

    Identical to ``NMPCController`` except that during each prediction rollout
    the influent lookup is frozen at the current real-time value.  This prevents
    the optimizer from using knowledge of future substrate concentrations or
    disturbance spikes that would be unavailable in practice.

    Args:
        env       : ADM1Env_v2 instance.
        horizon   : Prediction horizon in environment steps (default 4 = 1 hour).
        max_iter  : Maximum optimizer iterations per control step (default 10).
        n_restarts: Number of random restarts for the optimizer (default 1).
        verbose   : Print timing every ``verbose`` steps; 0 = silent.
    """

    def __init__(
        self,
        env,
        horizon: int = 4,
        max_iter: int = 10,
        n_restarts: int = 1,
        verbose: int = 0,
    ):
        super().__init__(
            env=env,
            horizon=horizon,
            max_iter=max_iter,
            n_restarts=n_restarts,
            verbose=verbose,
        )
        self.name = "MPC"

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute optimal action under the persistent-disturbance assumption.

        Temporarily patches ``env._build_influent_dict`` so that all rollout
        steps see the current (frozen) influent rather than future CSV values,
        then restores the original method unconditionally.

        Args:
            observation: Current environment observation vector.

        Returns:
            3-dim action array [q_ad, feed_mult, Q_HEX].
        """
        # Capture the influent at the current real timestep
        real_step = self.env.current_step
        frozen_base = self.env._build_influent_dict(real_step)

        # Freeze influent lookup during rollouts
        original_build = self.env._build_influent_dict
        self.env._build_influent_dict = lambda _step: frozen_base

        try:
            action = super().get_action(observation)
        finally:
            # Always restore the original method, even if an exception occurs
            self.env._build_influent_dict = original_build

        return action
