"""
Jaxified iterative Linear Quadrative Regulator (iLQR).

Please contact the author(s) of this library if you have any questions.
Author:  Haimin Hu (haiminh@princeton.edu)
"""

import time
import numpy as np
from typing import Tuple

from .dynamics import *
from .cost import *

from functools import partial
from jax import jit, lax
from jaxlib.xla_extension import ArrayImpl
import jax.numpy as jnp


class iLQR:

  def __init__(self, config, num_leaders: float = 0):
    self.horizon = config.N
    self.max_iter = config.MAX_ITER
    self.dim_x = config.DIM_X
    self.dim_u = config.DIM_U
    self.tol = 1e-2
    self.lambad_init = 10.0
    self.lambad_min = 1e-3
    self.alphas = 1.1**(-np.arange(10)**2)

    self.dynamics = globals()[config.DYNAMICS](config)
    self.cost = globals()[config.COST](config, num_leaders)

  def solve(
      self,
      cur_state: ArrayImpl,
      controls: ArrayImpl,
      leader_trajs: ArrayImpl,
      target: ArrayImpl,
  ) -> Tuple[np.ndarray, np.ndarray, float, float, int]:
    """
    Solves the iLQR-STP problem.

    Args:
        cur_state (ArrayImpl): (dim_x,)
        controls (ArrayImpl): (self.dim_u, N - 1)
        leader_trajs (ArrayImpl): (dim_x, N, N_leader) all leaders' optimized state trajs
        target (ArrayImpl): (dim_x,) target state

    Returns:
        states: np.ndarray
        controls: np.ndarray
        J: float
        t_process: float
        status: int
    """
    status = 0
    time0 = time.time()

    self.lambad = self.lambad_init

    # Initial forward pass.
    if controls is None:
      controls = jnp.zeros((self.dim_u, self.horizon))
    states = jnp.zeros((self.dim_x, self.horizon))
    states = states.at[:, 0].set(cur_state)

    for i in range(1, self.horizon):
      _states_i, _ = self.dynamics.integrate_forward(states[:, i - 1], controls[:, i - 1])
      states = states.at[:, i].set(_states_i)
    J = self.cost.get_cost(states, controls, leader_trajs, target)

    converged = False

    # Main loop.
    for i in range(self.max_iter):
      # Backward pass.
      Ks, ks = self.backward_pass(states, controls, leader_trajs, target)

      # Linesearch
      updated = False
      for alpha in self.alphas:
        X_new, U_new, J_new = self.forward_pass(
            states, controls, leader_trajs, Ks, ks, alpha, target
        )
        if J_new <= J:
          if jnp.abs((J-J_new) / J) < self.tol:
            converged = True
          J = J_new
          states = X_new
          controls = U_new
          updated = True
          break
      if updated:
        self.lambad *= 0.7
      else:
        status = 2
        break
      self.lambad = max(self.lambad_min, self.lambad)

      if converged:
        status = 1
        break
    t_process = time.time() - time0

    # print(t_process)

    return np.asarray(states), np.asarray(controls), J, t_process, status

  # ----------------------------- Jitted functions -----------------------------
  @partial(jit, static_argnames="self")
  def forward_pass(
      self,
      nominal_states: ArrayImpl,
      nominal_controls: ArrayImpl,
      leader_trajs: ArrayImpl,
      Ks: ArrayImpl,
      ks: ArrayImpl,
      alpha: float,
      target: ArrayImpl,
  ) -> Tuple[ArrayImpl, ArrayImpl, float]:
    """
    Jitted forward pass looped computation.

    Args:
        nominal_states (ArrayImpl): (dim_x, N)
        nominal_controls (ArrayImpl): (dim_u, N)
        leader_trajs (ArrayImpl): (dim_x, N, N_leader) all leaders' optimized state trajs
        Ks (ArrayImpl): gain matrices (dim_u, dim_x, N - 1)
        ks (ArrayImpl): gain vectors (dim_u, N - 1)
        alpha (float): scalar parameter
        target (ArrayImpl): (dim_x,) target state

    Returns:
        Xs (ArrayImpl): (dim_x, N)
        Us (ArrayImpl): (dim_u, N)
        J (float): total cost
    """

    @jit
    def forward_pass_looper(i, _carry):
      Xs, Us = _carry
      u = (
          nominal_controls[:, i] + alpha * ks[:, i]
          + Ks[:, :, i] @ (Xs[:, i] - nominal_states[:, i])
      )
      X_next, U_next = self.dynamics.integrate_forward(Xs[:, i], u)
      Xs = Xs.at[:, i + 1].set(X_next)
      Us = Us.at[:, i].set(U_next)
      return Xs, Us

    # Computes trajectories.
    Xs = jnp.zeros((self.dim_x, self.horizon))
    Us = jnp.zeros((self.dim_u, self.horizon))
    Xs = Xs.at[:, 0].set(nominal_states[:, 0])
    Xs, Us = lax.fori_loop(0, self.horizon - 1, forward_pass_looper, (Xs, Us))

    # Computes the total cost.
    J = self.cost.get_cost(Xs, Us, leader_trajs, target)

    return Xs, Us, J

  @partial(jit, static_argnames="self")
  def backward_pass(
      self,
      nominal_states: ArrayImpl,
      nominal_controls: ArrayImpl,
      leader_trajs: ArrayImpl,
      target: ArrayImpl,
  ) -> Tuple[ArrayImpl, ArrayImpl]:
    """
    Jitted backward pass looped computation.

    Args:
        nominal_states (ArrayImpl): (dim_x, N)
        nominal_controls (ArrayImpl): (dim_u, N)
        leader_trajs (ArrayImpl): (dim_x, N, N_leader) all leaders' optimized state trajs
        target (ArrayImpl): (dim_x,) target state

    Returns:
        Ks (ArrayImpl): gain matrices (dim_u, dim_x, N - 1)
        ks (ArrayImpl): gain vectors (dim_u, N - 1)
    """

    @jit
    def backward_pass_looper(i, _carry):
      V_x, V_xx, ks, Ks = _carry
      n = self.horizon - 2 - i

      Q_x = L_x[:, n] + fx[:, :, n].T @ V_x
      Q_u = L_u[:, n] + fu[:, :, n].T @ V_x
      Q_xx = L_xx[:, :, n] + fx[:, :, n].T @ V_xx @ fx[:, :, n]
      Q_ux = fu[:, :, n].T @ V_xx @ fx[:, :, n]
      Q_uu = L_uu[:, :, n] + fu[:, :, n].T @ V_xx @ fu[:, :, n]

      Q_uu_inv = jnp.linalg.inv(Q_uu + reg_mat)

      Ks = Ks.at[:, :, n].set(-Q_uu_inv @ Q_ux)
      ks = ks.at[:, n].set(-Q_uu_inv @ Q_u)

      V_x = Q_x - Ks[:, :, n].T @ Q_uu @ ks[:, n]
      V_xx = Q_xx - Ks[:, :, n].T @ Q_uu @ Ks[:, :, n]

      return V_x, V_xx, ks, Ks

    # Computes cost derivatives.
    L_x, L_xx, L_u, L_uu = self.cost.get_derivatives(
        nominal_states, nominal_controls, leader_trajs, target
    )

    # Computes dynamics Jacobians.
    fx, fu = self.dynamics.get_AB_matrix(nominal_states, nominal_controls)

    # Computes the control policy.
    Ks = jnp.zeros((self.dim_u, self.dim_x, self.horizon - 1))
    ks = jnp.zeros((self.dim_u, self.horizon - 1))
    V_x = L_x[:, -1]
    V_xx = L_xx[:, :, -1]
    reg_mat = self.lambad * jnp.eye(self.dim_u)

    V_x, V_xx, ks, Ks = lax.fori_loop(
        0, self.horizon - 1, backward_pass_looper, (V_x, V_xx, ks, Ks)
    )
    return Ks, ks

  @partial(jit, static_argnames="self")
  def compute_cost(
      self,
      nominal_states: ArrayImpl,
      nominal_controls: ArrayImpl,
      leader_trajs: ArrayImpl,
      target: ArrayImpl,
  ) -> float:
    """
    Computes accumulated cost along a trajectory.

    Args:
        nominal_states (ArrayImpl): (dim_x, N)
        nominal_controls (ArrayImpl): (dim_u, N)
        leader_trajs (ArrayImpl): (dim_x, N, N_leader) all leaders' optimized state trajs
        target (ArrayImpl): (dim_x,) target state

    Returns:
        J (float): total cost
    """
    return self.cost.get_cost(nominal_states, nominal_controls, leader_trajs, target)


class LQR_shielded:

  def __init__(self, config, num_leaders: float = 0):
    self.horizon = config.N
    self.dim_x = config.DIM_X
    self.dim_u = config.DIM_U
    self.config = config
    self.dynamics = globals()[config.DYNAMICS](config)
    self.cost = globals()[config.COST](config, num_leaders)
    self.dt = self.dynamics.dt

  def solve(
      self, cur_state: ArrayImpl, ctrl_ws: ArrayImpl, leader_trajs: ArrayImpl, target: ArrayImpl
  ) -> Tuple[np.ndarray, np.ndarray, float, float, int]:
    time0 = time.time()
    states, controls, J, status = self.solve_jax(cur_state, leader_trajs, target)
    t_process = time.time() - time0
    return np.asarray(states), np.asarray(controls), float(J), t_process, status

  @partial(jit, static_argnames="self")
  def solve_jax(self, cur_state: ArrayImpl, leader_trajs: ArrayImpl,
                target: ArrayImpl) -> Tuple[np.ndarray, np.ndarray, float, float, int]:
    """
    Solves the shielded LQR problem.

    Args:
        cur_state (ArrayImpl): (dim_x)
        leader_trajs (ArrayImpl): (dim_x, N, N_leader) all leaders' optimized state trajs
        target (ArrayImpl): (dim_x, N) target states

    Returns:
        states: np.ndarray
        controls: np.ndarray
        J: float
        t_process: float
        status: int
    """

    @jit
    def _looper(i, _carry):

      def true_fn(state, control):
        return jnp.array((-state[0] / self.dt, -state[1] / self.dt))  # Shielding action

      def false_fn(state, control):
        return control

      def _check_two_agents(state1, state2):
        _dx = self.config.PX_DIM
        _dy = self.config.PY_DIM
        pxpy_diff = jnp.stack((state1[_dx] - state2[_dx], state1[_dy] - state2[_dy]))
        sep = jnp.linalg.norm(pxpy_diff, axis=0)
        return sep < self.config.PROX_SEP_CHECK

      states, controls = _carry

      lqr_ctrl = -K_lqr @ (states[:, i] - target[:, i])
      _state_lqr, _ = self.dynamics.integrate_forward(states[:, i], lqr_ctrl)
      _check_two_agents_vmap = vmap(_check_two_agents, in_axes=(None, 1), out_axes=(0))
      pred = jnp.any(_check_two_agents_vmap(_state_lqr, leader_trajs[:, i, :]))
      shielded_ctrl = lax.cond(pred, true_fn, false_fn, states[:, i], lqr_ctrl)
      _state_nxt, _ctrl_nxt = self.dynamics.integrate_forward(states[:, i], shielded_ctrl)
      states = states.at[:, i + 1].set(_state_nxt)
      controls = controls.at[:, i].set(_ctrl_nxt)
      return states, controls

    status = 0
    K_lqr = self.cost.K_lqr
    states = jnp.zeros((self.dim_x, self.horizon))
    states = states.at[:, 0].set(cur_state)
    controls = jnp.zeros((self.dim_u, self.horizon))

    states, controls = lax.fori_loop(0, self.horizon, _looper, (states, controls))

    J = self.cost.get_cost(states, controls, leader_trajs, target)

    return states, controls, J, status
