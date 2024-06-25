"""
Costs, gradients, and Hessians.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
"""

import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod
from scipy.linalg import solve_discrete_are as dare

from functools import partial
from jax import jit, lax, vmap, jacfwd, hessian
from jaxlib.xla_extension import ArrayImpl
import jax.numpy as jnp

from .utils import Struct
from .dynamics import *


class Cost(ABC):

  def __init__(self, config: Struct, num_leaders: float):
    self.config = config

    # Planning parameters.
    self.N = config.N  # number of planning steps
    self.num_leaders = num_leaders

    # System parameters.
    self.dim_x = config.DIM_X
    self.dim_u = config.DIM_U
    self.px_dim = config.PX_DIM
    self.py_dim = config.PY_DIM

  @abstractmethod
  def get_cost(
      self, states: ArrayImpl, controls: ArrayImpl, leader_trajs: ArrayImpl, target: ArrayImpl
  ) -> ArrayImpl:
    """
    Calculates the cost given planned states and controls.

    Args:
        states (ArrayImpl): (dim_x, N) planned trajectory.
        controls (ArrayImpl): (dim_u, N) planned control sequence.
        leader_trajs (ArrayImpl): (dim_x, N, N_leader) all leaders' optimized state trajs.
        target (ArrayImpl): (dim_x,) target state.

    Returns:
    float: total cost.
    """
    raise NotImplementedError

  @abstractmethod
  def get_derivatives(
      self, states: ArrayImpl, controls: ArrayImpl, leader_trajs: ArrayImpl, target: ArrayImpl
  ) -> Tuple[ArrayImpl, ArrayImpl, ArrayImpl, ArrayImpl]:
    """
    Calculates gradients and Hessian of the overall cost using Jax.

    Args:
        states (ArrayImpl): (dim_x, N) planned trajectory.
        controls (ArrayImpl): (dim_u, N) planned control sequence.
        leader_trajs (ArrayImpl): (dim_x, N, N_leader) all leaders' optimized state trajs.
        target (ArrayImpl): (dim_x,) target state.

    Returns:
        ArrayImpl: lxs of the shape (dim_x, N).
        ArrayImpl: Hxxs of the shape (dim_x, dim_x, N).
        ArrayImpl: lus of the shape (dim_u, N).
        ArrayImpl: Huus of the shape (dim_u, dim_u, N).
    """
    raise NotImplementedError


class CostDubinsCar(Cost):

  def __init__(self, config: Struct, num_leaders: float):
    Cost.__init__(self, config, num_leaders)

    # Standard LQ weighting matrices.
    self.W_state = np.diag((config.W_X, config.W_Y, config.W_V, config.W_PSI))
    self.W_control = np.diag((config.W_ACCEL, config.W_DELTA))
    self.W_terminal = np.diag((config.W_X_T, config.W_Y_T, config.W_V_T, config.W_PSI_T))

    # Soft constraint parameters.
    self.q1_v = config.Q1_V
    self.q2_v = config.Q2_V
    self.q1_prox = config.Q1_PROX
    self.q2_prox = config.Q2_PROX
    self.barrier_thr = config.BARRIER_THR
    self.prox_sep = config.PROX_SEP
    self.v_min = config.V_MIN
    self.v_max = config.V_MAX

  @partial(jit, static_argnames="self")
  def get_cost(
      self, states: ArrayImpl, controls: ArrayImpl, leader_trajs: ArrayImpl, target: ArrayImpl
  ) -> ArrayImpl:
    """
    Calculates the cost given planned states and controls.

    Args:
        states (ArrayImpl): (dim_x, N) planned trajectory.
        controls (ArrayImpl): (dim_u, N) planned control sequence.
        leader_trajs (ArrayImpl): (dim_x, N, N_leader) all leaders' optimized state trajs.
        target (ArrayImpl): (dim_x,) target state.

    Returns:
    float: total cost.
    """
    # vmap all costs.
    c_state_vmap = vmap(self.state_cost_stage, in_axes=(1, None), out_axes=(0))
    c_cntrl_vmap = vmap(self.control_cost_stage, in_axes=(1), out_axes=(0))
    c_velbd_vmap = vmap(self.vel_bound_cost_stage, in_axes=(1), out_axes=(0))
    c_proxi_vmap = vmap(self.proximity_cost_stage, in_axes=(1, 1), out_axes=(0))

    # Evaluates all cost terms.
    c_state = c_state_vmap(states, target)
    c_cntrl = c_cntrl_vmap(controls)
    c_velbd = c_velbd_vmap(states)
    c_proxi = c_proxi_vmap(states, leader_trajs)
    c_termi = self.state_cost_terminal(states[:, -1], target)

    # Sums up all cost terms.
    J = jnp.sum(c_state + c_cntrl + c_velbd + c_proxi) + c_termi

    return J

  @partial(jit, static_argnames="self")
  def get_derivatives(
      self, states: ArrayImpl, controls: ArrayImpl, leader_trajs: ArrayImpl, target: ArrayImpl
  ) -> Tuple[ArrayImpl, ArrayImpl, ArrayImpl, ArrayImpl]:
    """
    Calculates gradients and Hessian of the overall cost using Jax.

    Args:
        states (ArrayImpl): (dim_x, N) planned trajectory.
        controls (ArrayImpl): (dim_u, N) planned control sequence.
        leader_trajs (ArrayImpl): (dim_x, N, N_leader) all leaders' optimized state trajs.
        target (ArrayImpl): (dim_x,) target state.

    Returns:
        ArrayImpl: lxs of the shape (dim_x, N).
        ArrayImpl: Hxxs of the shape (dim_x, dim_x, N).
        ArrayImpl: lus of the shape (dim_u, N).
        ArrayImpl: Huus of the shape (dim_u, dim_u, N).
    """
    # Creates cost gradient functions.
    lx_state_fn = jacfwd(self.state_cost_stage, argnums=0)
    lx_velbd_fn = jacfwd(self.vel_bound_cost_stage, argnums=0)
    lx_proxi_fn = jacfwd(self.proximity_cost_stage, argnums=0)
    lu_cntrl_fn = jacfwd(self.control_cost_stage, argnums=0)
    lx_termi_fn = jacfwd(self.state_cost_terminal, argnums=0)

    # Creates cost Hessian functions.
    Hxx_state_fn = hessian(self.state_cost_stage, argnums=0)
    Hxx_velbd_fn = hessian(self.vel_bound_cost_stage, argnums=0)
    Hxx_proxi_fn = hessian(self.proximity_cost_stage, argnums=0)
    Huu_cntrl_fn = hessian(self.control_cost_stage, argnums=0)
    Hxx_termi_fn = hessian(self.state_cost_terminal, argnums=0)

    # vmap all gradients and Hessians.
    lx_state_vmap = vmap(lx_state_fn, in_axes=(1, None), out_axes=(1))
    lx_velbd_vmap = vmap(lx_velbd_fn, in_axes=(1), out_axes=(1))
    lx_proxi_vmap = vmap(lx_proxi_fn, in_axes=(1, 1), out_axes=(1))
    lu_cntrl_vmap = vmap(lu_cntrl_fn, in_axes=(1), out_axes=(1))

    Hxx_state_vmap = vmap(Hxx_state_fn, in_axes=(1, None), out_axes=(2))
    Hxx_velbd_vmap = vmap(Hxx_velbd_fn, in_axes=(1), out_axes=(2))
    Hxx_proxi_vmap = vmap(Hxx_proxi_fn, in_axes=(1, 1), out_axes=(2))
    Huu_cntrl_vmap = vmap(Huu_cntrl_fn, in_axes=(1), out_axes=(2))

    # Evaluates all cost gradients and Hessians.
    lx_state = lx_state_vmap(states, target)
    lx_velbd = lx_velbd_vmap(states)
    lx_proxi = lx_proxi_vmap(states, leader_trajs)
    lu_cntrl = lu_cntrl_vmap(controls)
    lx_termi = lx_termi_fn(states[:, -1], target)

    Hxx_state = Hxx_state_vmap(states, target)
    Hxx_velbd = Hxx_velbd_vmap(states)
    Hxx_proxi = Hxx_proxi_vmap(states, leader_trajs)
    Huu_cntrl = Huu_cntrl_vmap(controls)
    Hxx_termi = Hxx_termi_fn(states[:, -1], target)

    lxs = lx_state + lx_velbd + lx_proxi
    lus = lu_cntrl
    Hxxs = Hxx_state + Hxx_velbd + Hxx_proxi
    Huus = Huu_cntrl

    lxs = lxs.at[:, -1].set(lxs[:, -1] + lx_termi)
    Hxxs = Hxxs.at[:, :, -1].set(Hxxs[:, :, -1] + Hxx_termi)

    return lxs, Hxxs, lus, Huus

  # --------------------------- Running performance cost terms ---------------------------
  @partial(jit, static_argnames="self")
  def state_cost_stage(self, state: ArrayImpl, target: ArrayImpl) -> ArrayImpl:
    """
    Computes the stage state cost.

    Args:
        state (ArrayImpl): (4,)
        target (ArrayImpl): (2,)

    Returns:
        ArrayImpl: cost (scalar)
    """
    return (state - target).T @ self.W_state @ (state-target)

  @partial(jit, static_argnames="self")
  def control_cost_stage(self, control: ArrayImpl) -> ArrayImpl:
    """
    Computes the stage control cost.

    Args:
        control (ArrayImpl): (2,)

    Returns:
        ArrayImpl: cost (scalar)
    """
    return control.T @ self.W_control @ control

  @partial(jit, static_argnames="self")
  def state_cost_terminal(self, state: ArrayImpl, target: ArrayImpl) -> ArrayImpl:
    """
    Computes the terminal state cost.
    HACK: terminal velocity is set to 0.

    Args:
        state (ArrayImpl): (4,)  [x, y, v, psi]
        target (ArrayImpl): (4,) [x, y, v, psi]

    Returns:
        ArrayImpl: cost (scalar)
    """
    t_state = jnp.array([target[0], target[1], 0., target[3]])
    return (state - t_state).T @ self.W_terminal @ (state-t_state)

  # ----------------------- Running soft constraint cost terms -----------------------
  @partial(jit, static_argnames="self")
  def vel_bound_cost_stage(self, state: ArrayImpl) -> ArrayImpl:
    """
    Calculates the velocity bound soft constraint cost.

    Args:
        state (ArrayImpl): (4,)

    Returns:
        ArrayImpl: cost (scalar)
    """
    cons_v_min = self.v_min - state[2]
    cons_v_max = state[2] - self.v_max
    barrier_v_min = self.q1_v * jnp.exp(jnp.clip(self.q2_v * cons_v_min, None, self.barrier_thr))
    barrier_v_max = self.q1_v * jnp.exp(jnp.clip(self.q2_v * cons_v_max, None, self.barrier_thr))
    return barrier_v_min + barrier_v_max

  @partial(jit, static_argnames="self")
  def proximity_cost_stage(self, state: ArrayImpl, leader_state: ArrayImpl) -> ArrayImpl:
    """
    Calculates the proximity soft constraint cost.

    Args:
        state (ArrayImpl): (4,) ego state
        leader_state (ArrayImpl): (4, N_leader) leaders' optimized state

    Returns:
        ArrayImpl: cost (scalar)
    """

    def _looper(ii, cost_vec):
      dx = state[self.px_dim] - leader_state[self.px_dim, ii]
      dy = state[self.py_dim] - leader_state[self.py_dim, ii]
      penalty_prox = -jnp.minimum(jnp.sqrt(dx**2 + dy**2) - self.prox_sep, 0.0)
      cost_vec = cost_vec.at[ii].set(
          self.q1_prox * jnp.exp(jnp.clip(self.q2_prox * penalty_prox, None, self.barrier_thr))
      )
      return cost_vec

    if self.num_leaders == 0:
      return 0.0
    else:
      cost_vec = jnp.zeros((self.num_leaders,))
      cost_vec = lax.fori_loop(0, self.num_leaders, _looper, leader_state)
      # return jnp.sum(cost_vec)
      return jnp.max(cost_vec)


class CostDoubleIntegrator(Cost):

  def __init__(self, config: Struct, num_leaders: float):
    Cost.__init__(self, config, num_leaders)

    # Standard LQ weighting matrices.
    self.W_state = np.diag((config.W_X, config.W_Y, config.W_V))
    self.W_control = np.diag((config.W_AX, config.W_AY))

    # Soft constraint parameters.
    self.q1_v = config.Q1_V
    self.q2_v = config.Q2_V
    self.q1_prox = config.Q1_PROX
    self.q2_prox = config.Q2_PROX
    self.barrier_thr = config.BARRIER_THR
    self.prox_sep = config.PROX_SEP
    self.v_min = config.V_MIN
    self.v_max = config.V_MAX
    _DI_sys = DoubleIntegrator(config)
    self.compute_terminal_cost(np.asarray(_DI_sys.Ad), np.asarray(_DI_sys.Bd))

  def compute_terminal_cost(self, Ad: np.array, Bd: np.array):
    """
    Computes the terminal cost-to-go via solving DARE.

    Args:
        Ad (np.array): system matrix
        Bd (np.array): system matrix
    """
    _Q = np.diag((self.config.W_X_T, self.config.W_Y_T, self.config.W_V_T, self.config.W_V_T))
    _Q_terminal = dare(a=Ad, b=Bd, q=_Q, r=self.W_control)
    self.Q_terminal = jnp.asarray(_Q_terminal)
    _K_lqr = np.linalg.inv(Bd.T @ _Q_terminal @ Bd + self.W_control) @ (Bd.T @ _Q_terminal @ Ad)
    self.K_lqr = jnp.asarray(_K_lqr)

  @partial(jit, static_argnames="self")
  def get_cost(
      self, states: ArrayImpl, controls: ArrayImpl, leader_trajs: ArrayImpl, target: ArrayImpl
  ) -> ArrayImpl:
    """
    Calculates the cost given planned states and controls.

    Args:
        states (ArrayImpl): (dim_x, N) planned trajectory.
        controls (ArrayImpl): (dim_u, N) planned control sequence.
        leader_trajs (ArrayImpl): (dim_x, N, N_leader) all leaders' optimized state trajs.
        target (ArrayImpl): (dim_x,) target state.

    Returns:
    float: total cost.
    """
    # vmap all costs.
    c_state_vmap = vmap(self.state_cost_stage, in_axes=(1, None), out_axes=(0))
    c_cntrl_vmap = vmap(self.control_cost_stage, in_axes=(1), out_axes=(0))
    c_velbd_vmap = vmap(self.vel_bound_cost_stage, in_axes=(1), out_axes=(0))
    c_proxi_vmap = vmap(self.proximity_cost_stage, in_axes=(1, 1), out_axes=(0))

    # Evaluates all cost terms.
    c_state = c_state_vmap(states, target)
    c_cntrl = c_cntrl_vmap(controls)
    c_velbd = c_velbd_vmap(states)
    c_proxi = c_proxi_vmap(states, leader_trajs)
    c_termi = self.state_cost_terminal(states[:, -1], target)

    # Sums up all cost terms.
    J = jnp.sum(c_state + c_cntrl + c_velbd + c_proxi) + c_termi

    return J

  @partial(jit, static_argnames="self")
  def get_derivatives(
      self, states: ArrayImpl, controls: ArrayImpl, leader_trajs: ArrayImpl, target: ArrayImpl
  ) -> Tuple[ArrayImpl, ArrayImpl, ArrayImpl, ArrayImpl]:
    """
    Calculates gradients and Hessian of the overall cost using Jax.

    Args:
        states (ArrayImpl): (dim_x, N) planned trajectory.
        controls (ArrayImpl): (dim_u, N) planned control sequence.
        leader_trajs (ArrayImpl): (dim_x, N, N_leader) all leaders' optimized state trajs.
        target (ArrayImpl): (dim_x,) target state.

    Returns:
        ArrayImpl: lxs of the shape (dim_x, N).
        ArrayImpl: Hxxs of the shape (dim_x, dim_x, N).
        ArrayImpl: lus of the shape (dim_u, N).
        ArrayImpl: Huus of the shape (dim_u, dim_u, N).
    """
    # Creates cost gradient functions.
    lx_state_fn = jacfwd(self.state_cost_stage, argnums=0)
    lx_velbd_fn = jacfwd(self.vel_bound_cost_stage, argnums=0)
    lx_proxi_fn = jacfwd(self.proximity_cost_stage, argnums=0)
    lu_cntrl_fn = jacfwd(self.control_cost_stage, argnums=0)
    lx_termi_fn = jacfwd(self.state_cost_terminal, argnums=0)

    # Creates cost Hessian functions.
    Hxx_state_fn = hessian(self.state_cost_stage, argnums=0)
    Hxx_velbd_fn = hessian(self.vel_bound_cost_stage, argnums=0)
    Hxx_proxi_fn = hessian(self.proximity_cost_stage, argnums=0)
    Huu_cntrl_fn = hessian(self.control_cost_stage, argnums=0)
    Hxx_termi_fn = hessian(self.state_cost_terminal, argnums=0)

    # vmap all gradients and Hessians.
    lx_state_vmap = vmap(lx_state_fn, in_axes=(1, None), out_axes=(1))
    lx_velbd_vmap = vmap(lx_velbd_fn, in_axes=(1), out_axes=(1))
    lx_proxi_vmap = vmap(lx_proxi_fn, in_axes=(1, 1), out_axes=(1))
    lu_cntrl_vmap = vmap(lu_cntrl_fn, in_axes=(1), out_axes=(1))

    Hxx_state_vmap = vmap(Hxx_state_fn, in_axes=(1, None), out_axes=(2))
    Hxx_velbd_vmap = vmap(Hxx_velbd_fn, in_axes=(1), out_axes=(2))
    Hxx_proxi_vmap = vmap(Hxx_proxi_fn, in_axes=(1, 1), out_axes=(2))
    Huu_cntrl_vmap = vmap(Huu_cntrl_fn, in_axes=(1), out_axes=(2))

    # Evaluates all cost gradients and Hessians.
    lx_state = lx_state_vmap(states, target)
    lx_velbd = lx_velbd_vmap(states)
    lx_proxi = lx_proxi_vmap(states, leader_trajs)
    lu_cntrl = lu_cntrl_vmap(controls)
    lx_termi = lx_termi_fn(states[:, -1], target)

    Hxx_state = Hxx_state_vmap(states, target)
    Hxx_velbd = Hxx_velbd_vmap(states)
    Hxx_proxi = Hxx_proxi_vmap(states, leader_trajs)
    Huu_cntrl = Huu_cntrl_vmap(controls)
    Hxx_termi = Hxx_termi_fn(states[:, -1], target)

    lxs = lx_state + lx_velbd + lx_proxi
    lus = lu_cntrl
    Hxxs = Hxx_state + Hxx_velbd + Hxx_proxi
    Huus = Huu_cntrl

    lxs = lxs.at[:, -1].set(lxs[:, -1] + lx_termi)
    Hxxs = Hxxs.at[:, :, -1].set(Hxxs[:, :, -1] + Hxx_termi)

    return lxs, Hxxs, lus, Huus

  # --------------------------- Running performance cost terms ---------------------------
  @partial(jit, static_argnames="self")
  def state_cost_stage(self, state: ArrayImpl, target: ArrayImpl) -> ArrayImpl:
    """
    Computes the stage state cost.

    Args:
        state (ArrayImpl): (4,)  [x, y, vx, vy]
        target (ArrayImpl): (3,) [x, y, v]

    Returns:
        ArrayImpl: cost (scalar)
    """
    _xyv = jnp.array((state[self.px_dim], state[self.py_dim], jnp.linalg.norm(state[2:])))
    return (_xyv - target).T @ self.W_state @ (_xyv-target)

  @partial(jit, static_argnames="self")
  def control_cost_stage(self, control: ArrayImpl) -> ArrayImpl:
    """
    Computes the stage control cost.

    Args:
        control (ArrayImpl): (dim_u,)

    Returns:
        ArrayImpl: cost (scalar)
    """
    return control.T @ self.W_control @ control

  @partial(jit, static_argnames="self")
  def state_cost_terminal(self, state: ArrayImpl, target: ArrayImpl) -> ArrayImpl:
    """
    Computes the terminal state cost.
    HACK: terminal velocity is set to 0.

    Args:
        state (ArrayImpl): (4,)  [x, y, vx, vy]
        target (ArrayImpl): (3,) [x, y, v]

    Returns:
        ArrayImpl: cost (scalar)
    """
    t_state = jnp.array([target[0], target[1], 0., 0.])
    return (state - t_state).T @ self.Q_terminal @ (state-t_state)

  # ----------------------- Running soft constraint cost terms -----------------------
  @partial(jit, static_argnames="self")
  def vel_bound_cost_stage(self, state: ArrayImpl) -> ArrayImpl:
    """
    Calculates the velocity bound soft constraint cost.

    Args:
        state (ArrayImpl): (4,)

    Returns:
        ArrayImpl: cost (scalar)
    """
    v = jnp.linalg.norm(state[2:])
    # cons_v_min = self.v_min - v
    cons_v_max = v - self.v_max
    # barrier_v_min = self.q1_v * jnp.exp(jnp.clip(self.q2_v * cons_v_min, None, self.barrier_thr))
    barrier_v_max = self.q1_v * jnp.exp(jnp.clip(self.q2_v * cons_v_max, None, self.barrier_thr))
    return barrier_v_max

  @partial(jit, static_argnames="self")
  def proximity_cost_stage(self, state: ArrayImpl, leader_state: ArrayImpl) -> ArrayImpl:
    """
    Calculates the proximity soft constraint cost.

    Args:
        state (ArrayImpl): (dim_x,) ego state
        leader_state (ArrayImpl): (dim_x, N_leader) leaders' optimized state

    Returns:
        ArrayImpl: cost (scalar)
    """

    def _looper(ii, cost_vec):
      dx = state[self.px_dim] - leader_state[self.px_dim, ii]
      dy = state[self.py_dim] - leader_state[self.py_dim, ii]
      penalty_prox = -jnp.minimum(jnp.sqrt(dx**2 + dy**2) - self.prox_sep, 0.0)
      cost_vec = cost_vec.at[ii].set(
          self.q1_prox * jnp.exp(jnp.clip(self.q2_prox * penalty_prox, None, self.barrier_thr))
      )
      return cost_vec

    if self.num_leaders == 0:
      return 0.0
    else:
      cost_vec = jnp.zeros((self.num_leaders,))
      cost_vec = lax.fori_loop(0, self.num_leaders, _looper, leader_state)
      # return jnp.sum(cost_vec)
      return jnp.max(cost_vec)


class CostDoubleIntegratorMovingTargets(CostDoubleIntegrator):

  def __init__(self, config: Struct, num_leaders: float):
    CostDoubleIntegrator.__init__(self, config, num_leaders)

    self.W_state = np.diag((config.W_X, config.W_Y, config.W_V, config.W_V))

  @partial(jit, static_argnames="self")
  def get_cost(
      self, states: ArrayImpl, controls: ArrayImpl, leader_trajs: ArrayImpl, target: ArrayImpl
  ) -> ArrayImpl:
    """
    Calculates the cost given planned states and controls.

    Args:
        states (ArrayImpl): (dim_x, N) planned trajectory.
        controls (ArrayImpl): (dim_u, N) planned control sequence.
        leader_trajs (ArrayImpl): (dim_x, N, N_leader) all leaders' optimized state trajs.
        target (ArrayImpl): (dim_x, N) target trajectory.

    Returns:
    float: total cost.
    """
    # vmap all costs.
    c_state_vmap = vmap(self.state_cost_stage, in_axes=(1, 1), out_axes=(0))
    c_cntrl_vmap = vmap(self.control_cost_stage, in_axes=(1), out_axes=(0))
    c_velbd_vmap = vmap(self.vel_bound_cost_stage, in_axes=(1), out_axes=(0))
    c_proxi_vmap = vmap(self.proximity_cost_stage, in_axes=(1, 1), out_axes=(0))

    # Evaluates all cost terms.
    c_state = c_state_vmap(states, target)
    c_cntrl = c_cntrl_vmap(controls)
    c_velbd = c_velbd_vmap(states)
    c_proxi = c_proxi_vmap(states, leader_trajs)
    c_termi = self.state_cost_terminal(states[:, -1], target[:, -1])

    # Sums up all cost terms.
    J = jnp.sum(c_state + c_cntrl + c_velbd + c_proxi) + c_termi

    return J

  @partial(jit, static_argnames="self")
  def get_derivatives(
      self, states: ArrayImpl, controls: ArrayImpl, leader_trajs: ArrayImpl, target: ArrayImpl
  ) -> Tuple[ArrayImpl, ArrayImpl, ArrayImpl, ArrayImpl]:
    """
    Calculates gradients and Hessian of the overall cost using Jax.

    Args:
        states (ArrayImpl): (dim_x, N) planned trajectory.
        controls (ArrayImpl): (dim_u, N) planned control sequence.
        leader_trajs (ArrayImpl): (dim_x, N, N_leader) all leaders' optimized state trajs.
        target (ArrayImpl): (dim_x,) target state.

    Returns:
        ArrayImpl: lxs of the shape (dim_x, N).
        ArrayImpl: Hxxs of the shape (dim_x, dim_x, N).
        ArrayImpl: lus of the shape (dim_u, N).
        ArrayImpl: Huus of the shape (dim_u, dim_u, N).
    """
    # Creates cost gradient functions.
    lx_state_fn = jacfwd(self.state_cost_stage, argnums=0)
    lx_velbd_fn = jacfwd(self.vel_bound_cost_stage, argnums=0)
    lx_proxi_fn = jacfwd(self.proximity_cost_stage, argnums=0)
    lu_cntrl_fn = jacfwd(self.control_cost_stage, argnums=0)
    lx_termi_fn = jacfwd(self.state_cost_terminal, argnums=0)

    # Creates cost Hessian functions.
    Hxx_state_fn = hessian(self.state_cost_stage, argnums=0)
    Hxx_velbd_fn = hessian(self.vel_bound_cost_stage, argnums=0)
    Hxx_proxi_fn = hessian(self.proximity_cost_stage, argnums=0)
    Huu_cntrl_fn = hessian(self.control_cost_stage, argnums=0)
    Hxx_termi_fn = hessian(self.state_cost_terminal, argnums=0)

    # vmap all gradients and Hessians.
    lx_state_vmap = vmap(lx_state_fn, in_axes=(1, 1), out_axes=(1))
    lx_velbd_vmap = vmap(lx_velbd_fn, in_axes=(1), out_axes=(1))
    lx_proxi_vmap = vmap(lx_proxi_fn, in_axes=(1, 1), out_axes=(1))
    lu_cntrl_vmap = vmap(lu_cntrl_fn, in_axes=(1), out_axes=(1))

    Hxx_state_vmap = vmap(Hxx_state_fn, in_axes=(1, 1), out_axes=(2))
    Hxx_velbd_vmap = vmap(Hxx_velbd_fn, in_axes=(1), out_axes=(2))
    Hxx_proxi_vmap = vmap(Hxx_proxi_fn, in_axes=(1, 1), out_axes=(2))
    Huu_cntrl_vmap = vmap(Huu_cntrl_fn, in_axes=(1), out_axes=(2))

    # Evaluates all cost gradients and Hessians.
    lx_state = lx_state_vmap(states, target)
    lx_velbd = lx_velbd_vmap(states)
    lx_proxi = lx_proxi_vmap(states, leader_trajs)
    lu_cntrl = lu_cntrl_vmap(controls)
    lx_termi = lx_termi_fn(states[:, -1], target[:, -1])

    Hxx_state = Hxx_state_vmap(states, target)
    Hxx_velbd = Hxx_velbd_vmap(states)
    Hxx_proxi = Hxx_proxi_vmap(states, leader_trajs)
    Huu_cntrl = Huu_cntrl_vmap(controls)
    Hxx_termi = Hxx_termi_fn(states[:, -1], target[:, -1])

    lxs = lx_state + lx_velbd + lx_proxi
    lus = lu_cntrl
    Hxxs = Hxx_state + Hxx_velbd + Hxx_proxi
    Huus = Huu_cntrl

    lxs = lxs.at[:, -1].set(lxs[:, -1] + lx_termi)
    Hxxs = Hxxs.at[:, :, -1].set(Hxxs[:, :, -1] + Hxx_termi)

    return lxs, Hxxs, lus, Huus

  @partial(jit, static_argnames="self")
  def state_cost_stage(self, state: ArrayImpl, target: ArrayImpl) -> ArrayImpl:
    """
    Computes the stage state cost.

    Args:
        state (ArrayImpl): (4,)  [x, y, vx, vy]
        target (ArrayImpl): (4,) [x, y, vx, vy]

    Returns:
        ArrayImpl: cost (scalar)
    """
    return (state - target).T @ self.W_state @ (state-target)

  @partial(jit, static_argnames="self")
  def state_cost_terminal(self, state: ArrayImpl, target: ArrayImpl) -> ArrayImpl:
    """
    Computes the terminal state cost.

    Args:
        state (ArrayImpl): (4,)  [x, y, vx, vy]
        target (ArrayImpl): (4,) [x, y, vx, vy]

    Returns:
        ArrayImpl: cost (scalar)
    """
    return (state - target).T @ self.Q_terminal @ (state-target)


class CostDubinsCarILQGame(Cost):

  def __init__(self, config: Struct, LMx: np.ndarray, num_players: int):
    Cost.__init__(self, config, num_leaders=0)

    # Standard LQ weighting matrices.
    self.W_state = np.diag((config.W_X, config.W_Y, config.W_V, config.W_PSI))
    self.W_control = np.diag((config.W_ACCEL, config.W_DELTA))
    self.W_terminal = np.diag((config.W_X_T, config.W_Y_T, config.W_V_T, config.W_PSI_T))

    # Soft constraint parameters.
    self.q1_v = config.Q1_V
    self.q2_v = config.Q2_V
    self.q1_prox = config.Q1_PROX
    self.q2_prox = config.Q2_PROX
    self.barrier_thr = config.BARRIER_THR
    self.prox_sep = config.PROX_SEP
    self.v_min = config.V_MIN
    self.v_max = config.V_MAX

    # Lifting matrix
    self.LMx = LMx

    # Problem dimensions.
    self.dim_xi = LMx.shape[0]
    self.dim_x = LMx.shape[1]
    self.num_players = num_players

  @partial(jit, static_argnames="self")
  def get_cost(self, states: ArrayImpl, controls: ArrayImpl, target: ArrayImpl) -> ArrayImpl:
    """
    Calculates the cost given planned states and controls.

    Args:
        states (ArrayImpl): (dim_x, N) planned trajectory.
        controls (ArrayImpl): (dim_u, N) planned control sequence.
        target (ArrayImpl): (dim_xi,) target state.

    Returns:
    float: total cost.
    """
    # vmap all costs.
    c_state_vmap = vmap(self.state_cost_stage, in_axes=(1, None), out_axes=(0))
    c_cntrl_vmap = vmap(self.control_cost_stage, in_axes=(1), out_axes=(0))
    c_velbd_vmap = vmap(self.vel_bound_cost_stage, in_axes=(1), out_axes=(0))
    c_proxi_vmap = vmap(self.proximity_cost_stage, in_axes=(1), out_axes=(0))

    # Evaluates all cost terms.
    c_state = c_state_vmap(states, target)
    c_cntrl = c_cntrl_vmap(controls)
    c_velbd = c_velbd_vmap(states)
    c_proxi = c_proxi_vmap(states)
    c_termi = self.state_cost_terminal(states[:, -1], target)

    # Sums up all cost terms.
    J = jnp.sum(c_state + c_cntrl + c_velbd + c_proxi) + c_termi

    return J

  @partial(jit, static_argnames="self")
  def get_derivatives(self, states: ArrayImpl, controls: ArrayImpl,
                      target: ArrayImpl) -> Tuple[ArrayImpl, ArrayImpl, ArrayImpl, ArrayImpl]:
    """
    Calculates gradients and Hessian of the overall cost using Jax.

    Args:
        states (ArrayImpl): (dim_x, N) planned trajectory.
        controls (ArrayImpl): (dim_u, N) planned control sequence.
        target (ArrayImpl): (dim_x,) target state.

    Returns:
        ArrayImpl: lxs of the shape (dim_x, N).
        ArrayImpl: Hxxs of the shape (dim_x, dim_x, N).
        ArrayImpl: lus of the shape (dim_u, N).
        ArrayImpl: Huus of the shape (dim_u, dim_u, N).
    """
    # Creates cost gradient functions.
    lx_state_fn = jacfwd(self.state_cost_stage, argnums=0)
    lx_velbd_fn = jacfwd(self.vel_bound_cost_stage, argnums=0)
    lx_proxi_fn = jacfwd(self.proximity_cost_stage, argnums=0)
    lu_cntrl_fn = jacfwd(self.control_cost_stage, argnums=0)
    lx_termi_fn = jacfwd(self.state_cost_terminal, argnums=0)

    # Creates cost Hessian functions.
    Hxx_state_fn = hessian(self.state_cost_stage, argnums=0)
    Hxx_velbd_fn = hessian(self.vel_bound_cost_stage, argnums=0)
    Hxx_proxi_fn = hessian(self.proximity_cost_stage, argnums=0)
    Huu_cntrl_fn = hessian(self.control_cost_stage, argnums=0)
    Hxx_termi_fn = hessian(self.state_cost_terminal, argnums=0)

    # vmap all gradients and Hessians.
    lx_state_vmap = vmap(lx_state_fn, in_axes=(1, None), out_axes=(1))
    lx_velbd_vmap = vmap(lx_velbd_fn, in_axes=(1), out_axes=(1))
    lx_proxi_vmap = vmap(lx_proxi_fn, in_axes=(1), out_axes=(1))
    lu_cntrl_vmap = vmap(lu_cntrl_fn, in_axes=(1), out_axes=(1))

    Hxx_state_vmap = vmap(Hxx_state_fn, in_axes=(1, None), out_axes=(2))
    Hxx_velbd_vmap = vmap(Hxx_velbd_fn, in_axes=(1), out_axes=(2))
    Hxx_proxi_vmap = vmap(Hxx_proxi_fn, in_axes=(1), out_axes=(2))
    Huu_cntrl_vmap = vmap(Huu_cntrl_fn, in_axes=(1), out_axes=(2))

    # Evaluates all cost gradients and Hessians.
    lx_state = lx_state_vmap(states, target)
    lx_velbd = lx_velbd_vmap(states)
    lx_proxi = lx_proxi_vmap(states)
    lu_cntrl = lu_cntrl_vmap(controls)
    lx_termi = lx_termi_fn(states[:, -1], target)

    Hxx_state = Hxx_state_vmap(states, target)
    Hxx_velbd = Hxx_velbd_vmap(states)
    Hxx_proxi = Hxx_proxi_vmap(states)
    Huu_cntrl = Huu_cntrl_vmap(controls)
    Hxx_termi = Hxx_termi_fn(states[:, -1], target)

    lxs = lx_state + lx_velbd + lx_proxi
    lus = lu_cntrl
    Hxxs = Hxx_state + Hxx_velbd + Hxx_proxi
    Huus = Huu_cntrl

    lxs = lxs.at[:, -1].set(lxs[:, -1] + lx_termi)
    Hxxs = Hxxs.at[:, :, -1].set(Hxxs[:, :, -1] + Hxx_termi)

    return lxs, lus, Hxxs, Huus

  # --------------------------- Running performance cost terms ---------------------------
  @partial(jit, static_argnames="self")
  def state_cost_stage(self, state: ArrayImpl, target: ArrayImpl) -> ArrayImpl:
    """
    Computes the stage state cost.

    Args:
        state (ArrayImpl): (4 * num_players,) [x, y, v, psi]
        target (ArrayImpl): (4,)

    Returns:
        ArrayImpl: cost (scalar)
    """
    return (self.LMx @ state - target).T @ self.W_state @ (self.LMx @ state - target)

  @partial(jit, static_argnames="self")
  def control_cost_stage(self, control: ArrayImpl) -> ArrayImpl:
    """
    Computes the stage control cost.

    Args:
        control (ArrayImpl): (2,)

    Returns:
        ArrayImpl: cost (scalar)
    """
    return control.T @ self.W_control @ control

  @partial(jit, static_argnames="self")
  def state_cost_terminal(self, state: ArrayImpl, target: ArrayImpl) -> ArrayImpl:
    """
    Computes the terminal state cost.
    HACK: terminal velocity is set to 0.

    Args:
        state (ArrayImpl): (4 * num_players,)
        target (ArrayImpl): (4,)

    Returns:
        ArrayImpl: cost (scalar)
    """
    t_state = jnp.array([target[0], target[1], 0., target[3]])
    return (self.LMx @ state - t_state).T @ self.W_terminal @ (self.LMx @ state - t_state)

  # ----------------------- Running soft constraint cost terms -----------------------
  @partial(jit, static_argnames="self")
  def vel_bound_cost_stage(self, state: ArrayImpl) -> ArrayImpl:
    """
    Calculates the velocity bound soft constraint cost.

    Args:
        state (ArrayImpl): (4 * num_players,)

    Returns:
        ArrayImpl: cost (scalar)
    """
    state = self.LMx @ state
    cons_v_min = self.v_min - state[2]
    cons_v_max = state[2] - self.v_max
    barrier_v_min = self.q1_v * jnp.exp(jnp.clip(self.q2_v * cons_v_min, None, self.barrier_thr))
    barrier_v_max = self.q1_v * jnp.exp(jnp.clip(self.q2_v * cons_v_max, None, self.barrier_thr))
    return barrier_v_min + barrier_v_max

  @partial(jit, static_argnames="self")
  def proximity_cost_stage(self, state: ArrayImpl) -> ArrayImpl:
    """
    Calculates the proximity soft constraint cost.

    Args:
        state (ArrayImpl): (4 * num_players,) ego state

    Returns:
        ArrayImpl: cost (scalar)
    """

    def _looper(ii, cost_vec):

      def true_fn(penalty_prox):
        res = self.q1_prox * jnp.exp(jnp.clip(self.q2_prox * penalty_prox, None, self.barrier_thr))
        return res

      def false_fn(penalty_prox):
        return 0.

      dx = state_ego[self.px_dim] - state[self.px_dim, ii]
      dy = state_ego[self.py_dim] - state[self.py_dim, ii]
      sep_sq = dx**2 + dy**2
      penalty_prox = -jnp.minimum(jnp.sqrt(sep_sq) - self.prox_sep, 0.0)
      pred = sep_sq > 1e-3
      cost_vec = cost_vec.at[ii].set(lax.cond(pred, true_fn, false_fn, penalty_prox))
      return cost_vec

    state_ego = self.LMx @ state
    state = state.reshape(self.dim_xi, self.num_players)
    cost_vec = jnp.zeros((self.num_players,))
    cost_vec = lax.fori_loop(0, self.num_players, _looper, state)
    # return jnp.sum(cost_vec)
    return jnp.max(cost_vec)
