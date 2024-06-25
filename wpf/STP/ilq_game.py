"""
Jaxified differentiable iterative LQ Game solver that computes a locally approximate Nash
equilibrium solution to a general-sum trajectory game.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
"""

import time
import numpy as np
from typing import List, Tuple
from functools import partial

from jax import jit, lax, vmap
from jaxlib.xla_extension import ArrayImpl
import jax.numpy as jnp

from .utils import Struct
from .cost import CostDubinsCarILQGame
from .multiplayer_dynamical_system import MultiPlayerDynamicalSystem


class ILQGame(object):

  def __init__(self, config: Struct, dynamics: MultiPlayerDynamicalSystem, verbose: str = False):
    """
    Initializer.
    """

    self.config = config
    self.horizon = config.N
    self.max_iter = config.MAX_ITER
    self.dynamics = dynamics
    self.num_players = dynamics._num_players
    # self.line_search_scaling = np.linspace(config.LS_MAX, config.LS_MIN, config.LS_NUM)
    self.line_search_scaling = 1.1**(-np.arange(10)**2)

    self.dim_x = self.dynamics._x_dim
    self.dim_x_ss = self.dynamics._subsystem.dim_x
    self.dim_u_ss = self.dynamics._subsystem.dim_u

    # Create costs for each player.
    self.costs = [
        CostDubinsCarILQGame(config, self.dynamics._LMx[:, :, ii], self.num_players)
        for ii in range(self.num_players)
    ]

    self.verbose = verbose
    self.reset()

  def solve(
      self, cur_state: List[np.ndarray], us_warmstart: List[np.ndarray], targets: List[np.ndarray]
  ):
    """
    Runs the iLQGame algorithm.

    Args:
        cur_state (List[np.ndarray]): [(nx,)] Current state.
        us_warmstart (List[np.ndarray]): [(nui, N)] Warmstart controls.

    Returns:
        states: List[np.ndarray]
        controls: List[np.ndarray]
        t_process: float
        status: int
    """
    status = 0
    time0 = time.time()
    cur_state = np.concatenate(cur_state)
    us_warmstart = np.stack(us_warmstart, axis=2)
    targets = np.stack(targets, axis=1)

    # Initial forward pass.
    x0, us = jnp.asarray(cur_state), jnp.asarray(us_warmstart)
    xs, cost_init = self.initial_forward_pass(x0, us, targets)
    cost_best = cost_init

    self.reset(_current_x=xs, _current_u=us, _current_J=cost_init)

    # Main loop.
    for iter in range(self.max_iter):
      t_start = time.time()

      # region: Forward & backward passes.
      As, Bs = self.linearize_dynamics(xs, us)
      lxs, lus, Hxxs, Huus = self.quadraticize_costs(xs, us, targets)
      Ps, alphas_bpass, _, _ = self.backward_pass(As, Bs, lxs, lus, Hxxs, Huus)
      for line_search_scaling in self.line_search_scaling:
        alphas_ls = alphas_bpass * line_search_scaling
        xs_ls, us_ls, cost_ls = self.compute_operating_point(
            xs, us, Ps, alphas_ls, cur_state, targets
        )
        if cost_ls < cost_best:
          xs = xs_ls
          us = us_ls
          cost_best = cost_ls
          break
      # print("iter", iter, " | cost_best: ", cost_best)
      # endregion

      # region: Updates operating points.
      self.last_operating_point = self.current_operating_point
      self.current_operating_point = (xs, us)

      self.last_social_cost = self.current_social_cost
      self.current_social_cost = cost_best

      if self.current_social_cost < self.best_social_cost:
        self.best_operating_point = self.current_operating_point
        self.best_social_cost = self.current_social_cost
      # endregion

      # region: Checks convergence.
      if self.is_converged_cost():
        status = 1
        if self.verbose:
          print(
              "[iLQGame] Social cost (", round(self.current_social_cost, 2), ") has converged! \n"
          )
        break
      # endregion

      t_iter = time.time() - t_start
      if self.verbose:
        print(
            "[iLQGame] Iteration", iter, "| Social cost: ", round(self.current_social_cost, 2),
            " | Iter. time: ", t_iter
        )

    t_process = time.time() - time0

    xs = np.asarray(xs).reshape(self.dim_x_ss, self.horizon, self.num_players)
    xs = [xs[:, :, ii] for ii in range(self.num_players)]
    us = np.asarray(us)
    us = [us[:, :, ii] for ii in range(self.num_players)]

    return xs, us, t_process, status

  def is_converged_cost(self):
    """
    Checks convergence based on social cost difference.
    """
    TOLERANCE_RATE = 0.005
    # COST_LB = 1e6

    if self.last_social_cost is None:
      return False

    cost_diff_rate = np.abs(
        (self.current_social_cost - self.last_social_cost) / self.last_social_cost
    )

    if cost_diff_rate > TOLERANCE_RATE:  #or self.current_social_cost > COST_LB:
      return False
    else:
      return True

  def reset(self, _current_x=None, _current_u=None, _current_J=None):
    """
    Resets the solver and warmstarts it if possible.
    """

    if _current_x is None:
      _current_x = jnp.zeros((self.dim_x, self.horizon))

    if _current_u is None:
      _current_u = jnp.zeros((self.dim_u_ss, self.horizon, self.num_players))

    self.last_operating_point = None
    self.current_operating_point = (_current_x, _current_u)
    self.best_operating_point = (_current_x, _current_u)

    self.last_social_cost = np.Inf
    self.current_social_cost = _current_J
    self.best_social_cost = _current_J

  def initial_forward_pass(self, cur_state: ArrayImpl, controls: ArrayImpl, targets: ArrayImpl):
    """
    Performs the initial forward pass given warmstart controls.

    Args:
        cur_state (ArrayImpl) (nx,)
        controls (ArrayImpl) (nui, N, num_players)
        targets: (ArrayImpl) (nxi, num_players)

    Returns:
        states (ArrayImpl): states (nx, N)
        social_cost (float): sum of all players costs
    """
    # Forward simulation.
    states = jnp.zeros((self.dim_x, self.horizon))
    states = states.at[:, 0].set(cur_state)
    for k in range(1, self.horizon):
      states_next, _ = self.dynamics.integrate_forward(states[:, k - 1], controls[:, k - 1, :])
      states = states.at[:, k].set(states_next)

    # Evaluates costs.
    cost_sum = self._evaluate_costs(states, controls, targets)

    return states, cost_sum

  def compute_operating_point(
      self, current_xs: ArrayImpl, current_us: ArrayImpl, Ps: ArrayImpl, alphas: ArrayImpl,
      cur_state: ArrayImpl, targets: ArrayImpl
  ) -> Tuple[ArrayImpl, ArrayImpl, float, ArrayImpl, ArrayImpl, ArrayImpl]:
    """
    Computes current operating point by propagating through dynamics.
    This function is a wrapper of _compute_operating_point_jax()

    Args:
        current_xs (ArrayImpl): (nx, N) current state traj, used as nominal
        current_us (ArrayImpl): (nui, N, num_players) current player controls, used as nominal
        Ps (ArrayImpl): (nui, nx, N, num_players)
        alphas (ArrayImpl): (nui, N, num_players)
        cur_state (ArrayImpl): (nx,) current (initial) state
        targets (ArrayImpl): (nxi, num_players)

    Returns:
        xs (ArrayImpl): updated states (nx, N)
        us (ArrayImpl): updated player controls (nui, N, num_players)
        social_cost (float): sum of all players costs
    """
    # Computes track info.
    xs, us = self._compute_operating_point_jax(current_xs, current_us, Ps, alphas, cur_state)

    # Evaluates costs.
    cost_sum = self._evaluate_costs(xs, us, targets)

    return xs, us, cost_sum

  @partial(jit, static_argnames='self')
  def _evaluate_costs(self, xs, us, targets):
    costs = jnp.zeros((self.num_players))
    for ii in range(self.num_players):
      costs = costs.at[ii].set(self.costs[ii].get_cost(xs, us[:, :, ii], targets[:, ii]))
    return jnp.sum(costs)

  @partial(jit, static_argnames='self')
  def _compute_operating_point_jax(
      self, nominal_states: ArrayImpl, nominal_controls: ArrayImpl, Ps: ArrayImpl,
      alphas: ArrayImpl, cur_state: ArrayImpl
  ) -> Tuple[ArrayImpl, ArrayImpl]:
    """
    Computes current operating point by propagating through dynamics.

    Args:
        nominal_states (ArrayImpl): (nx, N)
        nominal_controls (ArrayImpl): (nui, N, num_players)
        Ps (ArrayImpl): (nui, nx, N, num_players)
        alphas (ArrayImpl): (nui, N, num_players)
        cur_state (ArrayImpl): (nx,) current init. state

    Returns:
        xs (ArrayImpl): updated states (nx, N)
        us (ArrayImpl): updated player controls (nui, N, num_players)
    """

    def forward_pass_looper(k, _carry):

      def compute_agent_control(x, x_ref, uii_ref, Pii, alphaii):
        return uii_ref - Pii @ (x-x_ref) - alphaii

      compute_all_agents_controls = vmap(
          compute_agent_control, in_axes=(None, None, 1, 2, 1), out_axes=(1)
      )

      xs, us = _carry
      us_tmp = compute_all_agents_controls(
          xs[:, k], nominal_states[:, k], nominal_controls[:, k, :], Ps[:, :, k, :], alphas[:, k, :]
      )
      X_next, U_next = self.dynamics.integrate_forward(xs[:, k], us_tmp)
      xs = xs.at[:, k + 1].set(X_next)
      us = us.at[:, k, :].set(U_next)
      return xs, us

    xs = jnp.zeros_like(nominal_states)
    us = jnp.zeros_like(nominal_controls)
    xs = xs.at[:, 0].set(cur_state)
    xs, us = lax.fori_loop(0, self.horizon - 1, forward_pass_looper, (xs, us))
    return xs, us

  @partial(jit, static_argnames='self')
  def linearize_dynamics(self, xs: ArrayImpl, us: ArrayImpl) -> Tuple[ArrayImpl, ArrayImpl]:
    """
    Linearizes dynamics at the current operating point.

    Args:
        xs (ArrayImpl): (nx, N) nominal state traj
        us (ArrayImpl): (nui, N, num_players) nominal player controls

    Returns:
        As (ArrayImpl): (nx, nx, N) A matrices
        Bs (ArrayImpl): (nx, nui, N, num_players) B matrices
    """

    def linearize_single_time(x, u):
      A, B = self.dynamics.linearize_discrete_jitted(x, u)
      return A, B

    linearize_along_horizon = vmap(linearize_single_time, in_axes=(1, 1), out_axes=(2, 2))
    As, Bs = linearize_along_horizon(xs, us)

    return As, Bs

  @partial(jit, static_argnames='self')
  def quadraticize_costs(
      self,
      xs: ArrayImpl,
      us: ArrayImpl,
      targets: ArrayImpl,
  ) -> Tuple[ArrayImpl, ArrayImpl, ArrayImpl, ArrayImpl]:
    """
    Quadraticizes costs of all players at the current operating point.

    Args:
        xs (ArrayImpl): (nx, N) nominal state trajectory
        us (ArrayImpl): (nui, N, num_players) nominal player controls
        targets (ArrayImpl): (nxi, num_players)

    Returns:
        lxs (ArrayImpl): (nx, N, num_players) gradients lx = dc/dx of all playes
        lus (ArrayImpl): (nui, N, num_players) gradients lu = dc/du of all playes
        Hxxs (ArrayImpl): (nx, nx, N, num_players) Hessians Hxx of all playes
        Huus (ArrayImpl): (nui, nui, N, num_players) Hessians Huu of all playes
    """
    lxs = jnp.zeros((self.dim_x, self.horizon, self.num_players))
    lus = jnp.zeros((self.dim_u_ss, self.horizon, self.num_players))
    Hxxs = jnp.zeros((self.dim_x, self.dim_x, self.horizon, self.num_players))
    Huus = jnp.zeros((self.dim_u_ss, self.dim_u_ss, self.horizon, self.num_players))

    for ii in range(self.num_players):
      lxs_ii, lus_ii, Hxxs_ii, Huus_ii = self.costs[ii].get_derivatives(
          xs, us[:, :, ii], targets[:, ii]
      )
      lxs = lxs.at[:, :, ii].set(lxs_ii)
      lus = lus.at[:, :, ii].set(lus_ii)
      Hxxs = Hxxs.at[:, :, :, ii].set(Hxxs_ii)
      Huus = Huus.at[:, :, :, ii].set(Huus_ii)
    return lxs, lus, Hxxs, Huus

  @partial(jit, static_argnames='self')
  def backward_pass(
      self, As: ArrayImpl, Bs: ArrayImpl, lxs: ArrayImpl, lus: ArrayImpl, Hxxs: ArrayImpl,
      Huus: ArrayImpl
  ) -> Tuple[ArrayImpl, ArrayImpl, ArrayImpl, ArrayImpl]:
    """
    Solves a time-varying, finite horizon LQ game (finds closed-loop Nash
    feedback strategies for both players).
    Assumes that dynamics are given by
            ``` dx_{k+1} = A_k dx_k + \sum_i Bs[i]_k du[i]_k ```

    Derivation can be found in:
        https://github.com/HJReachability/ilqgames/blob/master/derivations/feedback_lq_nash.pdf

    Args:
        As (ArrayImpl): (nx, nx, N) A matrices
        Bs (ArrayImpl): (nui, nui, N, num_players) B matrices
        lxs (ArrayImpl): (nx, N, num_players) gradients lx = dc/dx of all playes
        lus (ArrayImpl): (nui, N, num_players) gradients lu = dc/du of all playes
        Hxxs (ArrayImpl): (nx, nx, N, num_players) Hessians Hxx of all playes
        Huus (ArrayImpl): (nui, nui, N, num_players) Hessians Huu of all playes

    Returns:
        ArrayImpl: Ps (dim_u_ss, dim_x, N-1, num_players)
        ArrayImpl: alphas (dim_u_ss, N-1, num_players)
    """

    @jit
    def backward_pass_looper(k, _carry):
      Ps, alphas, Z, zeta = _carry
      n = horizon - 1 - k

      # Computes Ps given previously computed Z.
      S = jnp.array(()).reshape(0, sum(self.dynamics._u_dims))
      Y1 = jnp.array(()).reshape(0, dim_x)
      for ii in range(num_players):
        Sii = jnp.array(()).reshape(dim_u_ss, 0)
        for jj in range(num_players):
          if jj == ii:
            Sii = jnp.hstack(
                (Sii, Bs[:, :, n, ii].T @ Z[:, :, ii] @ Bs[:, :, n, jj] + Huus[:, :, n, ii])
            )
          else:
            Sii = jnp.hstack((Sii, Bs[:, :, n, ii].T @ Z[:, :, ii] @ Bs[:, :, n, jj]))
        S = jnp.vstack((S, Sii))

        Y1ii = Bs[:, :, n, ii].T @ Z[:, :, ii] @ As[:, :, n]
        Y1 = jnp.vstack((Y1, Y1ii))

      P, _, _, _ = jnp.linalg.lstsq(a=S, b=Y1, rcond=None)
      # Sinv = jnp.linalg.pinv(S)
      # P = Sinv @ Y1

      for ii in range(num_players):
        Pii = self.dynamics._LMu[:, :, ii] @ P
        Ps = Ps.at[:, :, n, ii].set(Pii)

      # Computes F_k = A_k - B1_k P1_k - B2_k P2_k -...
      F = As[:, :, n]
      for ii in range(num_players):
        F -= Bs[:, :, n, ii] @ Ps[:, :, n, ii]

      # Computes alphas using previously computed zetas.
      Y2 = jnp.array(()).reshape(0, 1)
      for ii in range(num_players):
        # Y2ii = (Bs[:, :, n, ii].T @ zeta[:, ii]).reshape((dim_u_ss, 1))
        Y2ii = (Bs[:, :, n, ii].T @ zeta[:, ii] + lus[:, n, ii]).reshape((dim_u_ss, 1))
        Y2 = jnp.vstack((Y2, Y2ii))

      alpha, _, _, _ = jnp.linalg.lstsq(a=S, b=Y2, rcond=None)
      # alpha = Sinv @ Y2

      for ii in range(num_players):
        alphaii = self.dynamics._LMu[:, :, ii] @ alpha
        alphas = alphas.at[:, n, ii].set(alphaii[:, 0])

      # Computes beta_k = -B1_k alpha1 - B2_k alpha2_k -...
      beta = 0.
      for ii in range(num_players):
        beta -= Bs[:, :, n, ii] @ alphas[:, n, ii]

      # Updates zeta.
      for ii in range(num_players):
        _FZb = F.T @ (zeta[:, ii] + Z[:, :, ii] @ beta)
        _PRa = Ps[:, :, n, ii].T @ Huus[:, :, n, ii] @ alphas[:, n, ii]
        zeta = zeta.at[:, ii].set(_FZb + _PRa + lxs[:, n, ii])

      # Updates Z.
      for ii in range(num_players):
        _FZF = F.T @ Z[:, :, ii] @ F
        _PRP = Ps[:, :, n, ii].T @ Huus[:, :, n, ii] @ Ps[:, :, n, ii]
        Z = Z.at[:, :, ii].set(_FZF + _PRP + Hxxs[:, :, n, ii])

      return Ps, alphas, Z, zeta

    # Unpacks horizon and number of players.
    horizon = self.horizon
    num_players = self.num_players

    # Caches dimensions of state and controls for each player.
    dim_x = self.dim_x
    dim_u_ss = self.dim_u_ss

    # Recursively computes all intermediate and final variables.
    Z = Hxxs[:, :, -1, :]
    zeta = lxs[:, -1, :]

    # Initializes strategy matrices.
    Ps = jnp.zeros((dim_u_ss, dim_x, horizon, num_players))
    alphas = jnp.zeros((dim_u_ss, horizon, num_players))

    # Backward pass.
    Ps, alphas, Z, zeta = lax.fori_loop(
        0, self.horizon, backward_pass_looper, (Ps, alphas, Z, zeta)
    )

    return Ps, alphas, Z, zeta
