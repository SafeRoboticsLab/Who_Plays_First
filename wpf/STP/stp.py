"""
Sequential ILQR-based trajectory planning.

Please contact the author(s) of this library if you have any questions.
Author:  Haimin Hu (haiminh@princeton.edu)
"""

import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, vmap, lax
from jaxlib.xla_extension import ArrayImpl
from typing import Tuple, List, Dict
from copy import deepcopy

from .ilqr import iLQR, LQR_shielded
from .utils import Struct


class STP:

  def __init__(self, config: Struct):
    """
    Initializer.

    Args:
        config (Struct): config file
        order (List): list of int or None
    """

    self.config = config
    self.horizon = config.N
    self.max_iter = config.MAX_ITER
    self.num_agents = config.N_AGENT

    # Pre-compile all sub-iLQR problems
    self.solvers = []
    for iagent in range(self.num_agents):
      self.solvers.append(iLQR(config, num_leaders=iagent))

  def get_init_ctrl_from_LQR(
      self, N: float, cur_state: List[np.ndarray], targets: List[np.ndarray]
  ) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    @jit
    def get_LQR_ctrl(x_init, target):

      def _looper(i, carry):
        states, controls = carry
        control_LQR = -K_lqr @ (states[:, i] - _target)
        state_next, control = self.solvers[0].dynamics.integrate_forward(states[:, i], control_LQR)
        return states.at[:, i + 1].set(state_next), controls.at[:, i].set(control)

      states = jnp.zeros((self.config.DIM_X, N))
      states = states.at[:, 0].set(x_init)
      controls = jnp.zeros((self.config.DIM_U, N))
      _target = jnp.array([target[0], target[1], 0., 0.])
      states, controls = lax.fori_loop(0, N, _looper, (states, controls))
      return states, controls

    K_lqr = self.solvers[0].cost.K_lqr
    state_list, ctrl_list = [], []
    for ii in range(len(cur_state)):
      states_ii, controls_ii = get_LQR_ctrl(cur_state[ii], targets[ii])
      state_list.append(states_ii)
      ctrl_list.append(controls_ii)
    return state_list, ctrl_list

  def solve(
      self, cur_state: List[np.ndarray], ctrls_ws: List[np.ndarray], targets: List[np.ndarray],
      order: np.ndarray
  ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float]]:
    """
    Solves the iLQR-STP problem.

    Args:
        cur_state (List[np.ndarray]): [(dim_x,)]
        ctrls_ws (List[np.ndarray]): [(self.dim_u, N - 1)] warmstart control sequences
        targets (List[np.ndarray]): List of agents' target states
        order (np.ndarray): [id of player who plays 1st, id of player who plays 2nd, ...]

    Returns:
        states: List[np.ndarray]
        controls: List[np.ndarray]
        Js: List[float]
        ts: List[float]
    """
    _Na = self.num_agents
    states, controls, Js, ts = (
        [None] * _Na,
        [None] * _Na,
        [None] * _Na,
        [None] * _Na,
    )
    leader_trajs = []  # arranged according to orders but agnostic to agent id

    cur_state = [jnp.asarray(_cur_state) for _cur_state in cur_state]
    ctrls_ws = [jnp.asarray(_ctrls_ws) for _ctrls_ws in ctrls_ws]
    targets = [jnp.asarray(_target) for _target in targets]

    # Computes trajectories for agents who are assigned order of play.
    assert len(order) == self.num_agents
    _assigned_agents = []
    for i_order in range(len(order)):
      agent_id = order[i_order]
      if (agent_id is None):  # assuming order = [agent assigned order | agent not assigned order]
        break
      else:
        _assigned_agents.append(agent_id)
        if len(leader_trajs) > 0:
          _leader_trajs = np.stack(leader_trajs, axis=2)
        else:
          _leader_trajs = None
        states_tmp, controls_tmp, J_tmp, t_tmp, _ = self.solvers[i_order].solve(
            cur_state[agent_id],
            ctrls_ws[agent_id],
            _leader_trajs,
            targets[agent_id],
        )
        states[agent_id], controls[agent_id], Js[agent_id], ts[agent_id] = (
            states_tmp,
            controls_tmp,
            J_tmp,
            t_tmp,
        )
        leader_trajs.append(states[agent_id])

    # Computes trajectories for unassigned agents.
    if len(leader_trajs) > 0:
      _leader_trajs = np.stack(leader_trajs, axis=2)
    else:
      _leader_trajs = None

    if len(_assigned_agents) < self.num_agents:
      _order_unassigned = len(_assigned_agents)
      assert _order_unassigned == len(leader_trajs)
      for agent_id in range(self.num_agents):
        if agent_id not in _assigned_agents:
          # print('Unassigned agent detected, id =', agent_id)

          # -> Option A. Unassigned players play last.
          if not (hasattr(self.config, "OPTION_B") and self.config.OPTION_B):
            states_tmp, controls_tmp, J_tmp, t_tmp, _ = self.solvers[_order_unassigned].solve(
                cur_state[agent_id], ctrls_ws[agent_id], _leader_trajs, targets[agent_id]
            )

          # -> Option B. Unassigned players play blindly.
          else:
            states_tmp, controls_tmp, J_tmp, t_tmp, _ = self.solvers[0].solve(
                cur_state[agent_id], ctrls_ws[agent_id], None, targets[agent_id]
            )

          states[agent_id], controls[agent_id], Js[agent_id], ts[agent_id] = (
              states_tmp,
              controls_tmp,
              J_tmp,
              t_tmp,
          )

    states = [np.asarray(_states) for _states in states]
    controls = [np.asarray(_controls) for _controls in controls]
    Js = [float(_J) for _J in Js]

    return states, controls, Js, ts

  def solve_rhc(
      self,
      cur_state: List[np.ndarray],
      ctrls_ws: List[np.ndarray],
      targets: List[np.ndarray],
      order: np.ndarray,
  ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float]]:
    """
    Solves the iLQR-STP problem with receding horizon control.

    Args:
        cur_state (List[np.ndarray]): [(dim_x,)]
        ctrls_ws (List[np.ndarray]): [(self.dim_u, N - 1)] warmstart control sequences
        targets (List[np.ndarray]): List of agents' target states
        order (np.ndarray): [id of player who plays 1st, id of player who plays 2nd, ...]

    Returns:
        states: List[np.ndarray]
        controls: List[np.ndarray]
        Js: List[float]
        ts: List[float]
    """

    def _update_rhc(states_tmp, controls_tmp, t_tmp, leader_trajs, agent_id, i_order, horizon):
      if leader_trajs is not None:
        leader_trajs = leader_trajs[:, :_ols]

      if horizon == 0:
        states[agent_id], controls[agent_id], ts[agent_id] = (
            states_tmp[:, :_ols],
            controls_tmp[:, :_ols],
            t_tmp,
        )
        Js[agent_id] = self.solvers[i_order].compute_cost(
            states_tmp[:, :_ols],
            controls_tmp[:, :_ols],
            leader_trajs,
            targets[agent_id],
        )

      else:
        states[agent_id] = jnp.concatenate((states[agent_id], states_tmp[:, 1:_ols + 1]), axis=1)
        controls[agent_id] = jnp.concatenate((controls[agent_id], controls_tmp[:, :_ols]), axis=1)
        ts[agent_id] += t_tmp
        Js[agent_id] += self.solvers[i_order].compute_cost(
            states_tmp[:, :_ols],
            controls_tmp[:, :_ols],
            leader_trajs,
            targets[agent_id],
        )

      ctrls_ws[agent_id] = ctrls_ws[agent_id].at[:, :-1].set(controls_tmp[:, 1:])

    _N = self.config.N
    _Na = self.num_agents
    _ols = (self.config.OPEN_LOOP_STEP)  # Open-loop simulation steps between two RHC cycles.

    states, controls, Js, ts = (
        [None] * _Na,
        [None] * _Na,
        [None] * _Na,
        [None] * _Na,
    )
    cur_state = [jnp.asarray(_cur_state) for _cur_state in cur_state]
    ctrls_ws = [jnp.asarray(_ctrls_ws[:, :_N]) for _ctrls_ws in ctrls_ws]
    targets = [jnp.asarray(_target) for _target in targets]

    horizon = 0

    while horizon < self.config.RHC_STEPS:
      leader_trajs = []  # arranged according to orders but agnostic to agent id

      # Computes trajectories for agents who are assigned order of play.
      assert len(order) == self.num_agents
      _assigned_agents = []
      for i_order in range(len(order)):
        agent_id = order[i_order]
        if (agent_id is None):  # assuming order = [agent assigned order | agent not assigned order]
          break
        else:
          _assigned_agents.append(agent_id)
          if len(leader_trajs) > 0:
            _leader_trajs = np.stack(leader_trajs, axis=2)
          else:
            _leader_trajs = None
          states_tmp, controls_tmp, _, t_tmp, _ = self.solvers[i_order].solve(
              cur_state[agent_id],
              ctrls_ws[agent_id],
              _leader_trajs,
              targets[agent_id],
          )
          _update_rhc(
              states_tmp,
              controls_tmp,
              t_tmp,
              _leader_trajs,
              agent_id,
              i_order,
              horizon,
          )
          leader_trajs.append(states_tmp)

      # Computes trajectories for unassigned agents.
      if len(leader_trajs) > 0:
        _leader_trajs = np.stack(leader_trajs, axis=2)
      else:
        _leader_trajs = None

      if len(_assigned_agents) < self.num_agents:
        _order_unassigned = len(_assigned_agents)
        assert _order_unassigned == len(leader_trajs)
        for agent_id in range(self.num_agents):
          if agent_id not in _assigned_agents:
            # print('Unassigned agent detected, id =', agent_id)

            # -> Option A. Unassigned players play last.
            if not (hasattr(self.config, "OPTION_B") and self.config.OPTION_B):
              states_tmp, controls_tmp, _, t_tmp, _ = self.solvers[_order_unassigned].solve(
                  cur_state[agent_id], ctrls_ws[agent_id], _leader_trajs, targets[agent_id]
              )
              _update_rhc(
                  states_tmp, controls_tmp, t_tmp, _leader_trajs, agent_id, _order_unassigned,
                  horizon
              )

            # -> Option B. Unassigned players play blindly.
            else:
              states_tmp, controls_tmp, _, t_tmp, _ = self.solvers[0].solve(
                  cur_state[agent_id], ctrls_ws[agent_id], None, targets[agent_id]
              )
              _update_rhc(states_tmp, controls_tmp, t_tmp, None, agent_id, 0, horizon)

      # Updates current states.
      cur_state = [_state[:, -1] for _state in states]

      horizon += _ols

    # Prepares return values.
    states = [np.asarray(_states) for _states in states]
    controls = [np.asarray(_controls) for _controls in controls]
    Js = [float(_J) for _J in Js]

    return states, controls, Js, ts

  @partial(jit, static_argnames="self")
  def pairwise_collision_check(self, states: ArrayImpl):
    """
    Checks collisions pairwise for all agents.

    Args:
        states (ArrayImpl): (dim_x, N, N_agents)
    """

    def _check_two_agents(state1, state2):
      _dx = self.config.PX_DIM
      _dy = self.config.PY_DIM
      pxpy_diff = jnp.stack((state1[_dx, :] - state2[_dx, :], state1[_dy, :] - state2[_dy, :]))
      sep = jnp.linalg.norm(pxpy_diff, axis=0)
      return jnp.any(sep < self.config.PROX_SEP_CHECK)

    def _looper(i, col_mat):
      _check_two_agents_vmap = vmap(_check_two_agents, in_axes=(None, 2), out_axes=(0))
      _col_i = _check_two_agents_vmap(states[:, :, i], states)
      _col_i = _col_i.at[i].set(False)
      return col_mat.at[i, :].set(_col_i)

    col_mat = jnp.zeros((self.num_agents, self.num_agents), dtype=bool)
    col_mat = lax.fori_loop(0, self.num_agents, _looper, col_mat)
    return col_mat


class STPMovingTargets(STP):

  def __init__(self, config: Struct):
    STP.__init__(self, config)

    self.solvers = []
    for iagent in range(self.num_agents):
      self.solvers.append(iLQR(config, num_leaders=iagent + 1))

  @staticmethod
  def generate_circle_positions(r, N):
    angles = np.linspace(-np.pi / 2., np.pi / 2., N, endpoint=True)
    positions_x = r * np.cos(angles)
    positions_y = r * np.sin(angles)
    return positions_x, positions_y

  def get_init_ctrl_from_LQR(
      self, N: float, cur_state: List[np.ndarray], history: Dict, perm: List[int]
  ) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    @jit
    def get_LQR_ctrl(x_init, target):

      def _looper(i, carry):
        states, controls, = carry
        control_LQR = -K_lqr @ (states[:, i] - target[:, i])
        state_next, control = self.solvers[0].dynamics.integrate_forward(states[:, i], control_LQR)
        return states.at[:, i + 1].set(state_next), controls.at[:, i].set(control)

      states = jnp.zeros((self.config.DIM_X, N))
      states = states.at[:, 0].set(x_init)
      controls = jnp.zeros((self.config.DIM_U, N))
      states, controls = lax.fori_loop(0, N, _looper, (states, controls))
      return states, controls

    def set_targets(agent_id):
      _targets = deepcopy(history["alpha"])
      _targets[0, :] += tar_xs[agent_id]
      _targets[1, :] += tar_ys[agent_id]
      return _targets

    tar_xs, tar_ys = self.generate_circle_positions(self.config.TAR_RADIUS, self.num_agents)
    K_lqr = self.solvers[0].cost.K_lqr
    state_list, ctrl_list = [], []
    for ip in range(len(perm)):
      ii = perm[ip]
      _targets = set_targets(ii)
      # if ip == 0:
      #   states_ii, controls_ii = get_LQR_ctrl(cur_state[ii], history["alpha"])
      # else:
      #   states_ii, controls_ii = get_LQR_ctrl(cur_state[ii], history["stp"][ip - 1])
      states_ii, controls_ii = get_LQR_ctrl(cur_state[ii], _targets)
      state_list.append(states_ii)
      ctrl_list.append(controls_ii)
    return state_list, ctrl_list

  @partial(jit, static_argnames="self")
  def pairwise_collision_check_alpha(self, states: ArrayImpl):
    """
    Checks collisions pairwise for all agents (including the alpha).

    Args:
        states (ArrayImpl): (dim_x, N, N_agents + 1)
    """

    def _check_two_agents(state1, state2):
      _dx = self.config.PX_DIM
      _dy = self.config.PY_DIM
      pxpy_diff = jnp.stack((state1[_dx, :] - state2[_dx, :], state1[_dy, :] - state2[_dy, :]))
      sep = jnp.linalg.norm(pxpy_diff, axis=0)
      return jnp.any(sep < self.config.PROX_SEP_CHECK)

    def _looper(i, col_mat):
      _check_two_agents_vmap = vmap(_check_two_agents, in_axes=(None, 2), out_axes=(0))
      _col_i = _check_two_agents_vmap(states[:, :, i], states)
      _col_i = _col_i.at[i].set(False)
      return col_mat.at[i, :].set(_col_i)

    col_mat = jnp.zeros((self.num_agents + 1, self.num_agents + 1), dtype=bool)
    col_mat = lax.fori_loop(0, self.num_agents + 1, _looper, col_mat)
    return col_mat

  def solve(
      self, cur_state: List[np.ndarray], ctrls_ws: List[np.ndarray], alpha_traj: np.ndarray,
      history: Dict, order: np.ndarray
  ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float]]:
    """
    Solves the iLQR-STP problem

    Args:
        cur_state (List[np.ndarray]): [(dim_x,)]
        ctrls_ws (List[np.ndarray]): [(self.dim_u, N - 1)] warmstart control sequences
        alpha_traj (np.ndarray): The alpha's planned trajectory
        history (Dict): Agent's historic trajectory (keys: "alpha", "stp")
        order (np.ndarray): [id of player who plays 1st, id of player who plays 2nd, ...]

    Returns:
        states: List[np.ndarray]
        controls: List[np.ndarray]
        Js: List[float]
        ts: List[float]
    """

    def shift_by_delay(states, delay):
      states = deepcopy(states)
      states[:, :-delay] = states[:, delay:]
      states[:, -delay:] = states[:, -delay - 1:-delay]
      return states

    def set_targets(agent_id):
      # if agent_id == 0:
      #   _targets = history["alpha"]
      # else:
      #   _targets = history["stp"][agent_id - 1]
      # _targets = shift_by_delay(_targets, _delay)
      _targets = deepcopy(history["alpha"])
      # _targets = deepcopy(alpha_traj)
      if _delay > 0:
        _targets = shift_by_delay(_targets, _delay)
      _targets[0, :] += tar_xs[agent_id]
      _targets[1, :] += tar_ys[agent_id]
      return _targets

    _Na = self.num_agents
    _delay = self.config.DELAY
    tar_xs, tar_ys = self.generate_circle_positions(self.config.TAR_RADIUS, _Na)

    states, controls, Js, ts = ([None] * _Na, [None] * _Na, [None] * _Na, [None] * _Na)
    leader_trajs = [alpha_traj]  # arranged according to orders but agnostic to agent id

    cur_state = [jnp.asarray(_cur_state) for _cur_state in cur_state]
    ctrls_ws = [jnp.asarray(_ctrls_ws) for _ctrls_ws in ctrls_ws]

    # Computes trajectories for agents who are assigned order of play.
    assert len(order) == self.num_agents
    _assigned_agents = []
    for i_order in range(len(order)):
      agent_id = order[i_order]
      if (agent_id is None):  # assuming order = [agent assigned order | agent not assigned order]
        break
      else:
        _assigned_agents.append(agent_id)

        # Sets moving obstacles.
        _leader_trajs = np.stack(leader_trajs, axis=2)

        # Sets moving targets.
        _targets = set_targets(agent_id)

        # Solves iLQR/LQR.
        states_tmp, controls_tmp, J_tmp, t_tmp, _ = self.solvers[i_order].solve(
            cur_state[agent_id], ctrls_ws[agent_id], _leader_trajs, _targets
        )
        states[agent_id], controls[agent_id], Js[agent_id], ts[agent_id] = (
            states_tmp, controls_tmp, J_tmp, t_tmp
        )
        leader_trajs.append(states[agent_id])

    # Computes trajectories for unassigned agents
    _leader_trajs = np.stack(leader_trajs, axis=2)

    if len(_assigned_agents) < self.num_agents:
      _order_unassigned = len(_assigned_agents)
      assert _order_unassigned == len(leader_trajs) - 1  # alpha traj is always there
      for agent_id in range(self.num_agents):
        if agent_id not in _assigned_agents:
          # print('Unassigned agent detected, id =', agent_id)

          # Sets moving targets.
          _targets = set_targets(agent_id)

          # -> Option A. Unassigned players play last.
          if not (hasattr(self.config, "OPTION_B") and self.config.OPTION_B):
            states_tmp, controls_tmp, J_tmp, t_tmp, _ = self.solvers[_order_unassigned + 1].solve(
                cur_state[agent_id], ctrls_ws[agent_id], _leader_trajs, _targets
            )

          # -> Option B. Unassigned players play blindly.
          else:
            states_tmp, controls_tmp, J_tmp, t_tmp, _ = self.solvers[0].solve(
                cur_state[agent_id], ctrls_ws[agent_id], None, _targets
            )

          states[agent_id], controls[agent_id], Js[agent_id], ts[agent_id] = (
              states_tmp, controls_tmp, J_tmp, t_tmp
          )

    states = [np.asarray(_states) for _states in states]
    controls = [np.asarray(_controls) for _controls in controls]
    Js = [float(_J) for _J in Js]

    return states, controls, Js, ts

  def solve_rhc(
      self, cur_state: List[np.ndarray], ctrls_ws: List[np.ndarray], targets: List[np.ndarray],
      order: np.ndarray
  ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float]]:
    return NotImplementedError
