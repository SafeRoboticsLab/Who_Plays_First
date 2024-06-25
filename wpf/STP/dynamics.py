"""
Dynamics.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
"""

import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod

from functools import partial
from jax import jit, jacfwd
from jaxlib.xla_extension import ArrayImpl
import jax.numpy as jnp
import jax
from scipy.signal import cont2discrete as c2d

from .utils import Struct


class Dynamics(ABC):

  def __init__(self, config: Struct):
    self.dim_x = config.DIM_X
    self.dim_u = config.DIM_U

    self.T = config.T  # planning time horizon
    self.N = config.N  # number of planning steps
    self.dt = self.T / (self.N - 1)  # time step for each planning step

    # Useful constants.
    self.zeros = np.zeros((self.N))
    self.ones = np.ones((self.N))

    # Computes Jacobian matrices using Jax.
    self.jac_f = jit(jacfwd(self.dct_time_dyn, argnums=[0, 1]))

    # Vectorizes Jacobians using Jax.
    self.jac_f = jit(jax.vmap(self.jac_f, in_axes=(1, 1), out_axes=(2, 2)))

  @abstractmethod
  def dct_time_dyn(self, state: ArrayImpl, control: ArrayImpl) -> ArrayImpl:
    """
    Computes the one-step time evolution of the system with the forward Euler method.
    Args:
        state (ArrayImpl):   (nx,)
        control (ArrayImpl): (nu,)

    Returns:
        ArrayImpl: (nx,) next state.
    """
    raise NotImplementedError

  @abstractmethod
  def integrate_forward(self, state: ArrayImpl, control: ArrayImpl) -> Tuple[ArrayImpl, ArrayImpl]:
    """
    Computes the next state.

    Args:
        state (ArrayImpl):   (nx,)
        control (ArrayImpl): (nu,)

    Returns:
        state_next: (nx,) ArrayImpl
        control_next: (nu,) ArrayImpl
    """
    raise NotImplementedError

  @partial(jit, static_argnames="self")
  def get_AB_matrix(self, nominal_states: ArrayImpl,
                    nominal_controls: ArrayImpl) -> Tuple[ArrayImpl, ArrayImpl]:
    """
    Returns the linearized 'A' and 'B' matrix of the ego vehicle around
    nominal states and controls.

    Args:
        nominal_states (ArrayImpl): (nx, N) states along the nominal traj.
        nominal_controls (ArrayImpl): (nu, N) controls along the traj.

    Returns:
        ArrayImpl: the Jacobian of next state w.r.t. the current state.
        ArrayImpl: the Jacobian of next state w.r.t. the current control.
    """
    A, B = self.jac_f(nominal_states, nominal_controls)
    return A, B


class DubinsCar(Dynamics):

  def __init__(self, config: Struct):
    Dynamics.__init__(self, config)
    self.delta_min = config.DELTA_MIN  # min turn rate (rad/s)
    self.delta_max = config.DELTA_MAX  # max turn rate (rad/s)
    self.a_min = config.A_MIN  # min longitudial accel
    self.a_max = config.A_MAX  # max longitudial accel
    self.v_min = config.V_MIN  # min velocity
    self.v_max = config.V_MAX  # max velocity

  @partial(jit, static_argnames="self")
  def dct_time_dyn(self, state: ArrayImpl, control: ArrayImpl) -> ArrayImpl:
    """
    Computes the one-step time evolution of the system with the forward Euler method.
    Dynamics:
        \dot{x}   = v cos(psi)
        \dot{y}   = v sin(psi)
        \dot{v}   = a
        \dot{psi} = delta

    Args:
                              0  1  2  3
        state (ArrayImpl):   [x, y, v, psi]
        control (ArrayImpl): [a, delta]

    Returns:
        ArrayImpl: (4,) next state.
    """
    d_x = state[2] * jnp.cos(state[3])
    d_y = state[2] * jnp.sin(state[3])
    d_v = control[0]
    d_psi = control[1]
    state_next = state + jnp.hstack((d_x, d_y, d_v, d_psi)) * self.dt
    return state_next

  @partial(jit, static_argnames="self")
  def integrate_forward(self, state: ArrayImpl, control: ArrayImpl) -> Tuple[ArrayImpl, ArrayImpl]:
    """
    Computes the next state.

    Args:
        state (ArrayImpl): (4,) jnp array [x, y, v, psi].
        control (ArrayImpl): (2,) jnp array [a, delta].

    Returns:
        state_next: ArrayImpl
        control_next: ArrayImpl
    """
    # Clips the control values with their limits.
    accel = jnp.clip(control[0], self.a_min, self.a_max)
    delta = jnp.clip(control[1], self.delta_min, self.delta_max)

    # Integrates the system one-step forward in time using the Euler method.
    control_clip = jnp.hstack((accel, delta))
    state_next = self.dct_time_dyn(state, control_clip)

    return state_next, control_clip

  @partial(jit, static_argnames="self")
  def integrate_forward_norev(self, state: ArrayImpl,
                              control: ArrayImpl) -> Tuple[ArrayImpl, ArrayImpl]:
    """
    Computes the next state. The velocity is clipped so that the car cannot back up.

    Args:
        state (ArrayImpl): (4,) jnp array [x, y, v, psi].
        control (ArrayImpl): (2,) jnp array [a, delta].

    Returns:
        state_next: ArrayImpl
        control_next: ArrayImpl
    """
    # Clips the control values with their limits.
    accel = jnp.clip(control[0], self.a_min, self.a_max)
    delta = jnp.clip(control[1], self.delta_min, self.delta_max)

    # Integrates the system one-step forward in time using the Euler method.
    control_clip = jnp.hstack((accel, delta))
    state_next = self.dct_time_dyn(state, control_clip)
    state_next = state_next.at[2].set(jnp.maximum(state_next[2], 0.0))  # car cannot back up.

    return state_next, control_clip


class DoubleIntegrator(Dynamics):

  def __init__(self, config: Struct):
    Dynamics.__init__(self, config)
    self.ax_min = config.AX_MIN  # min x accel
    self.ax_max = config.AX_MAX  # max x accel
    self.ay_min = config.AY_MIN  # min y accel
    self.ay_max = config.AY_MAX  # max y accel
    self.Ac = np.array(([0., 0., 1., 0.], [0., 0., 0., 1.], [0., 0., 0., 0.], [0., 0., 0., 0.]))
    self.Bc = np.array(([0., 0.], [0., 0.], [1., 0.], [0., 1.]))
    _Ad, _Bd, _, _, _ = c2d(system=(self.Ac, self.Bc, None, None), dt=self.dt, method='zoh')
    self.Ad, self.Bd = jnp.asarray(_Ad), jnp.asarray(_Bd)

  @partial(jit, static_argnames="self")
  def dct_time_dyn(self, state: ArrayImpl, control: ArrayImpl) -> ArrayImpl:
    """
    Computes the one-step time evolution of the system with the forward Euler method.
    Dynamics:
        \dot{x}  = vx
        \dot{y}  = vy
        \dot{vx} = ax
        \dot{vy} = ay

    Args:
                              0   1  2   3
        state (ArrayImpl):   [x,  y, vx, vy]
        control (ArrayImpl): [ax, ay]

    Returns:
        ArrayImpl: (4,) next state.
    """
    state_next = self.Ad @ state + self.Bd @ control
    return state_next

  @partial(jit, static_argnames="self")
  def integrate_forward(self, state: ArrayImpl, control: ArrayImpl) -> Tuple[ArrayImpl, ArrayImpl]:
    """
    Computes the next state.

    Args:
        state (ArrayImpl): (4,) jnp array [x,  y, vx, vy]
        control (ArrayImpl): (2,) jnp array [ax, ay]

    Returns:
        state_next: ArrayImpl
        control_next: ArrayImpl
    """
    # Clips the control values with their limits.
    ax = jnp.clip(control[0], self.ax_min, self.ax_max)
    ay = jnp.clip(control[1], self.ay_min, self.ay_max)

    # Integrates the system one-step forward in time using the Euler method.
    control_clip = jnp.hstack((ax, ay))
    state_next = self.dct_time_dyn(state, control_clip)

    return state_next, control_clip
