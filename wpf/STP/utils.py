"""
Util functions.

Please contact the author(s) of this library if you have any questions.
Author: Haimin Hu (haiminh@princeton.edu)
"""

import os
import yaml
import numpy as np
import imageio.v2 as imageio
from typing import List
from IPython.display import Image
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.transforms import Affine2D

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


class ATCZone:
  """
  Air traffic control (ATC) zone.
  """

  def __init__(self, config, targets: List[np.ndarray], oz_planner, tr_planner=None) -> None:
    self.center = [config.ATC_X, config.ATC_Y]
    self.radius = config.ATC_RADIUS
    self.targets = targets
    self.target_radius = config.TARGET_RADIUS
    self.reach_radius = config.REACH_RADIUS
    self.oz_planner = oz_planner  # out-of-zone planner
    self.tr_planner = tr_planner  # target-reaching planner
    self.num_agent = config.N_AGENT
    self.reach_flags = [False] * self.num_agent
    self.zone_flags = [False] * self.num_agent

  def reset(self):
    self.reach_flags = [False] * self.num_agent
    self.zone_flags = [False] * self.num_agent

  def is_in_zone(self, states: List[np.ndarray]) -> List[bool]:
    """
    Checks if agents are in the ATC zone.
    """
    return [
        (s[0] - self.center[0])**2 + (s[1] - self.center[1])**2 <= self.radius**2 for s in states
    ]

  def is_near_target(self, states: List[np.ndarray]) -> List[bool]:
    return [(s[0] - t[0])**2 + (s[1] - t[1])**2 <= self.target_radius**2
            for (s, t) in zip(states, self.targets)]

  def is_reach_target(self, states: List[np.ndarray]) -> List[bool]:
    _reach_flags = [(s[0] - t[0])**2 + (s[1] - t[1])**2 <= self.reach_radius**2
                    for (s, t) in zip(states, self.targets)]
    self.reach_flags = [_a or _b for _a, _b in zip(_reach_flags, self.reach_flags)]
    return self.reach_flags

  def is_collision(self, states: List[np.ndarray]) -> List[bool]:
    if len(states[0].shape) == 1:
      states = [s[:, np.newaxis] for s in states]
    states = np.stack(states, axis=2)
    return (self.oz_planner.pairwise_collision_check(states)).any()

  def plan_stp(self, x_cur, init_control, targets, order=None):
    """
    Plan STP trajectories.
    """
    if order is None:
      order = [None] * len(x_cur)
    states, controls, _, _ = self.oz_planner.solve(x_cur, init_control, targets, order)
    return states, controls

  def plan_stp_rhc(self, x_cur, init_control, targets, order=None):
    """
    Plan STP trajectories with the RHC mode.
    """
    if order is None:
      order = [None] * len(x_cur)
    states, controls, Js, _ = self.oz_planner.solve_rhc(x_cur, init_control, targets, order)
    return states, controls, Js

  def check_sep(self, x_cur, thresh):
    for ii in range(self.num_agent):
      for jj in range(self.num_agent):
        if ii != jj:
          sep = (x_cur[ii][0] - x_cur[jj][0])**2 + (x_cur[ii][1] - x_cur[jj][1])**2
          if sep < thresh**2:
            return True
    return False

  def check_zone(self, x_cur, x_new, us_cur, init_control):
    """
    Checks agent location in zone.
    Out-of-zone agents use simple planning. Agents near target use target-reaching policy.
    """

    # Checks out-of-zone.
    zone_flags = self.is_in_zone(x_cur)
    self.zone_flags = [_a or _b for _a, _b in zip(zone_flags, self.zone_flags)]
    if not all(self.zone_flags):
      _states, _controls = self.plan_stp(x_cur, init_control, self.targets)
    for ii in range(len(x_cur)):
      if not self.zone_flags[ii]:
        x_new[ii] = _states[ii][:, 1]
        us_cur[ii] = _controls[ii]

    # Checks target-reaching.
    target_flags = self.is_near_target(x_cur)
    for ii in range(len(x_cur)):
      if target_flags[ii]:
        _states, _controls, _, _, _ = self.tr_planner.solve(
            x_cur[ii], init_control[ii], [], self.targets[ii]
        )
        x_new[ii] = _states[:, 1]
        us_cur[ii] = _controls

    return x_new, us_cur, zone_flags

  def generate_init_states(self, centers: List[np.ndarray], ranges: np.ndarray, rng, N):
    """
    Generates initial conditions
    """
    num_agent = len(centers)
    nx = centers[0].shape[0]
    init_states = []
    for ii in range(num_agent):
      _l = ranges[:, np.newaxis]
      _init_s = centers[ii][:, np.newaxis] + rng.uniform(low=-_l, high=_l, size=(nx, N))
      init_states.append(_init_s)
    return init_states


class Struct:
  """
  Struct for managing parameters.
  """

  def __init__(self, data) -> None:
    for key, value in data.items():
      setattr(self, key, value)


def load_config(file_path):
  """
    Loads the config file.

    Args:
        file_path (string): path to the parameter file.

    Returns:
        Struct: parameters.
    """
  with open(file_path) as f:
    data = yaml.safe_load(f)
  config = Struct(data)
  return config


def plot_receding_horizon(
    states, state_hist, k, config, fig_prog_folder, colors, xlim, ylim, figsize=(15, 15),
    fontsize=50, plot_pred=False
):
  """
  Plot receding horizon planned trajectory.
  """
  num_agent = len(colors)
  plt.figure(figsize=figsize)
  ax = plt.gca()
  for ii in range(num_agent):
    _xii, _yii, _pii = states[ii][0, 0], states[ii][1, 0], states[ii][3, 0]
    ax.add_patch(Circle((_xii, _yii), config.WIDTH, color=colors[ii], fill=False))  # footprint
    _len = config.WIDTH * 1.2
    plt.arrow(
        _xii,
        _yii,
        _len * np.cos(_pii),
        _len * np.sin(_pii),
        width=0.02,
        color=colors[ii],
    )
    if plot_pred:
      plt.plot(states[ii][0, :], states[ii][1, :], linewidth=2, c="k", linestyle="--")  # prediction
    plt.scatter(
        state_hist[ii][0, :k + 1], state_hist[ii][1, :k + 1], s=80, c=state_hist[ii][2, :k + 1],
        cmap=cm.jet, vmin=config.V_MIN, vmax=config.V_MAX, edgecolor="none", marker="o"
    )  # trajectory history
    # cbar = plt.colorbar(sc)
    # if ii < num_agent - 1:
    #   cbar.remove()
  # cbar.set_label(r"velocity [m/s]")
  plt.axis("equal")
  plt.xlim(xlim)
  plt.ylim(ylim)
  plt.rcParams.update({"font.size": fontsize})
  # plt.title(str(int(Js[0])) + ' | ' + str(int(Js[1])) + ' | ' + str(int(Js[2])))
  plt.savefig(os.path.join(fig_prog_folder, str(k) + ".png"), dpi=50)
  plt.close()


def generate_rgb_values(n):
  rgb_list = []

  # Warm colors
  for i in range(n // 2):
    ratio = i / (n//2 - 1)
    rgb = [1.0, ratio, 0.0]  # Red to Yellow
    rgb_list.append(rgb)

  # Cool colors
  for i in range(n // 2):
    ratio = i / (n//2 - 1)
    rgb = [0.0, 1.0 - ratio, 1.0]  # Blue to Violet
    rgb_list.append(rgb)

  return rgb_list


def plot_trajectory(
    states, config, fig_prog_folder, colors, xlim, ylim, figsize=(15, 15), fontsize=50, linewidth=5,
    image=None, targets=None, orders=None, plot_arrow=True, zone: ATCZone = None, zone_flags=None
):
  """
  Plots the planned trajectory.
  """
  step = states[0].shape[1]
  num_agent = len(colors)

  if orders is not None:
    rgb_list = generate_rgb_values(num_agent)

  for k in range(step):

    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Plots the ATC Zone.
    if zone is not None:
      atc_circ = Circle((zone.center[0], zone.center[1]), zone.radius, color=[.7, .7, .7],
                        fill=False, linestyle="--", linewidth=linewidth)
      ax.add_patch(atc_circ)

    for ii in range(num_agent):

      # Plots targets.
      if targets is not None:
        tar_circ = Circle((targets[ii][0], targets[ii][1]), config.WIDTH, color=colors[ii],
                          fill=False, linestyle="-", linewidth=linewidth)
        ax.add_patch(tar_circ)

      # Plots agent footprints and headings.
      _xii, _yii, _pii = states[ii][0, k], states[ii][1, k], states[ii][3, k]
      fpt_circ = Circle((_xii, _yii), config.WIDTH, color=colors[ii], fill=False, linestyle="--",
                        linewidth=linewidth / 2.)
      ax.add_patch(fpt_circ)
      if plot_arrow:
        _len = config.WIDTH * 1.2
        plt.arrow(
            _xii, _yii, _len * np.cos(_pii), _len * np.sin(_pii), width=0.015, color=colors[ii]
        )

      # Plots trajectory history.
      if orders is not None:
        for tau in range(k):
          if zone_flags is not None and not zone_flags[tau][ii]:  # Outside ATC zone
            _color = [.7, .7, .7]
          else:
            try:
              _order_tau = orders[tau]
              _color = rgb_list[_order_tau.index(ii)]
            except:
              _color = [.7, .7, .7]
          plt.scatter(
              states[ii][0, tau], states[ii][1, tau], s=figsize[0] * 7.5, color=_color,
              edgecolor="none", marker="o"
          )
      else:
        plt.scatter(
            states[ii][0, :k], states[ii][1, :k], s=figsize[0] * 7.5, c=states[ii][2, :k],
            cmap=cm.jet, vmin=config.V_MIN, vmax=config.V_MAX, edgecolor="none", marker="o"
        )

      # Plots agent images.
      if image is not None:
        transform_data = (
            Affine2D().rotate_deg_around(*(_xii, _yii), _pii / np.pi * 180) + plt.gca().transData
        )
        plt.imshow(
            image, transform=transform_data, interpolation="none", origin="lower", extent=[
                _xii - config.WIDTH, _xii + config.WIDTH, _yii - config.WIDTH, _yii + config.WIDTH
            ], alpha=1.0, zorder=10.0, clip_on=True
        )

    # Figure setup.
    plt.axis("equal")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.axis('off')
    plt.rcParams.update({"font.size": fontsize})
    plt.savefig(os.path.join(fig_prog_folder, str(k) + ".png"), dpi=50)
    plt.close()


def make_animation(steps, config, fig_prog_folder, name="rollout.gif"):
  gif_path = os.path.join(config.OUT_FOLDER, name)
  with imageio.get_writer(gif_path, mode="I", loop=0) as writer:
    for j in range(1, steps):
      filename = os.path.join(fig_prog_folder, str(j) + ".png")
      image = imageio.imread(filename)
      writer.append_data(image)
  return Image(open(gif_path, "rb").read())
