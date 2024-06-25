"""
Main functions of Branch and Play (B&P).

Please contact the author(s) of this library if you have any questions.
Authors: Gabriele Dragotto (hello@dragotto.net), Haimin Hu (haiminh@princeton.edu)
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Any

from .io import print_log_head, print_iteration


@dataclass
class Node(object):
  """
  Branch-and-Play node.
  """

  permutation: np.ndarray
  depth: int
  # Note that the solution is initialized to the one of the parent node
  solution: np.ndarray
  lb: float
  ub: float
  custom_data: Any = None  # custom data for each node


@dataclass
class Settings(object):
  """
  Branch-and-Play settings.
  """

  max_nodes: int = 1e20
  feas_tolerance: float = 1e-5
  min_gap: float = 0
  max_iter: int = 1e20
  random_heuristic_iterations: int = 1
  verbose: bool = True
  custom_settings: Any = None  # custom settings


@dataclass
class Statistics(object):
  """
  Branch-and-Play statistics.
  """

  explored_nodes = 0
  fathomed_nodes = 0
  num_feasible_solutions = 0
  time = 0.0
  global_ub = np.inf
  global_ub_history = np.ndarray
  global_lb = np.inf
  global_lb_history = np.ndarray
  gap_history = np.ndarray
  explored_permutations = np.ndarray
  incumbent = np.ndarray
  incumbent_permutation = np.ndarray
  custom_statistics: Any = None  # custom statistics


class BranchAndPlay(object):
  """
  Branch-and-Play tree.
  """

  def __init__(
      self, n, instance_data, exploration_cb, solver_cb, fathoming_cb, branching_cb, settings,
      initializer=None
  ):
    """
    Initializes the fields of the Branch and Play.
    :param n: Number of players
    :param instance_data: The instance data (passed to the solver_cb method)
    :param exploration_cb: Callback to select the next node to explore (arguments: vector of nodes and stats; returns the index of the selected node)
    :param solver_cb: Callback to solve the node (arguments: instance_data, node, settings and stats; writes in-place in node (no return))
    :param fathoming_cb: Callback to fathom nodes (arguments: vector of nodes, upper bound, incumbent and incumbent permutation; write in-place in nodes)
    :param branching_cb: Callback to create children nodes (arguments: vector of nodes and current node, n; write in-place in nodes)
    :param settings: Settings object with the solver settings
    :param initialzier: An optional function that will take, as arguments, the instance data, the nodes, the statistics and the settings. It
    writes in place in these objects
    :return The object Statistics (which includes the solution)
    """
    self.n = n
    self.data = instance_data
    self.exploration = exploration_cb
    self.solver = solver_cb
    self.fathoming = fathoming_cb
    self.branching = branching_cb
    self.settings = settings
    self.initializer = initializer

    self.update_problem(instance_data)

  @property
  def results(self):
    return self._stats

  def update_problem(self, instance_data, mode="init"):
    """
    Updates the problem data.

    mode = "init" or "update"
    """
    self.nodes = []
    self._stats = Statistics()
    self.create_root_node()

    if self.initializer is not None:
      self.initializer(instance_data, self.nodes, self._stats, self.settings, mode)

    # Store intermediate results
    self._stats.iter = 0
    self._stats.global_lb_history = []
    self._stats.global_ub_history = []
    self._stats.gap_history = []
    self._stats.explored_permutations = []

  def create_root_node(self):
    """
    Creates the root node for the tree.
    """
    self.nodes.append(
        Node(np.array([None for _ in range(self.n)]), 0, None, -np.inf, +np.inf, None)
    )

  def can_continue(self):
    """
    Determines whether the exploration can continue or not.
    """
    if len(self.nodes) == 0:
      return False

    if self._stats.explored_nodes > self.settings.max_nodes:
      return False

    if self._stats.iter > self.settings.max_iter:
      return False

    gap = np.inf
    if (abs(self._stats.global_ub) != np.inf and abs(self._stats.global_lb) != np.inf):
      gap = abs(self._stats.global_ub - self._stats.global_lb) / abs(self._stats.global_lb)
    self._stats.gap_history.append(gap)
    if gap <= self.settings.min_gap:
      print("Minimum gap reached. Terminating exploration with gap:", gap)
      return False

    return True

  def prune(self):
    """
    Performs pruning via bound and user-specified fathoming rule.
    """
    if self._stats.global_ub != np.inf:
      init_len = len(self.nodes)
      self.nodes = [
          node for node in self.nodes
          if self._stats.global_ub >= node.lb + self.settings.feas_tolerance
      ]
      self._stats.fathomed_nodes += init_len - len(self.nodes)
    # Custom fathoming rule
    self.fathoming(
        self.nodes,
        self._stats.global_ub,
        self._stats.incumbent,
        self._stats.incumbent_permutation,
    )

  def solve_node(self, node):
    """
    Solves the selected node and updates the statistics.
    :param node: Node to solve
    """
    prev_lb = lb = node.lb
    self.solver(self.data, node, self.settings, self._stats)
    feasible = False
    lb = node.lb

    assert lb >= prev_lb, "Lower bound cannot improve"

    # Check if node contains a feasible solution
    if lb != np.inf:
      if None not in node.permutation:
        # The permutation is complete (i.e., no missing elements)
        # Update the solution and statistics
        self._stats.num_feasible_solutions += 1
        feasible = True

        assert node.ub >= lb, "Upper bound cannot be lower than lower bound"

        if self._stats.global_ub > node.ub:
          self._stats.incumbent_permutation = node.permutation
          self._stats.incumbent = node.solution
          # Update upper bound
          self._stats.global_ub = node.ub
      else:
        self.branching(self.nodes, node, self.n, self._stats, self.settings)

    self._stats.explored_permutations.append(node.permutation.tolist())
    return feasible, lb

  def init_perm_heuristic(self, verbose=True):
    """
    Proposes an initial feasible permutation.
    """
    if (
        hasattr(self.settings, 'custom_settings')
        and self.settings.custom_settings["enable_custom_heuristic"]
    ):
      # Picks a permutation that prioritizes players with higher costs
      sorted_Js = sorted(
          enumerate(self.settings.custom_settings["Js_prev"]), key=lambda x: x[1], reverse=True
      )
      permutation = [index for index, _ in sorted_Js]
      if verbose:
        print("\tCustom heuristic is exploring permutation", permutation)
    else:
      # Randomly pick a permutation
      for _ in range(self.settings.random_heuristic_iterations):
        permutation = np.random.permutation(self.n)
        # Create the node
        node = Node(permutation, 0, None, self.nodes[0].lb, -np.inf, None)
        # Solve the node
        feasible, lb = self.solve_node(node)
        if verbose:
          print("\tRandom heuristic is exploring permutation", permutation)

  def update_lb(self):
    """
    Updates the lower bound in the statistics.
    """
    minimum = np.inf
    for node in self.nodes:
      minimum = min(node.lb, minimum)
    self._stats.global_lb = minimum

  def solve(self):
    """
    Main method for the user to solve the problem via Branch-and-Play.
    """
    verbose = self.settings.verbose

    # Start the timer
    start = time.time()
    if verbose:
      print_log_head()

    self.init_perm_heuristic(verbose)

    while self.can_continue():
      # Pick the next node
      index = self.exploration(self.nodes, self._stats)
      node = self.nodes[index]

      # Solve the node
      feasible, lb = self.solve_node(node)

      # Remove the node
      del self.nodes[index]

      # Fathom and prune nodes
      self.update_lb()
      self.prune()

      # Print iteration
      if verbose:
        print_iteration(
            self._stats.explored_nodes,
            time.time() - start, self._stats.fathomed_nodes, len(self.nodes), lb,
            self._stats.global_lb, self._stats.global_ub, feasible, node.permutation
        )

      # Store intermediate data for later plots
      if self._stats.global_ub == np.inf:
        self._stats.global_ub_history.append(None)
        self._stats.gap_history.append(None)
      else:
        self._stats.global_ub_history.append(self._stats.global_ub)
      self._stats.global_ub_history.append(self._stats.global_lb)

      # Update stats
      self._stats.explored_nodes += 1
      self._stats.iter += 1

    self._stats.time = time.time() - start


def bestfirst_cb(nodes, stats):
  """
  Defines the best-first exploration strategy.
  :param nodes: List of nodes
  :param stats: The statistics object
  :return The selected node index
  """
  best_index = 0
  best_bound = np.inf
  for index, node in enumerate(nodes):
    if node.lb < best_bound:
      best_index = index
      best_bound = node.lb
  return best_index


def depthfirst_cb(nodes, stats):
  """
  Defines the dept-first exploration strategy.
  :param nodes: List of nodes
  :param stats: The statistics object
  :return The selected node index
  """
  return len(nodes) - 1


def branchall_cb(nodes, node, n, stats, settings):
  """
  Defines the simple branching strategy of creating n children from the first unknown player.
  Note that this method performs an in-place operation on nodes.
  :param nodes: List of nodes
  :param node: Incumbent node
  :param stats: Statistics
  :param settings: Settings
  :param n: Number of players
  """

  undefined_locations = np.where(node.permutation == None)[0]
  if len(undefined_locations) > 0:
    location = undefined_locations[0]
    undefined = []

    for player in range(n):
      if player not in node.permutation:
        undefined.append(player)

    for index, player in enumerate(undefined):
      child_permutation = np.copy(node.permutation)
      child_permutation[location] = player

      undefined_locations = np.where(child_permutation == None)[0]
      # If only one is missing
      if len(undefined_locations) == 1:
        child_permutation[undefined_locations[0]] = undefined[(index+1) % len(undefined)]
      if child_permutation.tolist() not in stats.explored_permutations:
        nodes.append(
            Node(
                child_permutation,
                node.depth + 1,
                node.solution,
                node.lb,
                -np.inf,
                node.custom_data,
            )
        )
