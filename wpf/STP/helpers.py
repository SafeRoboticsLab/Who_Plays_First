import numpy as np
from ..bnp import Node
from .stp import *
import itertools


def stp_initializer(instance_data, nodes, stats, settings, mode="init"):
  """
  Initializes the BnP for the STP algorithm.
  In particular, it sets some custom settings, statistics, and node data

  mode = "init" or "update"
  """
  if mode == "init":
    x_cur, init_control, targets, config = instance_data
    settings.custom_settings = {
        "solver": STP(config),
        "RHC": config.RHC,
        "init_parent": config.INIT_WITH_PARENT,
        "jax_compile_threshold": 0.5,
        "x_root": x_cur,
        "control_root": init_control,
        "targets_root": targets,
        "enable_custom_heuristic": False,
    }
  elif mode == "update":
    x_cur, init_control, targets, Js_prev, config = instance_data
    settings.custom_settings["x_root"] = x_cur
    settings.custom_settings["control_root"] = init_control
    settings.custom_settings["targets_root"] = targets
    settings.custom_settings["Js_prev"] = Js_prev
    settings.custom_settings["enable_custom_heuristic"] = True
  stats.custom_statistics = {"jax_compile_time": [], "Js_prev": []}
  nodes[0].custom_data = {"collision_matrix": None}


def stp_solve_cb(data, node: Node, settings, stats):
  """
  This method solves the local node of the STP
  """
  # First, try to warmstart with the parent node's solution
  if (node.solution is None) or (not settings.custom_settings["init_parent"]):
    ctrls_ws = settings.custom_settings["control_root"]
  else:
    ctrls_ws = node.solution[1]

  # Solves STP.
  if settings.custom_settings["RHC"]:
    states, controls, Js, ts = settings.custom_settings["solver"].solve_rhc(
        settings.custom_settings["x_root"],
        ctrls_ws,
        settings.custom_settings["targets_root"],
        node.permutation,
    )
  else:
    states, controls, Js, ts = settings.custom_settings["solver"].solve(
        settings.custom_settings["x_root"],
        ctrls_ws,
        settings.custom_settings["targets_root"],
        node.permutation,
    )

  _total_time = sum(ts)
  if _total_time < settings.custom_settings["jax_compile_threshold"]:
    stats.custom_statistics["jax_compile_time"].append(_total_time)

  node.custom_data = {
      "collision_matrix":
          np.array(
              settings.custom_settings["solver"].pairwise_collision_check(
                  np.array(np.stack(states, axis=2))
              )
          )
  }

  node.solution = (states, controls, ts)
  node.lb = max(np.sum(Js), node.lb)
  node.Js = Js
  if None not in node.permutation:
    node.ub = node.lb


def stp_branching(nodes, node, n, stats, settings):
  """
  Defines the custom branching strategy for STP.
  Note that this method performs an in-place operation on nodes
  :param nodes: List of nodes
  :param node: Incumbent node
  :param n: Number of players
  """
  undefined_locations = np.where(node.permutation == None)[0]
  if len(undefined_locations) > 0:
    location = undefined_locations[0]
    undefined = []

    for player in range(n):
      if player not in node.permutation:
        undefined.append(player)

    no_collisions = True
    for player1 in undefined:
      for player2 in undefined:
        if player1 != player2:
          if node.custom_data["collision_matrix"][player1, player2]:
            no_collisions = False
            break

    # If no collisions, arbitrary order is fine
    if no_collisions:
      # This is a feasible solution
      undefined_completions = itertools.permutations(undefined)

      for index, completion in enumerate(undefined_completions):
        child_permutation = np.copy(node.permutation)
        # Fill remaining spots with undefined players
        undefined_locations = np.where(node.permutation == None)[0]
        for i in range(len(undefined)):
          child_permutation[undefined_locations[i]] = completion[i]
        if index == 0:
          if abs(node.lb) != np.inf:
            # Feasible solution
            stats.num_feasible_solutions += 1
            node.ub = node.lb
            if stats.global_ub > node.ub:
              stats.incumbent_permutation = child_permutation
              stats.incumbent = node.solution
              stats.custom_statistics["Js_prev"] = node.Js
              # Update upper bound
              stats.global_ub = node.ub
            if settings.verbose:
              print(
                  "\tNo collisions. Found feasible solution. Pruned symmetrics:",
                  len(list(undefined_completions)) - 1,
              )
          else:
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
            if settings.verbose:
              print(
                  "\tNo collisions. Pruned symmetrics:",
                  len(list(undefined_completions)) - 1,
              )
        else:
          stats.explored_permutations.append(child_permutation.tolist())
      # We are done
      return

    else:
      # We have to branch on all possible permutations
      skipped = 0
      for index, player in enumerate(undefined):
        child_permutation = np.copy(node.permutation)
        child_permutation[location] = player

        undefined_locations = np.where(child_permutation == None)[0]

        # If only one is missing
        if len(undefined_locations) == 1:
          child_permutation[undefined_locations[0]] = undefined[(index+1) % len(undefined)]
        child_permutation_list = child_permutation.tolist()
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
        # Handle symmetry
        # For players that do not collide
        # print("\tstarting parent", node.permutation)
        # print("\tstarting child", child_permutation_list)
        for player1, player2 in zip(*np.nonzero(node.custom_data["collision_matrix"] - 1)):
          if player1 != player2:
            # Check for congtiguos players
            l_1 = np.where(child_permutation == player1)[0]
            l_2 = np.where(child_permutation == player2)[0]
            # If the two players are in the child permutation
            if len(l_1) == 1 and len(l_2) == 1:
              l_1 = l_1[0]
              l_2 = l_2[0]
              # If their location is adjacent and were undefined in parent
              if (
                  abs(l_1 - l_2) == 1 and l_1 in undefined_locations and l_2 in undefined_locations
              ):
                # print("removed something")
                child_copy = child_permutation_list.copy()
                child_copy[l_1] = player2
                child_copy[l_2] = player1
                skipped += 1
                # print("\t\tplayers", player1, player2)
                # print("\t\tskipping", child_copy)
                stats.explored_permutations.append(child_copy)
      if skipped > 0:
        print("\tPruned symmetric nodes:", skipped)
