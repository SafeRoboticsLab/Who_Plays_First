"""
Logger functions for Branch and Play (B&P).

Please contact the author(s) of this library if you have any questions.
Authors: Gabriele Dragotto (hello@dragotto.net), Haimin Hu (haiminh@princeton.edu)
"""

import logging
import sys
import numpy as np


class CustomFormatter(logging.Formatter):
  grey = "\x1b[1;36m"
  yellow = "\x1b[33;20m"
  red = "\x1b[31;20m"
  bold_red = "\x1b[31;1m"
  reset = "\x1b[0m"
  format = ("%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)")

  FORMATS = {
      logging.DEBUG: grey + format + reset,
      logging.INFO: grey + format + reset,
      logging.WARNING: yellow + format + reset,
      logging.ERROR: red + format + reset,
      logging.CRITICAL: bold_red + format + reset,
  }

  def format(self, record):
    log_fmt = self.FORMATS.get(record.levelno)
    formatter = logging.Formatter(log_fmt)
    return formatter.format(record)


logger = logging.getLogger("wpf")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)
logger.propagate = False


def print_log_head():
  print("🎮👾🎯🎰🎲🎳🃏🎱⚽️🎮👾🎯🎰🎲🎳🃏🎱⚽️ Who Plays First? 🎮👾🎯🎰🎲🎳🃏🎱⚽️🎮👾🎯🎰🎲🎳🃏🎱⚽️")
  print(("{:<7}\t{:<7}\t{:<7}\t{:<7}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<7}").format(
      "Node",
      "Time",
      "Pruned",
      "Active",
      "NodeBnd",
      "LowerBnd",
      "UpperBnd",
      "Gap",
      "Feasible",
      "Permutation",
  ))


def print_iteration(node, time, fathomed, active, bound, best_lb, best_ub, feasible, permutation):
  _spec = "{:<7.0f}\t{:<7.3f}\t{:<7.0f}\t{:<7.0f}\t{:<10.2f}\t{:<10.2f}\t{:<10.2f}\t{:<10.4f}\t{:<10}\t{:<10}"
  print(
      _spec.format(
          node,
          time,
          fathomed,
          active,
          bound,
          best_lb,
          best_ub,
          np.inf if
          (best_ub == np.inf or best_lb == -np.inf) else abs(best_ub - best_lb) / abs(best_lb),
          "✅" if feasible else "❌",
          str(permutation),
      )
  )
