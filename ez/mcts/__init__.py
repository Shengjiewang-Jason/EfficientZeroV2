# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

from .py_mcts import PyMCTS
from .cy_mcts import CyMCTS



names = {
    'python': PyMCTS,
    'cython': CyMCTS
}