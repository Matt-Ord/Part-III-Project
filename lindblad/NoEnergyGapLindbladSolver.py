from FullLindbladSolver import FullLindbladSolver
import numpy as np


class NoEnergyGapLindbladSolver(FullLindbladSolver):

    _max_step = 1 * 10 ** (-12)

    def _get_gamma_energy_factor(self, a, b):
        return 1
