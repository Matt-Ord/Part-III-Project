from FullLindbaldSolver import FullLindbaldSolver
import numpy as np


class NoEnergyGapLindbaldSolver(FullLindbaldSolver):

    _max_step = 1 * 10 ** (-12)

    def _get_gamma_energy_factor(self, a, b):
        return 1
