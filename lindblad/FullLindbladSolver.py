from LindbladSolver import LindbladSolver
import numpy as np


class FullLindbladSolver(LindbladSolver):

    _max_step = np.inf

    def _get_phase_factor(self, t, a, b, c, d):
        return np.exp(1j * (self._get_omega_ab(a, b) - self._get_omega_ab(c, d)) * t)

    def _calculate_derivatie_of_pmn(self, t, probabilities, m, n):
        return sum(
            self._get_phase_factor(-t, n, j, m, i)
            * self._get_gamma_abcd_omega_ij(n, j, m, i, m, i)
            * probabilities[i][j]
            - self._get_phase_factor(-t, i, m, i, j)
            * self._get_gamma_abcd_omega_ij(i, m, i, j, i, j)
            * probabilities[j][n]
            + self._get_phase_factor(t, n, j, m, i)
            * self._get_gamma_abcd_omega_ij(n, j, m, i, n, j)
            * probabilities[i][j]
            - self._get_phase_factor(t, i, j, i, n)
            * self._get_gamma_abcd_omega_ij(i, j, i, n, i, j)
            * probabilities[m][j]
            for i in [0, 1]
            for j in [1, 0]
        )

    def _calculate_derivatie_of_p00(self, t, p00, p01, p10, p11):
        probabilities = [[p00, p01], [p10, p11]]
        return self._calculate_derivatie_of_pmn(t, probabilities, 0, 0)

    def _calculate_derivatie_of_p01(self, t, p00, p01, p10, p11):
        probabilities = [[p00, p01], [p10, p11]]
        return self._calculate_derivatie_of_pmn(t, probabilities, 0, 1)

    def _calculate_derivatie_of_p10(self, t, p00, p01, p10, p11):
        probabilities = [[p00, p01], [p10, p11]]
        return self._calculate_derivatie_of_pmn(t, probabilities, 1, 0)

    def _calculate_derivatie_of_p11(self, t, p00, p01, p10, p11):
        probabilities = [[p00, p01], [p10, p11]]
        return self._calculate_derivatie_of_pmn(t, probabilities, 1, 1)
