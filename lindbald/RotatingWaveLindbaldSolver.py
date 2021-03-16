from LindbaldSolver import LindbaldSolver


class RotatingWaveLindbaldSolver(LindbaldSolver):
    def _calculate_derivatie_of_p00(self, t, p00, p01, p10, p11):
        p00_term = -p00 * 2 * self._get_gamma_abcd(1, 0, 1, 0)
        p01_term = 0
        p10_term = 0
        p11_term = +p11 * 2 * self._get_gamma_abcd(0, 1, 0, 1)
        return p00_term + p01_term + p10_term + p11_term

    def _calculate_derivatie_of_p11(self, t, p00, p01, p10, p11):
        p00_term = +p00 * 2 * self._get_gamma_abcd(1, 0, 1, 0)
        p01_term = 0
        p10_term = 0
        p11_term = -p11 * 2 * self._get_gamma_abcd(0, 1, 0, 1)
        return p00_term + p01_term + p10_term + p11_term

    def _calculate_derivatie_of_p01(self, t, p00, p01, p10, p11):
        return 0

    def _calculate_derivatie_of_p10(self, t, p00, p01, p10, p11):
        return 0
