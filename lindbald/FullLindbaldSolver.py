from LindbaldSolver import LindbaldSolver
import numpy as np


class FullLindbaldSolver(LindbaldSolver):

    _max_step = 1 * 10 ** (-12)

    def _calculate_cross_derivative_factor_for_paa(self, t, a):
        b = int(not a)
        term1 = np.exp(1j * self._get_omega_ab(b, a) * t) * self._get_gamma_abcd(
            a, a, a, b
        )
        term2 = np.exp(1j * self._get_omega_ab(a, b) * t) * self._get_gamma_abcd(
            b, b, b, a
        )
        return term1 - term2

    def _calculate_cross_derivative_factor_for_p00(self, t):
        return self._calculate_cross_derivative_factor_for_paa(t, 0)

    def _calculate_cross_derivative_factor_for_p11(self, t):
        return self._calculate_cross_derivative_factor_for_paa(t, 1)

    def _calculate_derivatie_of_p00(self, t, p00, p01, p10, p11):
        p00_term = -p00 * 2 * self._get_gamma_abcd(1, 0, 1, 0)
        p01_term = +p01 * self._calculate_cross_derivative_factor_for_p00(t)
        p10_term = +p10 * self._calculate_cross_derivative_factor_for_p00(t)
        p11_term = +p11 * 2 * self._get_gamma_abcd(0, 1, 0, 1)
        return p00_term + p01_term + p10_term + p11_term

    def _calculate_derivatie_of_p11(self, t, p00, p01, p10, p11):
        p00_term = +p00 * 2 * self._get_gamma_abcd(1, 0, 1, 0)
        p01_term = +p01 * self._calculate_cross_derivative_factor_for_p11(t)
        p10_term = +p10 * self._calculate_cross_derivative_factor_for_p11(t)
        p11_term = -p11 * 2 * self._get_gamma_abcd(0, 1, 0, 1)
        return p00_term + p01_term + p10_term + p11_term

    def _calculate_paa_derivative_factor_for_pab(self, t, a):
        b = int(not a)
        term_1 = np.exp(1j * self._get_omega_ab(a, b) * t) * self._get_gamma_abcd(
            a, a, b, a
        )
        term_2 = np.exp(1j * self._get_omega_ab(b, a) * t) * self._get_gamma_abcd(
            a, a, a, b
        )
        return term_1 - term_2

    def _calculate_pbb_derivative_factor_for_pab(self, t, a):
        b = int(not a)
        term_1 = np.exp(1j * self._get_omega_ab(b, a) * t) * self._get_gamma_abcd(
            b, b, a, b
        )
        term_2 = np.exp(1j * self._get_omega_ab(a, b) * t) * self._get_gamma_abcd(
            b, b, b, a
        )
        return term_1 - term_2

    def _calculate_pab_derivative_factor_for_pab(self, t, a):
        b = int(not a)
        term_1 = self._get_gamma_abcd(a, b, a, b)
        term_2 = self._get_gamma_abcd(b, a, b, a)
        return -term_1 - term_2

    def _calculate_pba_derivative_factor_for_pab(self, t, a):
        b = int(not a)
        term_1 = np.exp(2j * self._get_omega_ab(b, a) * t) * self._get_gamma_abcd(
            b, a, a, b
        )
        term_2 = np.exp(2j * self._get_omega_ab(a, b) * t) * self._get_gamma_abcd(
            a, b, b, a
        )
        return term_1 - term_2

    def _calculate_derivatie_of_pab(self, t, a, paa, pab, pba, pbb):
        paa_term = paa * self._calculate_paa_derivative_factor_for_pab(t, a)
        pab_term = pab * self._calculate_pab_derivative_factor_for_pab(t, a)
        pba_term = pba * self._calculate_pba_derivative_factor_for_pab(t, a)
        pbb_term = pbb * self._calculate_pbb_derivative_factor_for_pab(t, a)
        return paa_term + pab_term + pba_term + pbb_term

    def _calculate_derivatie_of_p01(self, t, p00, p01, p10, p11):
        return self._calculate_derivatie_of_pab(t, 0, p00, p01, p10, p11)

    def _calculate_derivatie_of_p10(self, t, p00, p01, p10, p11):
        return self._calculate_derivatie_of_pab(t, 1, p11, p10, p01, p00)
