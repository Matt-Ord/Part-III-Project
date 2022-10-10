from abc import ABC, abstractclassmethod
import numpy as np
import scipy.constants, scipy.integrate
from properties.MaterialProperties import NICKEL_MATERIAL_PROPERTIES

# Used to test the q independance approximation
class GammaCalculator(ABC):

    a0 = (
        4
        * np.pi
        * scipy.constants.epsilon_0
        * scipy.constants.hbar ** 2
        / (scipy.constants.m_e * scipy.constants.elementary_charge ** 2)
    )

    3.79 * 10 ** 10
    omega = (
        NICKEL_MATERIAL_PROPERTIES.hydrogen_energies[0]
        - NICKEL_MATERIAL_PROPERTIES.hydrogen_energies[1]
    )
    k_f = NICKEL_MATERIAL_PROPERTIES.fermi_wavevector

    @staticmethod
    def get_delta_e(k_1, k_2):
        E1 = (scipy.constants.hbar * k_1) ** 2 / (2 * scipy.constants.m_e)
        E2 = (scipy.constants.hbar * k_2) ** 2 / (2 * scipy.constants.m_e)
        return E1 - E2

    @staticmethod
    def get_abs_q(k_1, k_2, cos_theta):
        q = np.sqrt(k_1 ** 2 + k_2 ** 2 - 2 * k_1 * k_2 * cos_theta)
        return q

    @classmethod
    def get_fermi_occupation(cls, k, boltzmann_energy):
        return 1 / (1 + np.exp((cls.get_delta_e(k, cls.k_f)) / boltzmann_energy))

    @abstractclassmethod
    def _integrate_gamma(k_start, k_end, boltzmann_energy) -> tuple[float, float]:
        return (0, 0)

    @classmethod
    def calculate_gamma(cls, temperature):
        boltzmann_energy = scipy.constants.Boltzmann * temperature
        d_k = (
            2
            * boltzmann_energy
            * scipy.constants.m_e
            / (scipy.constants.hbar ** 2 * cls.k_f)
        )
        integral_width = 20 * d_k
        return cls._integrate_gamma(
            cls.k_f - integral_width, cls.k_f + integral_width, boltzmann_energy
        )

    @classmethod
    def _get_interaction_potential(cls, abs_q):
        x = (abs_q * cls.a0 / 2) ** 2
        return -(2 + x) / (1 + x) ** 2

    @classmethod
    def _get_overlap_potential(cls, q_x, q_y, q_z):
        pass

    @classmethod
    def _get_potential(cls, k_1, k_3, cos_theta, phi):
        return cls._get_interaction_potential(cls.get_abs_q(k_1, k_3, cos_theta))

    @classmethod
    def _integrand(cls, k_1, cos_theta, phi, boltzmann_energy):
        k_3 = np.sqrt(
            k_1 ** 2 - 2 * scipy.constants.m_e * cls.omega / (scipy.constants.hbar ** 2)
        )

        return (
            (k_1 ** 2)
            * (k_3 ** 2)
            * cls._get_potential(k_1, k_3, cos_theta, phi) ** 2
            * cls.get_fermi_occupation(k_1, boltzmann_energy)
            * (1 - cls.get_fermi_occupation(k_3, boltzmann_energy))
            / (
                np.sqrt(
                    k_1 ** 2
                    - 2 * scipy.constants.m_e * cls.omega / (scipy.constants.hbar ** 2)
                )
            )
        )


class PhiDependantGammaCalculator(GammaCalculator):
    @classmethod
    def _integrate_gamma(cls, k_start, k_end, boltzmann_energy):
        integrand = lambda phi, cos_theta, k_1: cls._integrand(
            k_1, cos_theta, phi, boltzmann_energy
        )
        return scipy.integrate.tplquad(integrand, k_start, k_end, -1, 1, 0, 2 * np.pi)


class ThetaDependantGammaCalculator(GammaCalculator):
    @classmethod
    def _integrate_gamma(cls, k_start, k_end, boltzmann_energy):
        integrand = (
            lambda cos_theta, k_1: 2
            * np.pi
            * cls._integrand(k_1, cos_theta, 0, boltzmann_energy)
        )
        return scipy.integrate.dblquad(integrand, k_start, k_end, -1, 1)


class QIndependantGammaCalculator(GammaCalculator):
    @classmethod
    def _integrate_gamma(cls, k_start, k_end, boltzmann_energy):
        integrand = lambda k_1: 4 * np.pi * cls._integrand(k_1, 0, 0, boltzmann_energy)
        return scipy.integrate.quad(integrand, k_start, k_end)

    @staticmethod
    def get_abs_q(k_1, k_2, cos_theta):
        return 0


def calculate_fraction_error(temperature):
    q_indep, err = QIndependantGammaCalculator.calculate_gamma(temperature)
    q_dep, err = ThetaDependantGammaCalculator.calculate_gamma(temperature)
    print(q_indep, q_dep)
    return (q_dep - q_indep) / q_dep


if __name__ == "__main__":
    print(GammaCalculator.a0)
    print(GammaCalculator._get_interaction_potential(0.0000001 * GammaCalculator.k_f))
    print(calculate_fraction_error(150))
    print(calculate_fraction_error(100))