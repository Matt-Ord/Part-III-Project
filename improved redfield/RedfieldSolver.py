from abc import ABC, abstractmethod
from matplotlib.pyplot import sci
from numpy.linalg import solve
import scipy.integrate
import scipy.constants
from properties.MaterialProperties import NICKEL_MATERIAL_PROPERTIES
import numpy as np


class RedfieldSolver(ABC):
    def __init__(self, times, temperature, initial_state=[1, 0, 0, 0]) -> None:
        self.times = times
        self.temperature = temperature
        self.initial_state = [complex(x) for x in initial_state]
        print(self.initial_state)

    _soln = None
    _max_step = 1 * 10 ** (-14)

    @property
    def p00_values(self):
        if self._soln is None:
            self._solve_redfield_equation()
        return self._soln["y"][0]

    @property
    def p01_values(self):
        if self._soln is None:
            self._solve_redfield_equation()
        return self._soln["y"][1]

    @property
    def p10_values(self):
        if self._soln is None:
            self._solve_redfield_equation()
        return self._soln["y"][2]

    @property
    def p11_values(self):
        if self._soln is None:
            self._solve_redfield_equation()
        return self._soln["y"][3]

    @property
    def boltzmaan_energy(self):
        return scipy.constants.Boltzmann * self.temperature

    @property
    def fermi_wavevector(self):
        return NICKEL_MATERIAL_PROPERTIES.fermi_wavevector

    def calculate_electron_energy(self, wavevector):
        return (scipy.constants.hbar * wavevector) ** 2 / (2 * scipy.constants.m_e)

    def calculate_hydrogen_energy(self, index):
        return NICKEL_MATERIAL_PROPERTIES.hydrogen_energies[index]

    def calculate_electron_delta_e(self, q1, q2):
        return self.calculate_electron_energy(q1) - self.calculate_electron_energy(q2)

    def calculate_hydrogen_delta_e(self, a, b):
        return self.calculate_hydrogen_energy(a) - self.calculate_hydrogen_energy(b)

    def _calculate_phase_term(self, time, i, j):
        return np.exp(
            1j * time * (self.calculate_hydrogen_delta_e(i, j)) / scipy.constants.hbar
        )

    def calculate_boltzmann_term(self, a, b):
        return np.exp(
            -self.calculate_hydrogen_delta_e(a, b) / (2 * self.boltzmaan_energy)
        )

    def calculate_single_boltzmann_term(self, a):
        return np.exp(-self.calculate_hydrogen_energy(a) / (2 * self.boltzmaan_energy))

    def calculate_constant_potential_term(self):
        return (
            -8
            * np.pi ** 2
            * scipy.constants.epsilon_0
            * scipy.constants.hbar ** 4
            / (scipy.constants.m_e ** 2 * scipy.constants.elementary_charge ** 2)
        )

    def calculate_potential_term(self, i, j):
        return (
            self.calculate_constant_potential_term()
            * NICKEL_MATERIAL_PROPERTIES.hydrogen_overlaps[i][j]
        )

    def calculate_first_order_derivative_prefactor(self, time, i, j):
        return (
            -1j
            * (self.fermi_wavevector ** 3)
            / (12 * np.pi ** (1.5) * scipy.constants.hbar)
            # * self._calculate_phase_term(time, i, j)
        )

    def calculate_second_order_diagonal_derivative_prefactor(self, time, i, j):
        return -(
            scipy.constants.electron_mass ** 2
            * self.boltzmaan_energy
            * (self.fermi_wavevector ** 2)
            / (8 * (np.pi ** (3.5)) * (scipy.constants.hbar ** 5))
        )

    def calculate_second_order_cross_derivative_prefactor(self, time, i, j):
        return -(
            scipy.constants.electron_mass
            * (self.fermi_wavevector ** 6)
            * np.exp(-1 / 4)
            * (np.pi) ** (0.5)
            * self._calculate_phase_term(time, i, j)
            / (4 * (2 * np.pi) ** (4) * (scipy.constants.hbar ** 3))
        )

    def _calculate_second_order_diagonal_derivative(self, time, i, j, probabilities):
        return (
            sum(
                self.calculate_potential_term(i, m)
                * self.calculate_potential_term(m, j)
                * probabilities[j][j]
                * self._calculate_phase_term(time, i, m)
                * self._calculate_phase_term(time, j, m)
                * self.calculate_boltzmann_term(m, j)
                for m in [0, 1]
            )
            - sum(
                probabilities[i][i]
                * self.calculate_potential_term(i, m)
                * self.calculate_potential_term(m, j)
                * self._calculate_phase_term(time, m, i)
                * self._calculate_phase_term(time, m, j)
                * self.calculate_boltzmann_term(m, i)
                for m in [0, 1]
            )
            - sum(
                self.calculate_potential_term(i, m)
                * probabilities[m][m]
                * self.calculate_potential_term(m, j)
                * self._calculate_phase_term(time, i, m)
                * self._calculate_phase_term(time, j, m)
                * self.calculate_boltzmann_term(m, j)
                for m in [0, 1]
            )
            + sum(
                self.calculate_potential_term(i, m)
                * probabilities[m][m]
                * self.calculate_potential_term(m, j)
                * self._calculate_phase_term(time, m, i)
                * self._calculate_phase_term(time, m, j)
                * self.calculate_boltzmann_term(m, i)
                for m in [0, 1]
            )
        )

    def _calculate_second_order_cross_derivative(self, time, i, j, probabilities):
        # print(
        #     self.calculate_potential_term(i, 1),
        #     self.calculate_second_order_cross_derivative_prefactor(time, i, j),
        # )

        return (
            sum(
                self.calculate_potential_term(i, m)
                * self.calculate_potential_term(m, l)
                * probabilities[l][j]
                * (
                    self.calculate_boltzmann_term(j, l)
                    * (
                        self.calculate_boltzmann_term(m, l)
                        - self.calculate_boltzmann_term(m, j)
                    )
                )
                for m in [0, 1]
                for l in [0, 1]
            )
            - sum(
                probabilities[i][m]
                * self.calculate_potential_term(m, l)
                * self.calculate_potential_term(l, j)
                * (
                    self.calculate_boltzmann_term(m, i)
                    * (
                        self.calculate_boltzmann_term(l, i)
                        - self.calculate_boltzmann_term(l, m)
                    )
                )
                for m in [0, 1]
                for l in [0, 1]
            )
            - sum(
                self.calculate_potential_term(i, l)
                * probabilities[l][m]
                * self.calculate_potential_term(m, j)
                * (
                    self.calculate_boltzmann_term(m, l)
                    * (
                        self.calculate_boltzmann_term(m, j)
                        - self.calculate_boltzmann_term(l, j)
                    )
                )
                for m in [0, 1]
                for l in [0, 1]
            )
            + sum(
                self.calculate_potential_term(i, l)
                * probabilities[l][m]
                * self.calculate_potential_term(m, j)
                * (
                    self.calculate_boltzmann_term(m, l)
                    * (
                        self.calculate_boltzmann_term(m, i)
                        - self.calculate_boltzmann_term(l, i)
                    )
                )
                for m in [0, 1]
                for l in [0, 1]
            )
        )

    def _calculate_second_order_derivative(self, time, i, j, probabilities):
        return (
            self._calculate_second_order_diagonal_derivative(time, i, j, probabilities)
            * self.calculate_second_order_diagonal_derivative_prefactor(time, i, j)
            # self._calculate_second_order_cross_derivative(time, i, j, probabilities)
            # * self.calculate_second_order_cross_derivative_prefactor(time, i, j)
        )

    def _calculate_first_derivative(self, time, i, j, probabilities):
        return sum(
            self.calculate_potential_term(i, l)
            * probabilities[l][j]
            * self.calculate_boltzmann_term(l, j)
            for l in [0, 1]
        ) - sum(
            probabilities[i][l]
            * self.calculate_potential_term(l, j)
            * self.calculate_boltzmann_term(i, l)
            for l in [0, 1]
        )

    def _calculate_first_order_derivative(self, time, i, j, probabilities):
        return (
            self._calculate_first_derivative(time, i, j, probabilities)
        ) * self.calculate_first_order_derivative_prefactor(time, i, j)

    def _calculate_derivative(self, time, i, j, probabilities):
        first_order = self._calculate_first_order_derivative(time, i, j, probabilities)
        second_order = self._calculate_second_order_derivative(
            time, i, j, probabilities
        )
        # print(second_order)
        # return first_order
        # return second_order
        return first_order + second_order

    def redfield_derivatives(self, t, p):
        (p00, p01, p10, p11) = p
        probabilities = [[p00, p01], [p10, p11]]
        # print(
        #     self._calculate_derivative(t, 0, 0, probabilities),
        #     self._calculate_derivative(t, 1, 0, probabilities),
        #     self._calculate_derivative(t, 0, 1, probabilities),
        # )
        return (
            self._calculate_derivative(t, 0, 0, probabilities),
            self._calculate_derivative(t, 0, 1, probabilities),
            self._calculate_derivative(t, 1, 0, probabilities),
            self._calculate_derivative(t, 1, 1, probabilities),
        )

    def _solve_redfield_equation(self):
        print(
            [
                self.calculate_single_boltzmann_term(m)
                * self.calculate_single_boltzmann_term(j)
                - self.calculate_single_boltzmann_term(m)
                * self.calculate_single_boltzmann_term(l)
                for m in [0, 1]
                for j in [0, 1]
                for l in [0, 1]
            ]
        )
        soln = scipy.integrate.solve_ivp(
            fun=self.redfield_derivatives,
            t_span=(self.times[0], self.times[-1]),
            y0=self.initial_state,
            t_eval=self.times,
            max_step=self._max_step,
        )
        self._soln = {"t": soln.t, "y": soln.y}

    def save_to_file(self, file):
        if self._soln is None:
            np.savez(
                file,
                times=self.times,
                temperature=self.temperature,
                initial_state=self.initial_state,
                yvals=None,
                allow_pickle=True,
            )
        else:
            np.savez(
                file,
                times=self._soln["t"],
                temperature=self.temperature,
                initial_state=self.initial_state,
                yvals=self._soln["y"],
                allow_pickle=True,
            )

    @classmethod
    def load_from_file(cls, file):
        data = np.load(file, allow_pickle=True)
        temperature = data["temperature"]
        times = data["times"]
        initial_state = data["initial_state"]
        solver = cls(times, temperature, initial_state)
        yvals = data["yvals"]
        if not np.array_equal(yvals, None):
            solver._soln = {"t": times, "y": yvals}
        return solver


if __name__ == "__main__":
    a = [[0, 1], [10, 11]]
    print(a[1][0])
