from typing import Type
from FullLindbladSolver import FullLindbladSolver, FullLindbladWithSinkSolver
import numpy as np
import matplotlib.pyplot as plt
from LindbladSolver import LindbladSolver, TwoSiteLindbladSolver
from NoEnergyGapLindbladSolver import NoEnergyGapLindbladSolver
from RotatingWaveLindbladSolver import RotatingWaveLindbladSolver
import scipy.constants
import experemental_data

from properties.MaterialProperties import NICKEL_MATERIAL_PROPERTIES


class LindbladSolverPlotter:
    def __init__(self, solver: TwoSiteLindbladSolver) -> None:
        self.solver = solver

    @staticmethod
    def _plot_probabilities_against_time(times, propbabilities, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(times, np.abs(propbabilities), label=label)
        return ax

    def plot_diagonal_terms_against_time(self, ax=None, labels=["P00", "P11"]):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        self._plot_probabilities_against_time(
            self.solver.times, self.solver.p00_values, label=labels[0], ax=ax
        )
        self._plot_probabilities_against_time(
            self.solver.times, self.solver.p11_values, label=labels[1], ax=ax
        )
        ax.set_ylim([0, 1])
        ax.set_xlim([self.solver.times[0], None])
        ax.set_title("Plot of the diagonal terms of the density matrix against time")
        ax.legend()
        return fig, ax

    def plot_final_state_density_against_time(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        self._plot_probabilities_against_time(
            self.solver.times, self.solver.p11_values, label="P11", ax=ax
        )
        ax.set_xlim([self.solver.times[0], None])
        ax.set_title("Plot of the final state density against time")
        ax.legend()
        return fig, ax

    def plot_initial_state_density_against_time(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        self._plot_probabilities_against_time(
            self.solver.times, self.solver.p00_values, label="P00", ax=ax
        )
        ax.set_xlim([self.solver.times[0], None])
        ax.set_title("Plot of the initial state density against time")
        ax.legend()
        return fig, ax

    def plot_cross_terms_against_time(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        self._plot_probabilities_against_time(
            self.solver.times, self.solver.p01_values, label="P01", ax=ax
        )
        self._plot_probabilities_against_time(
            self.solver.times, self.solver.p10_values, label="P10", ax=ax
        )
        ax.set_xlim([self.solver.times[0], None])
        ax.set_title("Plot of the cross terms of the density matrix against time")
        ax.legend()
        return fig, ax

    def plot_solution(self):
        self.plot_diagonal_terms_against_time()
        self.plot_cross_terms_against_time()
        self.plot_final_state_density_against_time()
        plt.show()
        return self

    def print_final_density(self):
        print(
            self.solver.p00_values[-1],
            self.solver.p01_values[-1],
            self.solver.p10_values[-1],
            self.solver.p11_values[-1],
        )
        return self

    @classmethod
    def from_file(cls, solver_type: Type[TwoSiteLindbladSolver], file):
        return cls(solver_type.load_from_file(file))


def rotating_wave_solution1(time, gamma1, gamma2):
    return (gamma1 / (gamma1 + gamma2)) * (
        (gamma2 / gamma1) + np.exp(-(gamma1 + gamma2) * time)
    )


def rotating_wave_solution2(time, gamma1, gamma2):
    return (gamma1 / (gamma1 + gamma2)) * (1 - np.exp(-(gamma1 + gamma2) * time))


def plot_rotating_wave_lindblad():
    solver = RotatingWaveLindbladSolver(
        times=np.linspace(0, 1 * 10 ** (-8), 2000),
        temperature=150,
        initial_state=[1, 0, 0, 0],
    )
    fig, ax = LindbladSolverPlotter(solver).plot_diagonal_terms_against_time()

    gamma1 = 2 * solver._get_gamma_abcd_omega_ij(1, 0, 1, 0, 1, 0)
    gamma2 = 2 * solver._get_gamma_abcd_omega_ij(0, 1, 0, 1, 0, 1)

    decay_rate = gamma1 + gamma2
    print(decay_rate)
    lower_line = gamma1 / (gamma1 + gamma2)
    ax.axhline(
        y=lower_line,
        linestyle="dashed",
        color="black",
        alpha=0.3,
    )

    upper_line = gamma2 / (gamma1 + gamma2)
    ax.axhline(
        y=upper_line,
        linestyle="dashed",
        color="black",
        alpha=0.3,
    )
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Probablity")
    ax.set_title(
        "Plot of the diagonal elements of the matrix against time \n"
        + r"showing a decay with a characteristic time of $1.6x10^{-9}$s"
    )
    plt.show()


def calculate_gamma_prefactor(temperature):
    fermi_wavevector = NICKEL_MATERIAL_PROPERTIES.fermi_wavevector
    boltzmann_energy = scipy.constants.Boltzmann * temperature
    # Calculation grouped to reduce floating point errors
    a = (scipy.constants.hbar**2) / (scipy.constants.elementary_charge**2)
    b = (boltzmann_energy * scipy.constants.hbar * (fermi_wavevector**2)) / (
        scipy.constants.elementary_charge**2
    )
    c = (scipy.constants.epsilon_0**2) / (scipy.constants.electron_mass**2)
    d = 64 * np.sqrt(np.pi)
    return a * b * c * d


def calculate_combined_lindblad_rate(temperature):
    delta_e = (
        NICKEL_MATERIAL_PROPERTIES.hydrogen_energies[0]
        - NICKEL_MATERIAL_PROPERTIES.hydrogen_energies[1]
    )
    return (
        2
        * NICKEL_MATERIAL_PROPERTIES.hydrogen_overlaps[0][1]
        * NICKEL_MATERIAL_PROPERTIES.hydrogen_overlaps[1][0]
        * calculate_gamma_prefactor(temperature)
        * np.cosh(delta_e / (2 * scipy.constants.Boltzmann * temperature))
    )


def calculate_slow_direction_lindblad_rate(temperature):
    delta_e = (
        NICKEL_MATERIAL_PROPERTIES.hydrogen_energies[0]
        - NICKEL_MATERIAL_PROPERTIES.hydrogen_energies[1]
    )
    return (
        NICKEL_MATERIAL_PROPERTIES.hydrogen_overlaps[0][1]
        * NICKEL_MATERIAL_PROPERTIES.hydrogen_overlaps[1][0]
        * calculate_gamma_prefactor(temperature)
        * np.exp(delta_e / (2 * scipy.constants.Boltzmann * temperature))
    )


def calculate_fast_direction_lindblad_rate(temperature):
    delta_e = (
        NICKEL_MATERIAL_PROPERTIES.hydrogen_energies[0]
        - NICKEL_MATERIAL_PROPERTIES.hydrogen_energies[1]
    )
    return (
        NICKEL_MATERIAL_PROPERTIES.hydrogen_overlaps[0][1]
        * NICKEL_MATERIAL_PROPERTIES.hydrogen_overlaps[1][0]
        * calculate_gamma_prefactor(temperature)
        * np.exp(-delta_e / (2 * scipy.constants.Boltzmann * temperature))
    )


def plot_rate_against_temperature():
    temperatures = np.linspace(100, 255, 1000)
    fig, ax = plt.subplots()
    ax.plot(
        1 / temperatures,
        np.log10(3 * calculate_combined_lindblad_rate(temperatures)),
        label="lindblad",
    )
    ax.plot(
        1 / temperatures,
        np.log10(3 * calculate_slow_direction_lindblad_rate(temperatures)),
        label="slow direction",
    )
    ax.plot(
        1 / temperatures,
        np.log10(3 * calculate_fast_direction_lindblad_rate(temperatures)),
        label="fast direction",
    )
    ax.errorbar(
        1 / experemental_data.temperature,
        np.log10(experemental_data.jumprate),
        yerr=[
            np.log10(experemental_data.absuppererrorvalue / experemental_data.jumprate),
            np.log10(experemental_data.jumprate / experemental_data.abslowererrorvalue),
        ],
        label="experemental data",
    )
    ax.set_title("Plot of Experemental and Theoretical Tunnelling Rate")
    ax.set_ylabel("log(jumprate)")
    ax.set_xlabel("1/Temperature")
    print(3 * calculate_combined_lindblad_rate(150))
    ax.legend()
    plt.show()


def plot_full_lindblad():
    solver = FullLindbladSolver(
        times=np.linspace(0, 1 * 10 ** (-12), 2000),
        temperature=150,
        initial_state=[1, 0, 0, 0],
    )
    gamma1 = 2 * solver._get_gamma_abcd_omega_ij(1, 0, 1, 0, 1, 0)
    gamma2 = 2 * solver._get_gamma_abcd_omega_ij(0, 1, 0, 1, 0, 1)

    decay_rate = gamma1 + gamma2
    print(decay_rate)

    fig, ax = LindbladSolverPlotter(solver).plot_final_state_density_against_time()
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Final State Probablity")
    ax.set_title(
        "Plot of full linblad solution, displaying oscillations\n"
        + r"with a characteristic time of $2.2x10^{-13}$"
    )
    ax.plot(
        solver.times,
        rotating_wave_solution2(solver.times, gamma1, gamma2),
        alpha=0.3,
        color="red",
        label="rotating wave",
    )
    ax.legend()
    fig.tight_layout()
    plt.show()

    solver = FullLindbladSolver(
        times=np.linspace(0, 1 * 10 ** (-8), 2000),
        temperature=150,
        initial_state=[1, 0, 0, 0],
    )
    solver._max_step = 10 ** (-14)
    fig, ax = LindbladSolverPlotter(solver).plot_diagonal_terms_against_time()

    lower_line = gamma1 / (gamma1 + gamma2)
    ax.axhline(
        y=lower_line,
        linestyle="dashed",
        color="black",
        alpha=0.3,
    )

    upper_line = gamma2 / (gamma1 + gamma2)
    ax.axhline(
        y=upper_line,
        linestyle="dashed",
        color="black",
        alpha=0.3,
    )
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Probablity")
    ax.set_title(
        "Plot of the diagonal elements of the matrix against time \n"
        + r"alongside a decay with a characteristic time of $1.6x10^{-9}$s"
    )

    ax.plot(
        solver.times,
        rotating_wave_solution1(solver.times, gamma1, gamma2),
        alpha=0.3,
        color="red",
    )
    ax.plot(
        solver.times,
        rotating_wave_solution2(solver.times, gamma1, gamma2),
        alpha=0.3,
        color="red",
    )
    fig.tight_layout()
    plt.show()


def plot_full_lindblad_with_sink():
    solver = FullLindbladWithSinkSolver(
        times=np.linspace(0, 1 * 10 ** (-12), 2000),
        temperature=150,
        initial_state=[1, 0, 0, 0],
    )
    gamma1 = 2 * solver._get_gamma_abcd_omega_ij(1, 0, 1, 0, 1, 0)
    gamma2 = 2 * solver._get_gamma_abcd_omega_ij(0, 1, 0, 1, 0, 1)

    decay_rate = gamma1 + gamma2
    print(decay_rate)

    fig, ax = LindbladSolverPlotter(solver).plot_initial_state_density_against_time()
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Final State Probablity")
    ax.set_title(
        "Plot of full linblad solution, displaying oscillations\n"
        + r"with a characteristic time of $2.2x10^{-13}$"
    )
    ax.plot(
        solver.times,
        rotating_wave_solution1(solver.times, gamma1, gamma2),
        alpha=0.3,
        color="red",
        label="rotating wave",
    )
    ax.plot(
        solver.times,
        rotating_wave_solution1(solver.times, gamma1, 0),
        alpha=0.3,
        color="green",
        label="rotating wave with sink",
    )
    ax.legend()
    fig.tight_layout()
    plt.show()

    solver = FullLindbladWithSinkSolver(
        times=np.linspace(0, 1 * 10 ** (-8), 2000),
        temperature=150,
        initial_state=[1, 0, 0, 0],
    )
    solver._max_step = 10 ** (-14)
    fig, ax = LindbladSolverPlotter(solver).plot_diagonal_terms_against_time()
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Probablity")
    ax.set_title(
        "Plot of the diagonal elements of the matrix against time \n"
        + r"alongside the rotating wave decay"
    )
    ax.plot(
        solver.times,
        rotating_wave_solution1(solver.times, gamma1, 0),
        color="green",
        linestyle="dashed",
        label="rotating wave with sink",
    )
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_no_gap_lindblad():
    solver = NoEnergyGapLindbladSolver(
        times=np.linspace(0, 2 * 10 ** (-5), 20000),
        temperature=150,
        initial_state=[1 + 0j, 0, 0, 0],
    )
    solver._solve_lindbald_equation()
    solver.save_to_file("solvers/no_gap_solver.npz")
    LindbladSolverPlotter.from_file(
        NoEnergyGapLindbladSolver, "solvers/no_gap_solver.npz"
    ).print_final_density().plot_solution()


def plot_many_state_lindblad():
    pass


if __name__ == "__main__":
    print(calculate_gamma_prefactor(150))
    # calculate_combined_lindblad_rate(150)
    # plot_full_lindblad_with_sink()
    # plot_full_lindblad()
    # plot_rotating_wave_lindblad()
    # plot_rate_against_temperature()
