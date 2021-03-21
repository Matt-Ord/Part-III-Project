from abc import ABC, abstractmethod
from typing import Type
import numpy as np
import scipy.constants
import scipy.integrate
import matplotlib.pyplot as plt

from properties.MaterialProperties import MaterialProperties, NICKEL_MATERIAL_PROPERTIES


class OneBandResultsAnalyser(ABC):
    def __init__(
        self,
        amplitude: float,
        offset: float,
        temperature: float,
        material_properties: MaterialProperties,
        simulation_energy_bandwidth: float,
    ) -> None:
        self.amplitude = amplitude
        self.offset = offset
        self.material_properties = material_properties
        self.temperature = temperature
        self.simulation_energy_bandwidth = simulation_energy_bandwidth

    @property
    def energy_difference(self):
        return (
            self.material_properties.hydrogen_energies[1]
            - self.material_properties.hydrogen_energies[0]
        )

    @property
    def chemical_potential(self):
        return self._calculate_energy_of_wavevector(
            self.material_properties.fermi_wavevector
        )

    @property
    def boltzmann_energy(self):
        return self.temperature * scipy.constants.Boltzmann

    @staticmethod
    def _calculate_energy_of_wavevector(wavevector):
        return (scipy.constants.hbar * wavevector) ** 2 / (2 * scipy.constants.m_e)

    def calculate_occupation(self, energy):
        with np.errstate(over="ignore"):
            return 1 / (
                1 + np.exp((energy - self.chemical_potential) / self.boltzmann_energy)
            )

    def calculate_energy(self, occupation_fraction):
        return (
            self.boltzmann_energy
            * self._calculate_factor_from_fermi_energy(occupation_fraction)
            + self.chemical_potential
        )

    # Equal to beta (E - Ef)
    def _calculate_factor_from_fermi_energy(self, occupation_fraction):
        with np.errstate(divide="ignore"):
            return np.log((1 / occupation_fraction) - 1)

    def calculate_occupation_of_lower_band(self, occupation_fraction):
        # beta * egap
        energy_gap_exponent_factor = self.energy_difference / self.boltzmann_energy
        exponent_factor = (
            self._calculate_factor_from_fermi_energy(occupation_fraction)
            - energy_gap_exponent_factor
        )
        return 1 / (1 + np.exp(exponent_factor))

    def plot_adjusted_occupation_against_occupation(self):
        fig, ax = plt.subplots(1)
        occupation_fractions = np.linspace(0.01, 0.99, 1000)

        ax.plot(
            occupation_fractions,
            self.calculate_occupation_of_lower_band(occupation_fractions),
        )

        ax.set_xlabel("initial state occupation")
        ax.set_ylabel("final state occupation")
        ax.set_title(
            (
                f"plot of occupation against corrected occupation\n"
                + r"for $\Delta{}E="
                + "{:.3g}".format(self.energy_difference)
                + r"J$"
            )
        )
        ax.set_xlim([0, 1])
        plt.show()

    @abstractmethod
    def different_occupancy_rate(
        self, initial_occupation_fraction, final_occupation_fraction
    ) -> np.ndarray:
        pass

    def adjusted_log_rate_curve(self, occupation_fraction) -> np.ndarray:
        return np.log(self.adjusted_rate_curve(occupation_fraction))

    def adjusted_rate_curve(self, occupation_fraction) -> np.ndarray:
        lower_occupation_fraction = self.calculate_occupation_of_lower_band(
            occupation_fraction
        )
        return 0.5 * (
            self.different_occupancy_rate(
                occupation_fraction, lower_occupation_fraction
            )
            + self.different_occupancy_rate(
                lower_occupation_fraction, occupation_fraction
            )
        )

    def measured_rate_curve(self, occupation_fraction) -> np.ndarray:
        return self.different_occupancy_rate(occupation_fraction, occupation_fraction)

    def measured_log_rate_curve(self, occupation_fraction) -> np.ndarray:
        return np.log(self.measured_rate_curve(occupation_fraction))

    def plot_adjusted_log_rate_against_occupation(self):
        fig, ax = plt.subplots(1)
        occupation_fractions = np.linspace(0.01, 0.99, 1000)

        ax.plot(
            occupation_fractions,
            self.measured_log_rate_curve(occupation_fractions),
            label="Measured Curve",
        )

        ax.plot(
            occupation_fractions,
            self.adjusted_log_rate_curve(occupation_fractions),
            label="Corrected Curve",
        )
        ax.set_xlabel("occupation of initial state")
        ax.set_ylabel("ln(rate)")
        ax.set_title(
            (
                f"plot of ln(rate) against initial occupation\n"
                + r"for $\Delta{}E="
                + "{:.3g}".format(self.energy_difference)
                + r"J$"
            )
        )
        ax.set_xlim([0, 1])
        plt.legend()
        plt.show()

    def plot_adjusted_rate_against_occupation(self):
        fig, ax = plt.subplots(1)
        occupation_fractions = np.linspace(0.01, 0.99, 1000)

        ax.plot(
            occupation_fractions,
            self.measured_rate_curve(occupation_fractions),
            label="Measured Curve",
        )

        ax.plot(
            occupation_fractions,
            self.adjusted_rate_curve(occupation_fractions),
            label="Corrected Curve",
        )
        ax.set_xlabel("occupation of initial state")
        ax.set_ylabel("rate")
        ax.set_title(
            (
                f"plot of rate against initial occupation\n"
                + r"for $\Delta{}E="
                + "{:.3g}".format(self.energy_difference)
                + r"J$"
            )
        )
        ax.set_xlim([0, 1])
        plt.legend()
        plt.show()

    def plot_adjusted_rate_against_energy(self):
        fig, ax = plt.subplots(1)
        energies = np.linspace(
            self.chemical_potential - 2 * self.energy_difference,
            self.chemical_potential + 2 * self.energy_difference,
            1000,
        )
        occupation_fractions = self.calculate_occupation(energies)

        ax.plot(
            energies,
            self.measured_rate_curve(occupation_fractions),
            label="Measured Curve",
        )

        ax.plot(
            energies,
            self.adjusted_rate_curve(occupation_fractions),
            label="Corrected Curve",
        )

        ax.axvline(
            x=self.chemical_potential,
            linestyle="dashed",
            color="black",
            alpha=0.3,
        )
        ax.axvline(
            x=self.chemical_potential + self.energy_difference,
            linestyle="dashed",
            color="black",
            alpha=0.3,
        )

        ax.set_xlabel("energy of initial state")
        ax.set_ylabel("rate")
        ax.set_ylim([0, None])
        ax.set_title(
            (
                f"plot of rate against energy\n"
                + r"for $\Delta{}E="
                + "{:.3g}".format(self.energy_difference)
                + r"J$"
            )
        )
        plt.legend()
        plt.show()

    def calculate_measured_total_rate(self):
        sample_energies = np.arange(100000000) * self.simulation_energy_bandwidth

        rates = self.measured_rate_curve(self.calculate_occupation(sample_energies))

        total_rate = scipy.integrate.simps(y=rates)
        return total_rate

    def calculate_adjusted_total_rate(self):
        sample_energies = np.arange(100000000) * self.simulation_energy_bandwidth

        rates = self.adjusted_rate_curve(self.calculate_occupation(sample_energies))

        total_rate = scipy.integrate.simps(y=rates)
        return total_rate
