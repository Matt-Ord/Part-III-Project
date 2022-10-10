from abc import ABC, abstractmethod
from os import stat
from typing import Dict, List, Type, Union
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

    def calculate_occupation_of_higher_band(self, occupation_fraction):
        # beta * egap
        energy_gap_exponent_factor = self.energy_difference / self.boltzmann_energy
        exponent_factor = (
            self._calculate_factor_from_fermi_energy(occupation_fraction)
            + energy_gap_exponent_factor
        )
        with np.errstate(over="ignore"):
            return 1 / (1 + np.exp(exponent_factor))

    def plot_adjusted_occupation_against_occupation(self):
        fig, ax = plt.subplots(1)
        OneBandResultsAnalyserUtil.plot_adjusted_occupation_against_occupation(self, ax)
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
        OneBandResultsAnalyserUtil.plot_adjusted_log_rate_against_occupation(
            {"Corrected Curve": self}, ax
        )
        plt.show()

    def plot_adjusted_rate_against_occupation(self):
        fig, ax = plt.subplots(1)
        OneBandResultsAnalyserUtil.plot_adjusted_rate_against_occupation(
            {"Corrected Curve": self}, ax
        )
        plt.show()

    def plot_adjusted_rate_against_energy(self):
        fig, ax = plt.subplots(1)
        OneBandResultsAnalyserUtil.plot_adjusted_rate_against_energy(
            {"Corrected Curve": self}, ax
        )
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

    def integrate_measured_total_rate(self):
        total_rate = scipy.integrate.quad(
            lambda e: self.measured_rate_curve(self.calculate_occupation(e))
            / self.simulation_energy_bandwidth,
            self.calculate_energy(0.9999),
            self.calculate_energy(0.0001),
        )[0]
        return total_rate

    def integrate_adjusted_total_rate(self):
        total_rate = scipy.integrate.quad(
            lambda e: self.adjusted_rate_curve(self.calculate_occupation(e))
            / self.simulation_energy_bandwidth,
            self.calculate_energy(0.9999),
            self.calculate_energy(0.0001),
        )[0]
        return total_rate


class OneBandResultsAnalyserUtil:
    @staticmethod
    def plot_adjusted_occupation_against_occupation(
        analyser: OneBandResultsAnalyser, ax: Union[plt.Axes, None] = None
    ):
        if ax is None:
            fig, ax = plt.subplots(1)

        occupation_fractions = np.linspace(0.01, 0.99, 1000)

        ax.plot(
            occupation_fractions,
            analyser.calculate_occupation_of_lower_band(occupation_fractions),
        )

        ax.set_xlabel("Initial State Occupation")
        ax.set_ylabel("Final State Occupation")
        ax.set_title(
            (
                f"plot of Occupation Against Corrected Occupation\n"
                + r"for $\Delta{}E="
                + "{:.3g}".format(analyser.energy_difference)
                + r"J$"
            )
        )
        ax.set_xlim([0, 1])
        return ax.get_figure(), ax

    @staticmethod
    def plot_adjusted_log_rate_against_occupation(
        analysers: Dict[str, OneBandResultsAnalyser], ax: Union[plt.Axes, None] = None
    ):
        if ax is None:
            fig, ax = plt.subplots(1)
        occupation_fractions = np.linspace(0.01, 0.99, 1000)

        ax.plot(
            occupation_fractions,
            next(iter(analysers.values())).measured_log_rate_curve(
                occupation_fractions
            ),
            label="Measured Curve",
        )
        for (label, analyser) in analysers.items():
            ax.plot(
                occupation_fractions,
                next(iter(analysers.values())).adjusted_log_rate_curve(
                    occupation_fractions
                ),
                label="Corrected Curve",
            )
        ax.set_xlabel("Occupation of Initial State")
        ax.set_ylabel("ln(Rate)")
        ax.set_title(
            (
                f"Plot of ln(Rate) against Initial Occupation\n"
                + r"for $\Delta{}E="
                + "{:.3g}".format(next(iter(analysers.values())).energy_difference)
                + r"J$"
            )
        )
        ax.set_xlim([0, 1])
        ax.legend()

        return ax.get_figure(), ax

    @staticmethod
    def plot_adjusted_rate_against_occupation(
        analysers: Dict[str, OneBandResultsAnalyser], ax: Union[plt.Axes, None] = None
    ):
        if ax is None:
            fig, ax = plt.subplots(1)
        fig, ax = plt.subplots(1)
        occupation_fractions = np.linspace(0.01, 0.99, 1000)

        ax.plot(
            occupation_fractions,
            next(iter(analysers.values())).measured_rate_curve(occupation_fractions),
            label="Measured Curve",
        )
        for (label, analyser) in analysers.items():
            ax.plot(
                occupation_fractions,
                analyser.adjusted_rate_curve(occupation_fractions),
                label=label,
            )
        ax.set_xlabel("Initial State Occupation")
        ax.set_ylabel("Rate")
        ax.set_title(
            (
                f"Plot of Rate Against Initial Occupation\n"
                + r"for $\Delta{}E="
                + "{:.3g}".format(next(iter(analysers.values())).energy_difference)
                + r"J$"
            )
        )
        ax.set_xlim([0, 1])
        ax.legend()
        return ax.get_figure(), ax

    @staticmethod
    def plot_adjusted_rate_against_energy(
        analysers: Dict[str, OneBandResultsAnalyser],
        ax: Union[plt.Axes, None] = None,
        energy_range=2,
    ):
        if ax is None:
            fig, ax = plt.subplots(1)

        dummy_analyser = next(iter(analysers.values()))
        energies = np.linspace(
            dummy_analyser.chemical_potential
            - energy_range * dummy_analyser.energy_difference,
            dummy_analyser.chemical_potential
            + energy_range * dummy_analyser.energy_difference,
            1000,
        )
        occupation_fractions = dummy_analyser.calculate_occupation(energies)

        ax.plot(
            energies,
            dummy_analyser.measured_rate_curve(occupation_fractions),
            label="Measured Curve",
        )
        for (label, analyser) in analysers.items():
            ax.plot(
                energies,
                analyser.adjusted_rate_curve(occupation_fractions),
                label=label,
            )

        ax.axvline(
            x=dummy_analyser.chemical_potential,
            linestyle="dashed",
            color="black",
            alpha=0.3,
        )
        ax.axvline(
            x=dummy_analyser.chemical_potential + dummy_analyser.energy_difference,
            linestyle="dashed",
            color="black",
            alpha=0.3,
        )

        ax.set_xlabel(r"Energy of Initial State / J")
        ax.set_ylabel(r"Rate / $s^{-1}$")
        ax.set_ylim([0, None])
        ax.set_xlim([energies[0], energies[-1]])
        ax.set_title(
            (
                f"Plot of Rate Against Energy "
                + r"for $\Delta{}E=3.04\times{}10^{-21}"
                # + "{:.3g}".format(dummy_analyser.energy_difference)
                + r"J$"
            )
        )

        ax2 = ax.twinx()
        ax2.plot(
            energies,
            dummy_analyser.calculate_occupation(energies),
            color="black",
            alpha=0.3,
        )
        ax2.set_ylim([0, 1])
        ax2.set_ylabel("Occupation")
        ax.patch.set_visible(False)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.legend()

        fig: plt.Figure = ax.get_figure()
        fig.set_size_inches(8, 3)
        fig.tight_layout()
        return fig, ax