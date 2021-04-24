from abc import ABC, abstractmethod
from typing import List, Union

import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.colorbar
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.constants
from properties.MaterialProperties import MaterialProperties
from simulation.ElectronSimulation import ElectronSimulation, ElectronSimulationConfig
from simulation.ElectronSimulationPlotter import ElectronSimulationPlotter


class MaterialSimulator(ABC):
    def __init__(
        self,
        material_properties: MaterialProperties,
        temperature: float,
    ) -> None:
        self.material_properties = material_properties
        self.temperature = temperature
        self.electron_energies = self._generate_electron_energies()

    number_of_electrons = None

    @property
    def hydrogen_energies(self):
        return self.material_properties.hydrogen_energies

    @property
    def hydrogen_energies_for_simulation(self):
        return self.hydrogen_energies

    @property
    def hydrogen_energy_difference(self):
        return self.hydrogen_energies[1] - self.hydrogen_energies[0]

    @property
    def fermi_wavevector(self):
        return self.material_properties.fermi_wavevector

    @property
    def boltzmann_energy(self):
        return self.temperature * scipy.constants.Boltzmann

    @property
    def hydrogen_overlaps(self):
        return self.material_properties.hydrogen_overlaps

    _block_factors_for_simulation: Union[None, List[List[complex]]] = None

    @property
    def block_factors_for_simulation(self) -> list[list[complex]]:
        if self._block_factors_for_simulation is None:
            return self.hydrogen_overlaps
        return self._block_factors_for_simulation

    def reset_block_factors_for_simulation(self):
        self._block_factors_for_simulation = None

    def remove_diagonal_block_factors_for_simulation(self):
        M = self.hydrogen_overlaps
        self._block_factors_for_simulation = [
            [0, M[0][1]],
            [M[1][0], 0],
        ]

    def remove_off_diagonal_block_factors_for_simulation(self):
        M = self.hydrogen_overlaps
        self._block_factors_for_simulation = [[M[0][0], 0], [0, M[1][1]]]

    def remove_all_block_factors_for_simulation(self):
        self._block_factors_for_simulation = [[0, 0], [0, 0]]

    @abstractmethod
    def _generate_electron_energies(self, *args, **kwargs) -> List[float]:
        pass

    @abstractmethod
    def _get_energy_spacing(self):
        pass

    @staticmethod
    def _calculate_electron_energies(k_states):
        return (scipy.constants.hbar * k_states) ** 2 / (2 * scipy.constants.m_e)

    def _get_fermi_energy(self):
        return self._calculate_electron_energies(
            self.material_properties.fermi_wavevector
        )

    def _get_central_energy(self):
        return self._get_fermi_energy()

    # @abstractmethod
    # def _get_wavevector_spacing(self):
    #     pass

    # def _get_implied_volume_old(self):
    #     # old method using wavevector spacing instead of energy
    #     wavevector_spacing = self._get_wavevector_spacing()

    #     return ((scipy.constants.pi ** 2) /
    #             (wavevector_spacing *
    #              self.material_properties.fermi_wavevector ** 2))

    def _get_implied_volume(self):
        energy_spacing = self._get_energy_spacing()

        prefactor = (scipy.constants.pi ** 2) * (
            scipy.constants.hbar ** 2 / scipy.constants.m_e
        ) ** (3 / 2)
        energy_factor = (2 * self._get_central_energy()) ** (-1 / 2)

        return prefactor * energy_factor / energy_spacing

    def _get_interaction_prefactor(self):
        implied_volume = self._get_implied_volume()
        # Ignoring q dependance the interaction
        # takes the form -2e^2 / epsilon_0 alpha^2 (see 6.9 on lab book)
        alpha = 3.77948796 * 10 ** 10  # m^-1
        potential_factor = (
            -2 * (scipy.constants.e ** 2) / (scipy.constants.epsilon_0 * (alpha ** 2))
        )
        prefactor = 4 * np.pi * potential_factor / implied_volume
        return prefactor

    def _get_energy_jitter(self):
        return 0

    def _create_simulation(self, jitter_electrons=False, initial_occupancy=1):
        electron_energy_jitter = self._get_energy_jitter() if jitter_electrons else 0
        sim = ElectronSimulation(
            ElectronSimulationConfig(
                hbar=scipy.constants.hbar,
                boltzmann_energy=self.boltzmann_energy,
                electron_energies=self.electron_energies,
                hydrogen_energies=self.hydrogen_energies_for_simulation,
                block_factors=self.block_factors_for_simulation,
                q_prefactor=self._get_interaction_prefactor(),
                electron_energy_jitter=electron_energy_jitter,
                number_of_electrons=self.number_of_electrons,
                initial_occupancy=initial_occupancy,
            )
        )
        return sim

    def simulate_material(
        self, times: List[float], jitter_electrons: bool = False, initial_occupancy=1
    ):

        sim = self._create_simulation(jitter_electrons, initial_occupancy)

        ElectronSimulationPlotter.plot_random_system_evolved_coherently(
            sim,
            times,
            thermal=True,
        )

    @staticmethod
    def _plot_energies_and_overlaps(
        energies, overlaps, title="Plot of eigenstate energy levels"
    ):
        gs = gridspec.GridSpec(1, 6)
        plt.figure()
        ax1 = plt.subplot(gs[0, :-1])
        ax2 = plt.subplot(gs[0, -1])
        cmap = cm.get_cmap("viridis")

        for (energy, overlap) in zip(energies.tolist(), overlaps):
            ax1.axvline(energy, color=cmap(overlap))
        ax1.set_title(title)
        ax1.set_xlabel("Energy /J")

        cb1 = matplotlib.colorbar.ColorbarBase(
            ax2, cmap=cmap, orientation="vertical"
        )  # type: ignore
        ax2.set_ylabel("Proportion of state in initial")
        plt.show()

        fig, ax = plt.subplots(1)
        ax.plot(energies, overlaps, "+")
        ax.set_title(title)
        ax.set_xlabel("Energy /J")
        ax.set_ylabel("overlap")
        plt.show()

    def plot_material_energy_states(self):
        self.remove_all_block_factors_for_simulation()
        sim = self._create_simulation(jitter_electrons=False)
        energies, overlaps = sim.get_energies_and_overlaps()
        self._plot_energies_and_overlaps(
            energies, overlaps, title="Plot of unperturbed eigenstate energy levels"
        )

        # with diagonal interaction
        self.remove_off_diagonal_block_factors_for_simulation()
        sim = self._create_simulation(
            jitter_electrons=False,
        )
        energies, overlaps = sim.get_energies_and_overlaps()
        self._plot_energies_and_overlaps(
            energies,
            overlaps,
            title="Plot of eigenstate energy levels with diagonal correction",
        )

        # with off diagonal interaction
        self.remove_diagonal_block_factors_for_simulation()
        sim = self._create_simulation(
            jitter_electrons=False,
        )
        energies, overlaps = sim.get_energies_and_overlaps()
        self._plot_energies_and_overlaps(
            energies,
            overlaps,
            title="Plot of eigenstate energy levels with off diagonal perturbation",
        )

        # with full interaction
        self.reset_block_factors_for_simulation()
        sim = self._create_simulation(jitter_electrons=False)
        energies, overlaps = sim.get_energies_and_overlaps()
        self._plot_energies_and_overlaps(
            energies, overlaps, title="Plot of perturbed eigenstate energy levels"
        )

    def plot_average_material(
        self,
        times,
        average_over=10,
        jitter_electrons=False,
        title="",
    ):
        sim = self._create_simulation(jitter_electrons)
        electron_densities_for_each = sim.simulate_random_system_coherently_for_each(
            times,
            average_over=average_over,
            thermal=True,
            jitter_for_each=jitter_electrons,
        )

        fig, ax = ElectronSimulationPlotter.plot_total_number_against_time_for_each(
            [
                {
                    "fcc": [sum(x) for x in density[:, 0]],
                    "hcp": [sum(x) for x in density[:, 1]],
                }
                for density in electron_densities_for_each
            ],
            times,
        )
        ax.set_title(title)
        ax.set_xlabel("Time / s")
        plt.show()

    def simulate_average_material(
        self,
        times: list[float],
        average_over=10,
        jitter_electrons=False,
        initial_occupancy=0.5,
        **kwargs
    ):

        sim = self._create_simulation(jitter_electrons, initial_occupancy)

        ElectronSimulationPlotter.plot_average_densities_of_system_evolved_coherently(
            sim,
            times,
            average_over=average_over,
            thermal=True,
            jitter_for_each=jitter_electrons,
            **kwargs,
        )

    # def _occupation_curve_fit_function(self, time, omega, decay_time):
    #     return ElectronSimulationPlotter.occupation_curve_fit_function(
    #         self.number_of_electrons, time, omega, decay_time
    #     )

    # def _fit_electron_occupation_curve(self, times, initially_occupied_densities):
    #     initial_omega_guess = 10 / (times[-1] - times[0])
    #     initial_decay_time_guess = (times[-1] - times[0]) / 4

    #     return scipy.optimize.curve_fit(
    #         lambda t, w, d: self._occupation_curve_fit_function,
    #         times,
    #         initially_occupied_densities,
    #         p0=[initial_omega_guess, initial_decay_time_guess],
    #     )

    def simulate_average_densities(
        self, times, average_over=10, jitter_electrons=False, **kwargs
    ):
        sim = self._create_simulation(jitter_electrons)
        electron_densities_for_each = sim.simulate_random_system_coherently_for_each(
            times,
            average_over=average_over,
            thermal=True,
            jitter_for_each=jitter_electrons,
            **kwargs,
        )

        average_densities = ElectronSimulationPlotter.calculate_average_density(
            electron_densities_for_each
        )
        return average_densities

        initially_occupied_densities = [d[0] for d in average_densities]
        initially_unoccupied_densities = [d[1] for d in average_densities]

        (optimal_omega, optimal_decay_time), pcov = self._fit_electron_occupation_curve(
            times, initially_occupied_densities
        )

        print("omega", optimal_omega)
        print("decay_time", optimal_decay_time)
        print(pcov)
