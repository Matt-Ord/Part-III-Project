from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import scipy.constants
import scipy.optimize
from properties.MaterialProperties import MaterialProperties
from simulation.ElectronSimulation import (
    ElectronSimulation,
    ElectronSimulationConfig,
    randomise_electron_energies,
)
from simulation.ElectronSimulationPlotter import ElectronSimulationPlotter


class MaterialSimulator(ABC):
    def __init__(
        self,
        material_properties: MaterialProperties,
        temperature: float,
        number_of_electrons: int,
    ) -> None:
        self.material_properties = material_properties
        self.temperature = temperature
        self.number_of_electrons = number_of_electrons
        self.electron_energies = self._generate_electron_energies()

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

    def _electron_jitter_function(self):
        return lambda x: randomise_electron_energies(x, self._get_energy_jitter())

    def _create_simulation(
        self, jitter_electrons=False, initial_occupancy=1
    ) -> ElectronSimulation:
        electron_energy_jitter = (
            self._electron_jitter_function() if jitter_electrons else lambda x: x
        )
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

    def plot_electron_densities(
        self, times: List[float], jitter_electrons: bool = False, initial_occupancy=1
    ):

        sim = self._create_simulation(jitter_electrons, initial_occupancy)

        ElectronSimulationPlotter.plot_electron_densities(
            sim,
            times,
            thermal=True,
        )

    def plot_energy_lines_with_summed_overlap(self, axs=None):
        sim = self._create_simulation(jitter_electrons=False)
        ElectronSimulationPlotter.plot_energy_lines_with_summed_overlap(sim, axs)

    def plot_summed_overlap_against_energy(self, ax=None):
        sim = self._create_simulation(jitter_electrons=False)
        ElectronSimulationPlotter.plot_summed_overlap_against_energy(sim, ax)

    def plot_energies_with_maximum_overlap(self, ax=None):
        sim = self._create_simulation(jitter_electrons=False)
        ElectronSimulationPlotter.plot_energies_with_maximum_overlap(sim, ax)

    def plot_final_energy_against_initial_energy(self, ax=None, subplot_lims=None):
        sim = self._create_simulation(jitter_electrons=False)
        ElectronSimulationPlotter.plot_final_energy_against_initial_energy(
            sim, ax, subplot_lims
        )

    def plot_unpertubed_material_energy_states(self):
        self.remove_all_block_factors_for_simulation()
        self.plot_energy_lines_with_summed_overlap()
        self.plot_summed_overlap_against_energy()

        # with diagonal interaction
        self.remove_off_diagonal_block_factors_for_simulation()
        self.plot_energy_lines_with_summed_overlap()
        self.plot_summed_overlap_against_energy()

        # with off diagonal interaction
        self.remove_diagonal_block_factors_for_simulation()
        self.plot_energy_lines_with_summed_overlap()
        self.plot_summed_overlap_against_energy()

    def plot_material_energy_states(self, subplot_lims=None):
        self.plot_energy_lines_with_summed_overlap()
        self.plot_summed_overlap_against_energy()

        self.plot_energies_with_maximum_overlap()
        self.plot_final_energy_against_initial_energy(subplot_lims)

    def plot_electron_density_matrix(self, time):
        sim = self._create_simulation(jitter_electrons=False)
        return ElectronSimulationPlotter.plot_electron_density_matrix(
            sim, time, thermal=True
        )

    def plot_time_average_electron_density_matrix(self, times):
        sim = self._create_simulation(jitter_electrons=False)
        return ElectronSimulationPlotter.plot_time_average_electron_density_matrix(
            sim, times, thermal=True
        )

    def plot_density_matrix(self, time):
        sim = self._create_simulation(jitter_electrons=False)
        return ElectronSimulationPlotter.plot_density_matrix(sim, time, thermal=True)

    def plot_time_average_density_matrix(self, times):
        sim = self._create_simulation(jitter_electrons=False)
        return ElectronSimulationPlotter.plot_time_average_density_matrix(
            sim, times, thermal=True
        )

    def plot_off_diagonal_density_matrix(self, times):
        sim = self._create_simulation(jitter_electrons=False)
        return ElectronSimulationPlotter.plot_off_diagonal_density_matrix(
            sim, times, thermal=True
        )

    def plot_average_off_diagonal_density_matrix(
        self, initial_time, average_over_times, average_over=10
    ):
        sim = self._create_simulation(jitter_electrons=False)
        return ElectronSimulationPlotter.plot_average_off_diagonal_density_matrix(
            sim, initial_time, average_over_times, average_over=average_over
        )

    def plot_average_material(
        self,
        times,
        average_over=10,
        jitter_electrons=False,
        ax=None,
        initial_occupancy=1,
    ):
        sim = self._create_simulation(
            jitter_electrons, initial_occupancy=initial_occupancy
        )
        electron_densities_for_each = sim.get_electron_densities_for_each(
            times,
            average_over=average_over,
            thermal=True,
            jitter_for_each=jitter_electrons,
        )

        fig, ax = ElectronSimulationPlotter.plot_average_number_against_time_for_each(
            [
                {
                    "FCC": [sum(x) for x in density[:, 0]],
                    "HCP": [sum(x) for x in density[:, 1]],
                }
                for density in electron_densities_for_each
            ],
            times,
            ax,
        )
        ax.set_xlabel("Time / s")
        return fig, ax

    def plot_average_electron_distribution(
        self,
        times,
        average_over=10,
        ax=None,
    ):
        sim = self._create_simulation(jitter_electrons=True)
        return ElectronSimulationPlotter.plot_average_electron_distribution(
            sim, times, average_over, ax
        )

    def plot_average_densities(
        self,
        times: list[float],
        average_over=10,
        jitter_electrons=False,
        initial_occupancy=1,
        **kwargs
    ):
        sim = self._create_simulation(jitter_electrons, initial_occupancy)
        ElectronSimulationPlotter.plot_average_densities(
            sim,
            times,
            average_over=average_over,
            thermal=True,
            jitter_for_each=jitter_electrons,
            **kwargs,
        )

    def get_average_electron_densities(
        self,
        times,
        average_over=10,
        jitter_electrons=False,
        initial_occupancy=1,
        **kwargs
    ):
        sim = self._create_simulation(jitter_electrons, initial_occupancy)
        electron_densities_for_each = sim.get_electron_densities_for_each(
            times,
            average_over=average_over,
            thermal=True,
            jitter_for_each=jitter_electrons,
            **kwargs,
        )

        average_densities = (
            ElectronSimulationPlotter.calculate_average_electron_density(
                electron_densities_for_each
            )
        )
        return average_densities

    def get_initial_electron_densities(self, initial_occupancy=1, average_over=1):
        vals = []
        for _ in range(average_over):
            sim = self._create_simulation(
                jitter_electrons=False, initial_occupancy=initial_occupancy
            )
            vals.append(sim.get_electron_densities(times=[0], thermal=True)[0][0])
        return np.average(vals, axis=0)
