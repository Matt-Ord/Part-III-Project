from abc import ABC, abstractmethod
from typing import List, Union

import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.colorbar
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from numpy.lib.function_base import average
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
    def _plot_energies_and_summed_overlaps(
        energies, overlaps, title="Plot of eigenstate energy levels"
    ):
        gs = gridspec.GridSpec(1, 6)
        plt.figure()
        ax1 = plt.subplot(gs[0, :-1])
        ax2 = plt.subplot(gs[0, -1])
        cmap = cm.get_cmap("viridis")

        for (energy, overlap) in zip(energies, overlaps):
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

    @staticmethod
    def _plot_energies_and_overlaps(
        energies,
        overlaps,
        noninteracting_energies,
        title="Plot of eigenstate energy levels",
        subplot_lims=None,
    ):

        flat_overlaps = np.array(
            [[x[0] for x in o] + [x[1] for x in o] for o in overlaps]
        )

        largest_overlap_energies = noninteracting_energies[
            np.argmax(flat_overlaps, axis=1)
        ]
        rtol = 0.01
        # number_of_states_within_tolerance = [
        #     np.count_nonzero(np.isclose(x, noninteracting_energies, atol=0, rtol=rtol))
        #     for x in noninteracting_energies
        # ]
        # print(number_of_states_within_tolerance)

        fraction_of_overlap_degenerate_in_energy = [
            np.sum(
                np.where(
                    np.isclose(
                        largest_overlap_energy,
                        noninteracting_energies,
                        atol=0,
                        rtol=rtol,
                    ),
                    flat_overlap,
                    0,
                )
            )
            for (largest_overlap_energy, flat_overlap) in zip(
                largest_overlap_energies, flat_overlaps
            )
        ]

        fig, ax = plt.subplots(1)

        ax.plot(energies, fraction_of_overlap_degenerate_in_energy)
        ax.set_xlabel("final state energies")
        ax.set_ylabel("largest initial state energy overlap")
        ax.set_title("Plot of initial state energy overlap fraction")
        plt.show()

        fig, ax = plt.subplots(1)
        average_noninteracting_energies = np.array(
            [
                np.average(noninteracting_energies, axis=None, weights=flat_overlap)
                for flat_overlap in flat_overlaps
            ]
        )
        varience_noninteracting_energies = [
            np.average(
                (noninteracting_energies - average_energy) ** 2, weights=flat_overlap
            )
            for (flat_overlap, average_energy) in zip(
                flat_overlaps, average_noninteracting_energies
            )
        ]

        ax.errorbar(
            average_noninteracting_energies,
            energies,
            xerr=np.sqrt(varience_noninteracting_energies),
            fmt="+",
        )
        if subplot_lims is not None:
            ax2: plt.Axes = fig.add_axes([0.2, 0.55, 0.25, 0.25])  # , facecolor="y")
            ax2.errorbar(
                average_noninteracting_energies,
                energies,
                xerr=np.sqrt(varience_noninteracting_energies),
                fmt="+",
            )
            ax2.set_xlim(subplot_lims[0])
            ax2.set_ylim(subplot_lims[1])
            ax2.set_yticks([subplot_lims[1][0], 0, subplot_lims[1][1]])

        ax.set_ylabel("final state energies / J")
        ax.set_xlabel("initial state energies / J")
        ax.set_title("Plot of initial vs final state energies")
        plt.show()

        # # TODO
        # plot_largest_n = 5
        # cmap = cm.get_cmap("Blues")
        # initial_state_colours = cmap(np.linspace(0.2, 0.8, plot_largest_n + 1))[::-1]
        # final_state_colours = cmap(np.linspace(0.2, 0.8, plot_largest_n + 1))[::-1]

        # initial_overlaps = np.array(
        #     [[o[0] for o in state_overlaps] for state_overlaps in overlaps]
        # )
        # initial_overlaps = np.array(
        #     [[o[0] for o in state_overlaps] for state_overlaps in overlaps]
        # )

        # largest_initial = initial_overlaps[
        #     np.argsort(initial_overlaps, axis=1)[:, -5::]
        # ]

        # fig, ax = plt.subplots(1)
        # for index, (energy, state_overlaps) in enumerate(zip(energies, overlaps)):
        #     # print(state_overlaps)
        #     initial_overlaps = np.array([o[0] for o in state_overlaps])
        #     total_initial_overlap = np.sum(initial_overlaps)
        #     largest = initial_overlaps[np.argsort(initial_overlaps)[-5::]]
        #     largest = np.concatenate(
        #         ([total_initial_overlap - np.sum(largest)], largest)
        #     )

        #     if True:
        #         y_offset = 0
        #         for i, overlap in enumerate(largest):
        #             ax.bar(
        #                 index,
        #                 largest,
        #                 1,
        #                 bottom=y_offset,
        #                 color=initial_state_colours[i],
        #             )
        #             y_offset += overlap
        #         # ax.bar(
        #         #     index, [d[0] for d in state_overlaps], 1, #bottom=0
        #         # )  # color=colors[row])
        #         # ax.bar(index, [d[1] for d in state_overlaps], 1, bottom=0)
        # ax.set_title(title)
        # ax.set_xlabel("Energy /J")
        # ax.set_ylabel("overlap")
        # plt.show()

    def plot_unpertubed_material_energy_states(self):
        self.remove_all_block_factors_for_simulation()
        sim = self._create_simulation(jitter_electrons=False)
        energies, summed_overlaps = sim.get_energies_and_summed_overlaps()
        self._plot_energies_and_summed_overlaps(
            energies,
            summed_overlaps,
            title="Plot of unperturbed eigenstate energy levels",
        )

        # with diagonal interaction
        self.remove_off_diagonal_block_factors_for_simulation()
        sim = self._create_simulation(
            jitter_electrons=False,
        )
        energies, summed_overlaps = sim.get_energies_and_summed_overlaps()
        self._plot_energies_and_summed_overlaps(
            energies,
            summed_overlaps,
            title="Plot of eigenstate energy levels with diagonal correction",
        )

        # with off diagonal interaction
        self.remove_diagonal_block_factors_for_simulation()
        sim = self._create_simulation(
            jitter_electrons=False,
        )
        energies, summed_overlaps = sim.get_energies_and_summed_overlaps()
        self._plot_energies_and_summed_overlaps(
            energies,
            summed_overlaps,
            title="Plot of eigenstate energy levels with off diagonal perturbation",
        )

    def plot_material_energy_states(self, subplot_lims=None):

        # with full interaction
        self.reset_block_factors_for_simulation()
        sim = self._create_simulation(jitter_electrons=False)
        energies, summed_overlaps = sim.get_energies_and_summed_overlaps()
        self._plot_energies_and_summed_overlaps(
            energies,
            summed_overlaps,
            title="Plot of eigenstate energy levels",
        )

        # full interaction, full split of overlaps
        self.reset_block_factors_for_simulation()
        sim = self._create_simulation(jitter_electrons=False)
        energies, overlaps = sim.get_energies_and_overlaps()
        noninteracting_energies = sim.get_energies_without_interaction()
        self._plot_energies_and_overlaps(
            energies,
            overlaps,
            noninteracting_energies,
            title="Plot of perturbed eigenstate energy levels",
            subplot_lims=subplot_lims,
        )

    def plot_average_material(
        self,
        times,
        average_over=10,
        jitter_electrons=False,
        ax=None,
        initial_occupancy=1,
    ):
        if ax is None:
            fig, ax = plt.subplots()
        sim = self._create_simulation(
            jitter_electrons, initial_occupancy=initial_occupancy
        )
        electron_densities_for_each = sim.simulate_random_system_coherently_for_each(
            times,
            average_over=average_over,
            thermal=True,
            jitter_for_each=jitter_electrons,
        )

        ElectronSimulationPlotter.plot_average_number_against_time_for_each(
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
        return ax.get_figure(), ax

    def plot_average_electron_distribution(
        self,
        times,
        average_over=10,
        jitter_electrons=False,
        ax=None,
    ):
        if ax is None:
            fig, ax = plt.subplots()

        def expected_occupation(mu):
            return sum(
                [
                    1 / (1 + np.exp((energy - mu) / self.boltzmann_energy))
                    for energy in self.electron_energies
                ]
            )

        electron_energy_range = max(self.electron_energies) - min(
            self.electron_energies
        )
        chemical_potential: float = scipy.optimize.brentq(
            lambda x: expected_occupation(x) - self.number_of_electrons,
            max(self.electron_energies) - 2 * electron_energy_range,
            min(self.electron_energies) + 2 * electron_energy_range,
            xtol=electron_energy_range * 10 ** -6,
        )  # type: ignore

        normalised_energies = np.array(self.electron_energies) - chemical_potential

        initial_densities = []
        final_densities = []
        for _ in range(average_over):
            sim = self._create_simulation(jitter_electrons)
            electron_densities_for_each = sim.simulate_random_system_coherently(
                times,
                thermal=True,
            )

            initial_state_densities_for_each = [
                d[0] for d in electron_densities_for_each
            ]
            normalised_initial_state_densities = [
                d if sum(d) == 0 else self.number_of_electrons * d / sum(d)
                for d in initial_state_densities_for_each
            ]
            initial_densities += normalised_initial_state_densities

            final_state_densities_for_each = [d[1] for d in electron_densities_for_each]
            normalised_final_state_densities = [
                d if sum(d) == 0 else self.number_of_electrons * d / sum(d)
                for d in final_state_densities_for_each
            ]
            final_densities += normalised_final_state_densities

        ElectronSimulationPlotter.plot_average_density_against_energy(
            densities=initial_densities,
            energies=normalised_energies,
            ax=ax,
            label="FCC density",
        )

        ElectronSimulationPlotter.plot_average_density_against_energy(
            densities=final_densities,
            energies=normalised_energies,
            ax=ax,
            label="HCP density",
        )
        energies_for_plot = np.linspace(
            normalised_energies[0], normalised_energies[-1], 1000
        )

        ax.plot(
            energies_for_plot,
            ElectronSimulationPlotter.fermi_distribution(
                self.boltzmann_energy,
                energies_for_plot,
            ),
            label="fermi-dirac",
        )
        ax.set_xlabel("Energy / J")
        return ax.get_figure(), ax

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

    def get_initial_electron_densities(self, initial_occupancy=1, average_over=1):
        vals = []
        for _ in range(average_over):
            sim = self._create_simulation(
                jitter_electrons=False, initial_occupancy=initial_occupancy
            )
            vals.append(
                sim.simulate_random_system_coherently(times=[0], thermal=True)[0][0]
            )
        return np.average(vals, axis=0)
