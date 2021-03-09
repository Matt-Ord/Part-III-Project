from __future__ import annotations

from typing import Dict, List, NamedTuple
from simulation.Hamiltonian import Hamiltonian
import numpy as np
import matplotlib.pyplot as plt
from simulation.ElectronSystem import ElectronSystem, ElectronSystemUtil
import matplotlib as mpl


class ElectronSimulatorConfig(NamedTuple):
    hbar: float
    electron_mass: float


class ElectronSimulator():

    def __init__(self, config: ElectronSimulatorConfig) -> None:
        self.config = config

    def _calculate_electron_energies(self, k_states):
        return ((self.config.hbar * np.array(k_states)) ** 2 /
                (2 * self.config.electron_mass))

    @staticmethod
    def get_electron_densities(electron_states):
        electron_densities = [state.get_electron_density_for_each_hydrogen()
                              for state in electron_states]
        return electron_densities

    @staticmethod
    def get_electron_states(
        initial_system: ElectronSystem,
        hamiltonian: Hamiltonian,
        times: List[float],
        hbar: float,
    ):
        evolved_states = [initial_system.evolve_system(
            hamiltonian,
            time,
            hbar
        )
            for time in times]
        return evolved_states

    @staticmethod
    def get_electron_states_decoherently(
        initial_system: ElectronSystem,
        hamiltonian: Hamiltonian,
        times: List[float],
        hbar: float,
    ):
        evolved_states = [initial_system.evolve_system(
            hamiltonian,
            time,
            hbar
        )
            for time in times]

        timesteps = [end - start for (start, end)
                     in zip(times[:-1], times[1:])]

        evolved_states = [initial_system]
        for t in timesteps:
            evolved_states.append(
                evolved_states[-1].evolve_system_decoherently(
                    hamiltonian,
                    t,
                    hbar
                ))
        return evolved_states

    def create_hamiltonian(
            self,
            initial_system: ElectronSystem,
            k_states,
            block_factors,
            q_factor=lambda x: 1
    ):
        electron_energies = self._calculate_electron_energies(k_states)

        kinetic_hamiltonian = ElectronSystemUtil\
            .given(initial_system)\
            .create_kinetic(Hamiltonian, electron_energies, [0, 0])

        interaction_hamiltonian = ElectronSystemUtil\
            .given(initial_system)\
            .create_q_dependent_interaction(
                Hamiltonian,
                block_factors,
                k_states,
                q_factor)

        print('kinetic_energy', kinetic_hamiltonian[0, 0])
        print('interaction_energy', interaction_hamiltonian[0, 0])

        hamiltonian = kinetic_hamiltonian + interaction_hamiltonian
        np.savetxt("hamiltonian.csv",
                   hamiltonian._matrix_representation, delimiter=",")
        return hamiltonian

    @classmethod
    def _plot_electron_densities(cls, electron_densities, times):
        initially_occupied_densities = [d[0] for d in electron_densities]
        initially_unoccupied_densities = [d[1] for d in electron_densities]

        cls.plot_varying_density(initially_occupied_densities, times)
        cls.plot_varying_density(initially_unoccupied_densities, times)
        cls.plot_average_density(initially_occupied_densities, times)
        cls.plot_total_number(
            number_in_each_state={
                'fcc': [sum(x)
                        for x in initially_occupied_densities],
                'hcp': [sum(x)
                        for x in initially_unoccupied_densities],
            },
            times=times
        )

    @staticmethod
    def plot_varying_density(densities, times):
        colour_cycle = plt.cm.Spectral(
            np.linspace(0, 1, len(densities)))

        gs = mpl.gridspec.GridSpec(1, 6)

        ax1 = plt.subplot(gs[0, :-1])
        ax2 = plt.subplot(gs[0, -1])

        ax1.set_prop_cycle('color', colour_cycle)
        for density in densities:
            ax1.plot(density)

        ax1.set_ylabel('Electron Density')
        ax1.set_ylim([0, 1])
        ax1.set_xlabel('k_vector')
        ax1.set_title('Plot of electron density against k')
        norm = mpl.colors.Normalize(vmin=times[0], vmax=times[-1])
        cmap = mpl.colors.ListedColormap(colour_cycle)
        cb1 = mpl.colorbar.ColorbarBase(ax2,
                                        cmap=cmap,
                                        norm=norm,
                                        orientation='vertical')
        ax2.set_ylabel('Color against time')

        plt.show()

    @staticmethod
    def plot_total_number(number_in_each_state: Dict[str, List[float]], times):

        (fig, ax) = plt.subplots(1)

        for (state_name, numbers) in number_in_each_state.items():
            ax.plot(times, numbers, label=state_name)

        ax.legend()
        ax.set_ylabel('Total Electron Density')
        # ax1.set_ylim([0, 1])
        ax.set_xlabel('time')
        ax.set_title('Plot of electron density against time ')
        ax.set_xlim(left=0)
        plt.show()

    @staticmethod
    def plot_average_density(densities, times):
        fig, ax = plt.subplots()
        ax.plot(np.average(densities, axis=0))
        ax.set_ylabel('Electron Density')
        ax.set_xlabel('k_vector')
        ax.set_ylim([0, 1])
        ax.set_title('Plot of Average electron density against k')
        plt.show()

    def _setup_explicit_initial_state(self, k_states):
        initial_electron_state_vector = np.zeros_like(k_states)
        initial_electron_state_vector[:
                                      int(len(k_states) / 2)] = 1

        initial_state = ElectronSystemUtil.create_explicit(
            ElectronSystem,
            initial_electron_state_vector,
            0)
        return initial_state

    def _setup_random_initial_state(self, k_states):
        number_of_electron_states = len(k_states)
        number_of_electrons = int(number_of_electron_states / 2)
        hydrogen_state = 0

        initial_state = ElectronSystemUtil.create_random(
            ElectronSystem,
            number_of_electron_states,
            number_of_electrons,
            hydrogen_state
        )
        return initial_state

    def simulate_system_coherently(
            self,
            k_states: List[float],
            times: List[float],
            block_factors: List[List[float]] = [[0, 0], [0, 0]],
            q_factor=lambda x: 1
    ):
        initial_state = self._setup_explicit_initial_state(
            k_states
        )

        hamiltonian = self.create_hamiltonian(
            initial_state,
            k_states,
            block_factors,
            q_factor
        )

        electron_densities = self.get_electron_densities(
            self.get_electron_states(
                initial_state,
                hamiltonian,
                times,
                self.config.hbar
            )
        )

        self._plot_electron_densities(electron_densities, times)

    def simulate_random_system_coherently(
            self,
            k_states: List[float],
            times: List[float],
            block_factors: List[List[float]] = [[0, 0], [0, 0]],
            q_factor=lambda x: 1
    ):
        initial_state = self._setup_random_initial_state(
            k_states
        )

        hamiltonian = self.create_hamiltonian(
            initial_state,
            k_states,
            block_factors,
            q_factor
        )

        electron_densities = self.get_electron_densities(
            self.get_electron_states(
                initial_state,
                hamiltonian,
                times,
                self.config.hbar
            )
        )

        self._plot_electron_densities(electron_densities, times)

    def simulate_system_decoherently(
            self,
            k_states,
            times,
            block_factors=[[0, 0], [0, 0]],
            q_factor=lambda x: 1
    ):
        initial_state = self._setup_explicit_initial_state(
            k_states
        )

        hamiltonian = self.create_hamiltonian(
            initial_state,
            k_states,
            block_factors,
            q_factor
        )

        electron_densities = self.get_electron_densities(
            self.get_electron_states_decoherently(
                initial_state,
                hamiltonian,
                times,
                self.config.hbar
            )
        )

        self._plot_electron_densities(electron_densities, times)

    def simulate_uncoupled_system_coherently(
        self,
        k_states,
        times,
        self_interaction=-1,
        q_factor=lambda x: 1
    ):
        block_factors = [
            [self_interaction, 0],
            [0, self_interaction]]

        self.simulate_system_coherently(
            k_states,
            times,
            block_factors,
            q_factor
        )


if __name__ == '__main__':
    config = ElectronSimulatorConfig(1, 1)
    simulator = ElectronSimulator(config)
    # simulator.simulate_uncoupled_system(
    #     np.linspace(1, 4, 4),
    #     np.linspace(1000, 500000, 100)
    # )

    # simulator.simulate_coupled_system(
    #     np.linspace(10, 13, 4),
    #     np.linspace(1000000, 5000000, 20)
    # )

    simulator.simulate_random_system_coherently(
        np.linspace(0, 0.01, 6),
        np.linspace(0, 7500, 1000),
        block_factors=[[1, 0.001j], [-0.001j, 1]]
    )
