from __future__ import annotations
from Hamiltonian import Hamiltonian, HamiltonianUtil
import numpy as np
import matplotlib.pyplot as plt
from ElectronSystem import ElectronSystem, ElectronSystemUtil
import matplotlib as mpl


class ElectronSimulator():

    @ staticmethod
    def _calculate_k_energies(k_states):
        return np.array(k_states) ** 2

    @staticmethod
    def get_electron_densities(initial_state, hamiltonian, times):
        evolved_states = [initial_state.evolve_system(
            hamiltonian,
            time)
            for time in times]
        electron_densities = [state.get_electron_density_for_each_hydrogen()
                              for state in evolved_states]
        return electron_densities

    @staticmethod
    def create_hamiltonian(
            initial_system: ElectronSystem,
            k_states,
            block_factors,
            q_factor=lambda x: 1):
        electron_energies = k_states ** 2

        kinetic_hamiltonian = ElectronSystemUtil\
            .given(initial_system)\
            .create_kinetic(Hamiltonian, electron_energies, [10, 10.1])

        interaction_hamiltonian = ElectronSystemUtil\
            .given(initial_system)\
            .create_q_dependent_interaction(
                Hamiltonian,
                block_factors,
                k_states,
                q_factor)

        hamiltonian = kinetic_hamiltonian + interaction_hamiltonian
        return hamiltonian

    @staticmethod
    def plot_varying_density(densities, times):
        colour_cycle = plt.cm.Spectral(
            np.linspace(0, 1, len(densities)))

        gs = mpl.gridspec.GridSpec(1, 6)
        # fig, (ax1, ax2) = plt.subplots(1, 2)
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
    def plot_average_density(densities, times):
        fig, ax = plt.subplots()
        ax.plot(np.average(densities, axis=0))
        ax.set_ylabel('Electron Density')
        ax.set_xlabel('k_vector')
        ax.set_ylim([0, 1])
        ax.set_title('Plot of Average electron density against k')
        plt.show()

    @ classmethod
    def simulate_coupled_system(cls, k_states, times, q_factor=lambda x: 1):
        initial_electron_state_vector = np.zeros_like(k_states)
        initial_electron_state_vector[:
                                      int(len(k_states) / 2)] = 1

        initial_state = ElectronSystemUtil.create(
            ElectronSystem,
            initial_electron_state_vector,
            0)

        hamiltonian = cls.create_hamiltonian(
            initial_state,
            k_states,
            [[0, 0.1], [0.1, 0]],
            q_factor)

        electron_densities = cls.get_electron_densities(
            initial_state,
            hamiltonian,
            times)

        initially_occupied_densities = [d[0] for d in electron_densities]
        initially_unoccupied_densities = [d[1] for d in electron_densities]

        cls.plot_varying_density(initially_occupied_densities, times)
        cls.plot_varying_density(initially_unoccupied_densities, times)
        cls.plot_average_density(initially_occupied_densities, times)

    @ classmethod
    def simulate_uncoupled_system(cls, k_states, times):

        initial_electron_state_vector = np.zeros_like(k_states)
        initial_electron_state_vector[:
                                      int(len(k_states) / 2)] = 1

        initial_state = ElectronSystemUtil.create(
            ElectronSystem,
            initial_electron_state_vector,
            0)

        hamiltonian = cls.create_hamiltonian(
            initial_state, k_states, [[-1, 0], [0, -1]])

        electron_densities = cls.get_electron_densities(
            initial_state,
            hamiltonian,
            times)

        occupied_densities = [d[0] for d in electron_densities]
        unoccupied_densities = [d[1] for d in electron_densities]

        cls.plot_varying_density(occupied_densities, times)
        cls.plot_average_density(occupied_densities, times)


if __name__ == '__main__':
    # ElectronSimulator.simulate_uncoupled_system(
    #     np.linspace(1, 4, 4),
    #     np.linspace(1000, 500000, 100)
    # )

    # ElectronSimulator.simulate_coupled_system(
    #     np.linspace(10, 13, 4),
    #     np.linspace(1000000, 5000000, 20)
    # )

    ElectronSimulator.simulate_coupled_system(
        np.linspace(10, 13, 4),
        np.linspace(0, 20, 80),
    )
