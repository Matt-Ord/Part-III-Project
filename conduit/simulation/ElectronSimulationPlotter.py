from typing import Dict, List
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from simulation.ElectronSimulation import (
    ElectronSimulation,
    ElectronSimulationConfig,
)


class ElectronSimulationPlotter:
    @staticmethod
    def _plot_varying_density(densities, times):
        colour_cycle = plt.cm.Spectral(np.linspace(0, 1, len(densities)))

        gs = mpl.gridspec.GridSpec(1, 6)

        ax1 = plt.subplot(gs[0, :-1])
        ax2 = plt.subplot(gs[0, -1])

        ax1.set_prop_cycle("color", colour_cycle)
        for density in densities:
            ax1.plot(density)

        ax1.set_ylabel("Electron Density")
        ax1.set_ylim([0, 1])
        ax1.set_xlabel("k_vector")
        ax1.set_title("Plot of electron density against k")
        norm = mpl.colors.Normalize(vmin=times[0], vmax=times[-1])
        cmap = mpl.colors.ListedColormap(colour_cycle)
        cb1 = mpl.colorbar.ColorbarBase(
            ax2, cmap=cmap, norm=norm, orientation="vertical"
        )
        ax2.set_ylabel("Color against time")

        plt.show()

    @staticmethod
    def _plot_total_number(number_in_each_state: Dict[str, List[float]], times):

        (fig, ax) = plt.subplots(1)

        for (state_name, numbers) in number_in_each_state.items():
            ax.plot(times, numbers, label=state_name)

        ax.legend()
        ax.set_ylabel("Total Electron Density")
        # ax1.set_ylim([0, 1])
        ax.set_xlabel("time")
        ax.set_title("Plot of electron density against time ")
        ax.set_xlim(left=0)
        plt.show()

    @staticmethod
    def _plot_total_number_for_each(
        number_in_each_state_for_each: List[Dict[str, List[float]]],
        times,
    ):
        colour_cycle = plt.cm.Spectral(
            np.linspace(0, 1, len(number_in_each_state_for_each))
        )
        (fig, ax) = plt.subplots(1)

        # Plot individual lines
        for i, number_in_each_state in enumerate(number_in_each_state_for_each):
            for (state_name, numbers) in number_in_each_state.items():
                ax.plot(times, numbers, label=state_name, color=colour_cycle[i])

        # Plot average line
        for key in number_in_each_state_for_each[0].keys():
            average = np.average(
                [x[key] for x in number_in_each_state_for_each], axis=0
            )
            ax.plot(times, average, label=f"Average {key}", color="black")

        ax.set_ylabel("Total Electron Density")
        # ax1.set_ylim([0, 1])
        ax.set_xlabel("time")
        ax.set_title("Plot of electron density against time ")
        ax.set_xlim(left=0)
        plt.show()

    @staticmethod
    def _plot_average_density(densities, times):
        fig, ax = plt.subplots()
        ax.plot(np.average(densities, axis=0))
        ax.set_ylabel("Electron Density")
        ax.set_xlabel("k_vector")
        ax.set_ylim([0, 1])
        ax.set_title("Plot of Average electron density against k")
        plt.show()

    @classmethod
    def _plot_electron_densities(cls, electron_densities, times):
        initially_occupied_densities = [d[0] for d in electron_densities]
        initially_unoccupied_densities = [d[1] for d in electron_densities]

        cls._plot_varying_density(initially_occupied_densities, times)
        cls._plot_varying_density(initially_unoccupied_densities, times)
        cls._plot_average_density(initially_occupied_densities, times)
        cls._plot_total_number(
            number_in_each_state={
                "fcc": [sum(x) for x in initially_occupied_densities],
                "hcp": [sum(x) for x in initially_unoccupied_densities],
            },
            times=times,
        )

    @staticmethod
    def _calculate_average_density(densities_for_each):
        return np.average(densities_for_each, axis=0)

    @classmethod
    def _plot_electron_densities_for_each(cls, electron_densities_for_each, times):
        cls._plot_total_number_for_each(
            [
                {
                    "fcc": [sum(x) for x in density[:, 0]],
                    "hcp": [sum(x) for x in density[:, 1]],
                }
                for density in electron_densities_for_each
            ],
            times,
        )

        average_densities = cls._calculate_average_density(electron_densities_for_each)

        initially_occupied_densities = [d[0] for d in average_densities]
        initially_unoccupied_densities = [d[1] for d in average_densities]

        cls._plot_total_number(
            number_in_each_state={
                "fcc": [sum(x) for x in initially_occupied_densities],
                "hcp": [sum(x) for x in initially_unoccupied_densities],
            },
            times=times,
        )

    @classmethod
    def plot_system_evolved_coherently(
        cls,
        sim: ElectronSimulation,
        times: List[float],
    ):
        electron_densities = sim.simulate_system_coherently(times)

        cls._plot_electron_densities(electron_densities, times)

    @classmethod
    def plot_random_system_evolved_coherently(
        cls,
        sim: ElectronSimulation,
        times: List[float],
        *args,
        **kwargs,
    ):

        electron_densities = sim.simulate_random_system_coherently(
            times, *args, **kwargs
        )

        cls._plot_electron_densities(electron_densities, times)

    @classmethod
    def plot_system_evolved_decoherently(
        cls, sim: ElectronSimulation, times, *args, **kwargs
    ):
        electron_densities = sim.simulate_system_decoherently(times, *args, **kwargs)

        cls._plot_electron_densities(electron_densities, times)

    @classmethod
    def plot_average_densities_of_system_evolved_coherently(
        cls,
        sim: ElectronSimulation,
        times: List[float],
        average_over=50,
    ):
        electron_densities_for_each = sim.simulate_random_system_coherently_for_each(
            times,
            average_over,
        )

        cls._plot_electron_densities_for_each(electron_densities_for_each, times)


if __name__ == "__main__":
    config = ElectronSimulationConfig(
        hbar=1,
        electron_energies=np.linspace(0, 100, 8),
        hydrogen_energies=[0, 0],
        block_factors=[[1, 0.001], [0.001, 1]],
        q_prefactor=1,
    )
    simulator = ElectronSimulation(config)

    ElectronSimulationPlotter.plot_average_densities_of_system_evolved_coherently(
        simulator,
        np.linspace(0, 10000, 1000),
        average_over=100,
    )
