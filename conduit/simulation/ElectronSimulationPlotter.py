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
    def _plot_varying_density_against_time(densities, times, energies):
        colour_cycle = plt.cm.Spectral(np.linspace(0, 1, len(densities)))

        gs = mpl.gridspec.GridSpec(1, 6)

        ax1 = plt.subplot(gs[0, :-1])
        ax2 = plt.subplot(gs[0, -1])

        ax1.set_prop_cycle("color", colour_cycle)
        for density in densities:
            ax1.plot(energies, density)

        ax1.set_ylabel("Electron Density")
        # ax1.set_ylim([0, 1])
        ax1.set_xlabel("Energy")
        ax1.set_title("Plot of Electron Density against Energy")
        norm = mpl.colors.Normalize(vmin=times[0], vmax=times[-1])
        cmap = mpl.colors.ListedColormap(colour_cycle)
        cb1 = mpl.colorbar.ColorbarBase(
            ax2, cmap=cmap, norm=norm, orientation="vertical"
        )
        ax2.set_ylabel("Color against time")

        plt.show()

    @staticmethod
    def _plot_total_number_against_time(
        number_in_each_state: Dict[str, List[float]], times
    ):

        (fig, ax) = plt.subplots(1)

        for (state_name, numbers) in number_in_each_state.items():
            ax.plot(times, numbers, label=state_name)

        ax.set_title("Plot of Total Electron Density Against Time")
        ax.legend()
        ax.set_ylabel("Total Electron Density")
        # ax1.set_ylim([0, 1])
        ax.set_xlabel("time")
        ax.set_xlim(left=0)
        return (fig, ax)

    @staticmethod
    def _plot_total_number_against_time_for_each(
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
        return (fig, ax)

    @staticmethod
    def _plot_average_density_against_energy(densities, energies):
        fig, ax = plt.subplots()
        ax.plot(energies, np.average(densities, axis=0))
        ax.set_ylabel("Electron Density")
        ax.set_xlabel("Energy")
        ax.set_title("Plot of Average Electron Density Against Energy")
        return (fig, ax)

    @classmethod
    def _plot_electron_densities(cls, electron_densities, times, energies):
        initially_occupied_densities = [d[0] for d in electron_densities]
        initially_unoccupied_densities = [d[1] for d in electron_densities]

        cls._plot_varying_density_against_time(
            initially_occupied_densities, times, energies
        )
        plt.show()

        cls._plot_varying_density_against_time(
            initially_unoccupied_densities, times, energies
        )
        plt.show()

        (fig, ax) = cls._plot_average_density_against_energy(
            initially_occupied_densities, energies
        )
        ax.set_ylim([0, 1])
        plt.show()

        cls._plot_total_number_against_time(
            number_in_each_state={
                "fcc": [sum(x) for x in initially_occupied_densities],
                "hcp": [sum(x) for x in initially_unoccupied_densities],
            },
            times=times,
        )
        plt.show()

    @staticmethod
    def _calculate_average_density(densities_for_each):
        return np.average(densities_for_each, axis=0)

    @classmethod
    def _plot_electron_densities_for_each(
        cls, electron_densities_for_each, times, energies
    ):
        cls._plot_total_number_against_time_for_each(
            [
                {
                    "fcc": [sum(x) for x in density[:, 0]],
                    "hcp": [sum(x) for x in density[:, 1]],
                }
                for density in electron_densities_for_each
            ],
            times,
        )
        plt.show()

        average_densities = cls._calculate_average_density(electron_densities_for_each)

        initially_occupied_densities = [d[0] for d in average_densities]
        initially_unoccupied_densities = [d[1] for d in average_densities]

        (fig, ax) = cls._plot_total_number_against_time(
            number_in_each_state={
                "fcc": [sum(x) for x in initially_occupied_densities],
                "hcp": [sum(x) for x in initially_unoccupied_densities],
            },
            times=times,
        )
        ax.set_title("Plot of average Electron Density Against Time")
        plt.show()

        (fig, ax) = cls._plot_average_density_against_energy(
            initially_occupied_densities, energies
        )
        ax.set_ylim([0, 1])
        ax.set_title("Plot of average Electron Density in the Initially Occupied State")
        plt.show()

        (fig, ax) = cls._plot_average_density_against_energy(
            initially_unoccupied_densities, energies
        )
        ax.set_ylim([0, None])
        ax.set_title("Plot of average Electron Density in Initially Unoccupied State")
        plt.show()

    @classmethod
    def plot_system_evolved_coherently(
        cls, sim: ElectronSimulation, times: List[float], *args, **kwargs
    ):
        electron_densities = sim.simulate_system_coherently(times, *args, **kwargs)

        cls._plot_electron_densities(electron_densities, times, sim.electron_energies)

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

        cls._plot_electron_densities(electron_densities, times, sim.electron_energies)

    @classmethod
    def plot_system_evolved_decoherently(
        cls, sim: ElectronSimulation, times, *args, **kwargs
    ):
        electron_densities = sim.simulate_system_decoherently(times, *args, **kwargs)

        cls._plot_electron_densities(electron_densities, times, sim.electron_energies)

    @classmethod
    def plot_average_densities_of_system_evolved_coherently(
        cls, sim: ElectronSimulation, times: List[float], *args, **kwargs
    ):
        electron_densities_for_each = sim.simulate_random_system_coherently_for_each(
            times, *args, **kwargs
        )

        cls._plot_electron_densities_for_each(
            electron_densities_for_each, times, sim.electron_energies
        )

    @classmethod
    def plot_tunneling_overlaps(
        cls,
        sim: ElectronSimulation,
    ):
        overlaps = sim.characterise_tunnelling_overlaps()

        print(overlaps)


def plot_average_densities_example():
    config = ElectronSimulationConfig(
        hbar=1,
        boltzmann_energy=1,
        electron_energies=np.linspace(0, 100, 6),
        hydrogen_energies=[0, 0],
        block_factors=[[1, 0.001], [0.001, 1]],
        q_prefactor=1,
    )
    simulator = ElectronSimulation(config)

    ElectronSimulationPlotter.plot_random_system_evolved_coherently(
        simulator,
        np.linspace(0, 10000, 1000),
    )

    ElectronSimulationPlotter.plot_average_densities_of_system_evolved_coherently(
        simulator,
        np.linspace(0, 10000, 1000),
        average_over=100,
    )


def plot_thermal_example():
    # Low temp 10
    # High temp 30
    config = ElectronSimulationConfig(
        hbar=1,
        boltzmann_energy=10,
        electron_energies=np.linspace(0, 100, 8),
        hydrogen_energies=[0, 0],
        temperature=30,
        block_factors=[[1, 0.001], [0.001, 1]],
        q_prefactor=1,
        electron_energy_jitter=4,
    )
    simulator = ElectronSimulation(config)

    ElectronSimulationPlotter.plot_random_system_evolved_coherently(
        simulator,
        times=np.linspace(0, 10, 1000),
        thermal=True,
    )

    ElectronSimulationPlotter.plot_average_densities_of_system_evolved_coherently(
        simulator,
        times=np.linspace(0, 40000, 1000),
        average_over=20,
        thermal=True,
        jitter_for_each=True,
    )


def plot_state_overlaps_example():
    config = ElectronSimulationConfig(
        hbar=1,
        electron_energies=np.linspace(0, 100, 4),
        hydrogen_energies=[0, 0],
        temperature=200,
        block_factors=[[1, 0.001], [0.001, 1]],
        q_prefactor=1,
    )
    simulator = ElectronSimulation(config)

    ElectronSimulationPlotter.plot_tunneling_overlaps(simulator)


if __name__ == "__main__":
    # plot_state_overlaps_example()
    # plot_average_densities_example()
    plot_thermal_example()
