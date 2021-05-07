from typing import Any, Dict, List
from matplotlib import gridspec, colors, colorbar
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import beta
import numpy.typing
import scipy.optimize

from simulation.ElectronSimulation import (
    ElectronSimulation,
    ElectronSimulationConfig,
    randomise_electron_energies,
)


class ElectronSimulationPlotter:
    @staticmethod
    def _filter_data_exponentially(data, decay_factor):
        filtered_data = [data[0]]
        for datapoint in data[1:]:
            new_average = (
                decay_factor * datapoint + (1 - decay_factor) * filtered_data[-1]
            )

            filtered_data.append(new_average)
        return filtered_data

    @staticmethod
    def _plot_varying_density_against_time(densities, times, energies):
        colour_cycle = plt.cm.Spectral(np.linspace(0, 1, len(densities)))  # type: ignore

        gs = gridspec.GridSpec(1, 6)

        ax1 = plt.subplot(gs[0, :-1])
        ax2 = plt.subplot(gs[0, -1])

        ax1.set_prop_cycle("color", colour_cycle)
        for density in densities:
            ax1.plot(energies, density)

        ax1.set_ylabel("Electron Density")
        # ax1.set_ylim([0, 1])
        ax1.set_xlabel("Energy")
        ax1.set_title("Plot of Electron Density against Energy")
        norm = colors.Normalize(vmin=times[0], vmax=times[-1])
        cmap = colors.ListedColormap(colour_cycle)
        cb1 = colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, orientation="vertical")  # type: ignore
        ax2.set_ylabel("Color against time")

        return ax1.get_figure(), ax1

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
    def _calculate_decay_factor(period_of_fluctuations, times):
        time_since_start = times - times[0]
        times_in_first_period = time_since_start <= period_of_fluctuations
        number_of_timesteps_in_first_period = np.sum(times_in_first_period)
        return 1 / number_of_timesteps_in_first_period

    @classmethod
    def _plot_average_number_deviation_against_time(
        cls,
        target_numbers,
        number_in_each_state: Dict[str, List[float]],
        times,
        period_of_noise_fluctuation=0,
    ):
        decay_factor = cls._calculate_decay_factor(period_of_noise_fluctuation, times)

        (fig, ax) = plt.subplots(1)

        for (state_name, numbers) in number_in_each_state.items():
            number_fluctuation = np.abs(np.array(numbers) - target_numbers[state_name])
            filtered_number_fluctuation = cls._filter_data_exponentially(
                number_fluctuation, decay_factor
            )
            ax.plot(times, filtered_number_fluctuation, label=state_name)

        ax.set_title("Plot of Total Electron Density Against Time")
        ax.legend()
        ax.set_ylabel("Total Electron Density")
        # ax1.set_ylim([0, 1])
        ax.set_xlabel("time")
        return (fig, ax)

    @staticmethod
    def plot_average_number_against_time_for_each(
        number_in_each_state_for_each: List[Dict[str, List[float]]], times, ax=None
    ):
        if ax is None:
            fig, ax = plt.subplots(1)

        # Plot average line
        for key in number_in_each_state_for_each[0].keys():
            average = np.average(
                [x[key] for x in number_in_each_state_for_each], axis=0
            )
            ax.plot(times, average, label=f"Average {key}")

        ax.set_ylabel("Total Electron Density")
        # ax1.set_ylim([0, 1])
        ax.set_xlabel("time")
        ax.set_title("Plot of average electron density against time")
        ax.set_xlim(left=0)
        return (ax.get_figure(), ax)

    @staticmethod
    def plot_total_number_against_time_for_each(
        number_in_each_state_for_each: List[Dict[str, List[float]]], times, ax=None
    ):
        if ax is None:
            fig, ax = plt.subplots(1)
        colour_cycle = plt.cm.Spectral(  # type: ignore
            np.linspace(0, 1, len(number_in_each_state_for_each))
        )

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
        return (ax.get_figure(), ax)

    @staticmethod
    def plot_average_density_against_energy(
        densities: List[Any], energies, label="average density", ax=None
    ):
        if ax is None:
            fig, ax = plt.subplots()

        # ax.plot(energies, np.average(densities, axis=0))
        ax.errorbar(
            energies,
            np.average(densities, axis=0),
            yerr=np.std(densities, axis=0),
            label=label,
        )
        ax.set_ylabel("Electron Density")
        ax.set_xlabel("Energy")
        ax.set_title("Plot of Average Electron Density Against Energy")
        return (ax.get_figure(), ax)

    @staticmethod
    def _plot_normalistation_against_time(normalisations, times):
        fig, ax = plt.subplots()
        ax.plot(times, normalisations)
        ax.set_ylabel("Normalisation")
        ax.set_xlabel("Time")
        ax.set_title("Plot of Normalisation against Time")
        return (fig, ax)

    @classmethod
    def _plot_electron_densities(cls, electron_densities, times, energies):
        initially_occupied_densities = [d[0] for d in electron_densities]
        initially_unoccupied_densities = [d[1] for d in electron_densities]

        cls._plot_varying_density_against_time(
            initially_occupied_densities, times, energies
        )
        plt.show()

        fig, ax = cls._plot_varying_density_against_time(
            initially_unoccupied_densities, times, energies
        )
        ax.set_title("Plot of final state density against time")
        plt.show()

        fig, ax = cls._plot_varying_density_against_time(
            [
                a + b
                for (a, b) in zip(
                    initially_occupied_densities, initially_unoccupied_densities
                )
            ],
            times,
            energies,
        )
        ax.set_title("Plot of combined density against time")
        plt.show()

        (fig, ax) = cls.plot_average_density_against_energy(
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

        fig, ax = cls._plot_total_number_against_time(
            number_in_each_state={
                "hcp": [sum(x) for x in initially_unoccupied_densities],
            },
            times=times,
        )
        plt.show()

    @staticmethod
    def calculate_average_density(densities_for_each: numpy.typing.ArrayLike):
        return np.average(densities_for_each, axis=0)

    @staticmethod
    def occupation_curve_fit_function(number_of_electrons, time, omega, decay_time):
        amplitude = number_of_electrons / 2
        return amplitude * np.exp(-time / decay_time) * np.cos(omega * time) + amplitude

    @classmethod
    def _plot_densities_with_fit(
        cls, initially_occupied_densities, initially_unoccupied_densities, times
    ):
        number_of_electrons = initially_occupied_densities[0]
        initial_omega_guess = 100 / (times[-1] - times[0])
        initial_decay_time_guess = (times[-1] - times[0]) / 4

        (optimal_omega, optimal_decay_time), pcov = scipy.optimize.curve_fit(  # type: ignore
            lambda t, w, d: cls.occupation_curve_fit_function(
                number_of_electrons, t, w, d
            ),
            times,
            initially_occupied_densities,
            p0=[initial_omega_guess, initial_decay_time_guess],
        )

        print("omega", optimal_omega)
        print("decay_time", optimal_decay_time)
        print(pcov)

        (fig, ax) = cls._plot_total_number_against_time(
            number_in_each_state={
                "fcc": initially_occupied_densities,
                "hcp": initially_unoccupied_densities,
                "fcc_fit": cls.occupation_curve_fit_function(
                    initially_occupied_densities[0],
                    times,
                    optimal_omega,
                    optimal_decay_time,
                ),
            },
            times=times,
        )
        ax.set_title("Plot of average Electron Density Against Time")
        return (fig, ax)

    @classmethod
    def _plot_electron_densities_for_each(
        cls, electron_densities_for_each, times, energies, period_of_noise_fluctuation=0
    ):
        cls.plot_total_number_against_time_for_each(
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

        average_densities = cls.calculate_average_density(electron_densities_for_each)

        initially_occupied_densities = [d[0] for d in average_densities]
        initially_unoccupied_densities = [d[1] for d in average_densities]

        # (fig, ax) = cls._plot_densities_with_fit(
        #     [sum(x) for x in initially_occupied_densities],
        #     [sum(x) for x in initially_unoccupied_densities],
        #     times,
        # )
        # plt.show()

        # target_number = sum(initially_occupied_densities[0]) / 2
        # (fig, ax) = cls._plot_average_number_deviation_against_time(
        #     target_numbers={key: target_number for key in ["fcc", "hcp"]},
        #     number_in_each_state={
        #         "fcc": [sum(x) for x in initially_occupied_densities],
        #         "hcp": [sum(x) for x in initially_unoccupied_densities],
        #     },
        #     times=times,
        #     period_of_noise_fluctuation=period_of_noise_fluctuation,
        # )
        # ax.set_ylim([0, None])
        # plt.show()

        (fig, ax) = cls.plot_average_density_against_energy(
            initially_occupied_densities, energies
        )
        ax.set_ylim([0, 1])
        ax.set_title("Plot of average Electron Density in the Initially Occupied State")
        plt.show()

        (fig, ax) = cls.plot_average_density_against_energy(
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
        cls,
        sim: ElectronSimulation,
        times: List[float],
        *args,
        period_of_noise_fluctuation=0,
        **kwargs,
    ):
        electron_densities_for_each = sim.simulate_random_system_coherently_for_each(
            times, *args, **kwargs
        )

        cls._plot_electron_densities_for_each(
            electron_densities_for_each,
            times,
            sim.electron_energies,
            period_of_noise_fluctuation,
        )
        return electron_densities_for_each

    @staticmethod
    def fermi_distribution(boltzmann_energy, E):
        return 1 / (1 + np.exp(E / boltzmann_energy))

    @classmethod
    def plot_normalisation_demonstration(
        cls, config: ElectronSimulationConfig, times: list[float]
    ):
        sim = ElectronSimulation(config)
        normalisations = sim.simulate_random_system_normalisations_coherently(
            times, thermal=True
        )

        (fig, ax) = cls._plot_normalistation_against_time(normalisations, times)
        plt.show()

    @classmethod
    def plot_thermal_demonstration_centered_kf(
        cls, config: ElectronSimulationConfig, repeats=10, ax=None
    ):
        if ax is None:
            fig, ax = plt.subplots()

        # Demonstrates the FD Distribution
        single_n_config = ElectronSimulationConfig(
            hbar=config.hbar,
            boltzmann_energy=config.boltzmann_energy,
            electron_energies=config.electron_energies,
            hydrogen_energies=config.hydrogen_energies,
            block_factors=config.block_factors,
            q_prefactor=config.q_prefactor,
            electron_energy_jitter=lambda x: randomise_electron_energies(x, 0.005),
            number_of_electrons=None,
        )
        single_n_electron_densities = []
        for _ in range(repeats):
            sim = ElectronSimulation(single_n_config)
            # print(sim.simulate_random_system_coherently(times=[500000], thermal=True)[
            #             0
            #         ])
            single_n_electron_densities.append(
                np.sum(
                    sim.simulate_random_system_coherently(times=[500000], thermal=True)[
                        0
                    ],
                    axis=0,
                )
            )

        normalised_energies = np.array(config.electron_energies) - np.average(
            config.electron_energies
        )
        cls.plot_average_density_against_energy(
            single_n_electron_densities, normalised_energies.tolist(), ax=ax
        )
        energies_for_plot = np.linspace(
            normalised_energies[0], normalised_energies[-1], 1000
        )
        ax.plot(
            energies_for_plot,
            cls.fermi_distribution(
                config.boltzmann_energy,
                energies_for_plot,
            ),
            label="fermi-dirac",
        )
        ax.set_ylim([0, 1])
        return ax.get_figure(), ax

    @classmethod
    def plot_thermal_corrected_demonstration_centered_kf(
        cls, config: ElectronSimulationConfig, repeats=10
    ):
        energy_spacing = config.electron_energies[1] - config.electron_energies[0]
        # Demonstrates the FD Distribution
        config_for_demo = ElectronSimulationConfig(
            hbar=config.hbar,
            boltzmann_energy=config.boltzmann_energy,
            electron_energies=config.electron_energies,
            hydrogen_energies=config.hydrogen_energies,
            block_factors=config.block_factors,
            q_prefactor=config.q_prefactor,
            electron_energy_jitter=lambda x: randomise_electron_energies(
                x, 0.1 * (energy_spacing)
            ),
            number_of_electrons=None,
        )
        corrected_electron_densities = []
        for _ in range(repeats):
            total_electron_densities = []
            total_thermal_weight = 0
            for n in range(1, 1 + len(config.electron_energies)):
                config_dict = config_for_demo._asdict()
                config_dict["number_of_electrons"] = n
                varying_n_config = ElectronSimulationConfig(**config_dict)
                sim = ElectronSimulation(varying_n_config)
                thermal_weight = 1
                number_of_states = sim.number_of_electron_basis_states
                total_thermal_weight += thermal_weight * number_of_states
                total_electron_densities.append(
                    thermal_weight
                    * number_of_states
                    * sim.simulate_random_system_coherently(times=[0], thermal=True)[0][
                        0
                    ]
                )
            corrected_electron_densities.append(
                np.sum(total_electron_densities, axis=0) / (total_thermal_weight)
            )
        normalised_energies = np.array(config.electron_energies) - np.average(
            config.electron_energies
        )
        (fig, ax) = cls.plot_average_density_against_energy(
            corrected_electron_densities, normalised_energies.tolist()
        )
        energies_for_plot = np.linspace(
            normalised_energies[0], normalised_energies[-1], 1000
        )
        ax.plot(
            energies_for_plot,
            cls.fermi_distribution(
                config.boltzmann_energy,
                energies_for_plot,
            ),
            label="fermi-dirac",
        )
        ax.set_ylim([0, 1])
        plt.show()

    @classmethod
    def plot_thermal_demonstration_off_center(
        cls, config: ElectronSimulationConfig, dominant_occupation, repeats=10, ax=None
    ):
        if ax is None:
            fig, ax = plt.subplots()
        energy_spacing = config.electron_energies[1] - config.electron_energies[0]
        single_n_config = ElectronSimulationConfig(
            hbar=config.hbar,
            boltzmann_energy=config.boltzmann_energy,
            electron_energies=config.electron_energies,
            hydrogen_energies=config.hydrogen_energies,
            block_factors=config.block_factors,
            q_prefactor=config.q_prefactor,
            electron_energy_jitter=lambda x: randomise_electron_energies(
                x, 0.1 * (energy_spacing)
            ),
            number_of_electrons=dominant_occupation,
        )
        single_n_electron_densities = []
        for _ in range(repeats):
            sim = ElectronSimulation(single_n_config)
            single_n_electron_densities.append(
                sim.simulate_random_system_coherently(times=[0], thermal=True)[0][0]
            )
        # Assuming closely spaced- not true here
        central_energy_incorrect = config.boltzmann_energy * np.log(
            dominant_occupation / (len(config.electron_energies) - dominant_occupation)
        )

        def expected_occupation(mu):
            return sum(
                [
                    1 / (1 + np.exp((energy - mu) / config.boltzmann_energy))
                    for energy in config.electron_energies
                ]
            )

        chemical_potential: float = scipy.optimize.brentq(
            lambda x: expected_occupation(x) - dominant_occupation,
            0,
            2 * config.electron_energies[-1],
        )  # type: ignore
        print(
            chemical_potential,
            dominant_occupation,
            expected_occupation(chemical_potential),
        )

        normalised_energies = np.array(config.electron_energies) - chemical_potential
        cls.plot_average_density_against_energy(
            single_n_electron_densities,
            normalised_energies.tolist(),
            ax=ax,
            label="average density",
        )
        energies_for_plot = np.linspace(
            normalised_energies[0], normalised_energies[-1], 1000
        )
        ax.plot(
            energies_for_plot,
            cls.fermi_distribution(
                config.boltzmann_energy,
                energies_for_plot,
            ),
            label="fermi-dirac",
        )
        ax.set_ylim([0, 1])
        return (ax.get_figure(), ax)

    @classmethod
    def plot_thermal_demonstration_off_center_corrected(
        cls, config: ElectronSimulationConfig, dominant_occupation, repeats=10, ax=None
    ):
        if ax is None:
            fig, ax = plt.subplots()

        def expected_occupation(mu):
            return sum(
                [
                    1 / (1 + np.exp(config.boltzmann_energy * (energy - mu)))
                    for energy in config.electron_energies
                ]
            )

        chemical_potential: float = scipy.optimize.brentq(
            lambda x: expected_occupation(x) - dominant_occupation,
            0,
            2 * config.electron_energies[-1],
        )  # type: ignore
        print(
            chemical_potential,
            dominant_occupation,
            expected_occupation(chemical_potential),
        )

        normalised_energies = np.array(config.electron_energies) - chemical_potential

        # Corrected Plot
        corrected_electron_densities = []
        for _ in range(repeats):
            total_electron_densities = []
            total_thermal_weight = 0
            for number_of_electrons in range(1, 1 + len(config.electron_energies)):
                varying_n_config = ElectronSimulationConfig(
                    hbar=config.hbar,
                    boltzmann_energy=config.boltzmann_energy,
                    electron_energies=config.electron_energies,
                    hydrogen_energies=config.hydrogen_energies,
                    block_factors=config.block_factors,
                    q_prefactor=config.q_prefactor,
                    electron_energy_jitter=lambda x: x,
                    number_of_electrons=number_of_electrons,
                )
                sim = ElectronSimulation(varying_n_config)
                thermal_weight = np.exp(
                    -(chemical_potential * number_of_electrons)
                    / (config.boltzmann_energy)
                )
                thermal_weight = 1
                number_of_states = sim.number_of_electron_basis_states
                # number_of_states = 1
                total_thermal_weight += thermal_weight * number_of_states
                total_electron_densities.append(
                    thermal_weight
                    * number_of_states
                    * sim.simulate_random_system_coherently(times=[0], thermal=True)[0][
                        0
                    ]
                )

            corrected_electron_densities.append(
                np.sum(total_electron_densities, axis=0) / (total_thermal_weight)
            )

        cls.plot_average_density_against_energy(
            corrected_electron_densities, normalised_energies.tolist(), ax=ax
        )
        energies_for_plot = np.linspace(
            normalised_energies[0], normalised_energies[-1], 1000
        )
        ax.plot(
            energies_for_plot,
            cls.fermi_distribution(
                config.boltzmann_energy,
                energies_for_plot,
            ),
            label="fermi-dirac",
        )
        return (ax.get_figure, ax)


def plot_average_densities_example():
    config = ElectronSimulationConfig(
        hbar=1,
        boltzmann_energy=1,
        electron_energies=np.linspace(0, 100, 6).tolist(),
        hydrogen_energies=[0, 0],
        block_factors=[[1, 0.001], [0.001, 1]],
        q_prefactor=1,
    )
    simulator = ElectronSimulation(config)

    ElectronSimulationPlotter.plot_random_system_evolved_coherently(
        simulator,
        np.linspace(0, 10000, 1000).tolist(),
    )

    ElectronSimulationPlotter.plot_average_densities_of_system_evolved_coherently(
        simulator,
        np.linspace(0, 10000, 1000).tolist(),
        average_over=100,
    )


def plot_normalisation_example():
    config = ElectronSimulationConfig(
        hbar=1,
        boltzmann_energy=10,
        electron_energies=np.linspace(0, 100, 8).tolist(),
        hydrogen_energies=[0, 0],
        block_factors=[[1, 0.001], [0.001, 1]],
        q_prefactor=1,
        electron_energy_jitter=lambda x: x,
    )
    ElectronSimulationPlotter.plot_normalisation_demonstration(
        config, times=np.linspace(0, 1000, 1000).tolist()
    )


def plot_thermal_example():
    # Low temp 10
    # High temp 30
    config = ElectronSimulationConfig(
        hbar=1,
        boltzmann_energy=10,
        electron_energies=np.linspace(0, 100, 8).tolist(),
        hydrogen_energies=[0, 0],
        block_factors=[[1, 0.001], [0.001, 1]],
        q_prefactor=1,
        electron_energy_jitter=lambda x: x,
    )
    simulator = ElectronSimulation(config)

    ElectronSimulationPlotter.plot_random_system_evolved_coherently(
        simulator,
        times=np.linspace(0, 10, 1000).tolist(),
        thermal=True,
    )

    ElectronSimulationPlotter.plot_average_densities_of_system_evolved_coherently(
        simulator,
        times=np.linspace(0, 40000, 1000).tolist(),
        average_over=20,
        thermal=True,
        jitter_for_each=True,
    )


def thermal_energy_investigation_centered_kf():
    config = ElectronSimulationConfig(
        hbar=1,
        boltzmann_energy=1,
        electron_energies=np.linspace(0, 10, 8).tolist(),
        hydrogen_energies=[0, 0],
        block_factors=[[0, 0], [0, 0]],
        q_prefactor=1,
        electron_energy_jitter=lambda x: x,
    )
    fig, ax = plt.subplots()
    ElectronSimulationPlotter.plot_thermal_demonstration_centered_kf(
        config, repeats=1000, ax=ax
    )
    ax.set_title(
        "Thermal demonstration with 5 electrons and 10 States"
        "\nshowing the correct fermi-dirac distribution"
    )
    ax.legend()
    plt.show()

    config = ElectronSimulationConfig(
        hbar=1,
        boltzmann_energy=1,
        electron_energies=np.linspace(0, 10, 8).tolist(),
        hydrogen_energies=[0, 0],
        block_factors=[[1, 0.001], [0.001, 1]],
        q_prefactor=1,
        electron_energy_jitter=lambda x: x,
    )
    fig, ax = plt.subplots()
    ElectronSimulationPlotter.plot_thermal_demonstration_centered_kf(
        config, repeats=1000, ax=ax
    )
    ax.set_title(
        "Thermal demonstration with 5 electrons and 10 States"
        + "\nshowing the incorrect fermi-dirac distribution"
    )
    ax.legend()
    plt.show()

    config = ElectronSimulationConfig(
        hbar=1,
        boltzmann_energy=1,
        electron_energies=np.linspace(0, 10, 8).tolist(),
        hydrogen_energies=[0, 0],
        block_factors=[[1, 0.001], [0.001, 1]],
        q_prefactor=0.1,
        electron_energy_jitter=lambda x: x,
    )
    fig, ax = plt.subplots()
    ElectronSimulationPlotter.plot_thermal_demonstration_centered_kf(
        config, repeats=1000, ax=ax
    )
    ax.set_title(
        "Thermal demonstration with 5 electrons and 10 States"
        + "\nwith a small interaction"
    )
    ax.legend()
    fig.tight_layout()
    plt.show()

    ElectronSimulationPlotter.plot_thermal_corrected_demonstration_centered_kf(
        config, repeats=1000
    )


def thermal_energy_investigation_off_center():
    config = ElectronSimulationConfig(
        hbar=1,
        boltzmann_energy=10,
        electron_energies=np.linspace(0, 100, 10).tolist(),
        hydrogen_energies=[0, 0],
        block_factors=[[1, 0.001], [0.001, 1]],
        q_prefactor=1,
        electron_energy_jitter=lambda x: x,
    )
    fig, ax = plt.subplots()
    ElectronSimulationPlotter.plot_thermal_demonstration_off_center(
        config, dominant_occupation=4, repeats=1000, ax=ax
    )
    ax.set_title(
        "Thermal demonstration with 3 electrons and 10 States"
        + "\nshowing the correct fermi-dirac distribution"
    )
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # plot_state_overlaps_example()
    # plot_average_densities_example()
    # plot_normalisation_example()
    thermal_energy_investigation_centered_kf()
    thermal_energy_investigation_off_center()
