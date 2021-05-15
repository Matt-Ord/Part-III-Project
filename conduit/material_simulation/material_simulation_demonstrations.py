from numpy.core.function_base import linspace
from numpy.lib.function_base import average
from material_simulation.MultiBandMaterialSimulator import (
    MultiBandNickelMaterialSimulatorUtil,
)
from material_simulation.OneBandMaterialSimulator import OneBandMaterialSimulator
import scipy.constants
import numpy as np
import matplotlib.pyplot as plt

from material_simulation.TwoBandMaterialSimulator import TwoBandMaterialSimulator
from properties.MaterialProperties import NICKEL_MATERIAL_PROPERTIES


def plot_simulation_energy_levels():
    # Two Band
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        TwoBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=5,
        number_of_electrons=5,
        target_frequency=1 * 10 ** (2),
    )
    nickel_sim.plot_average_electron_distribution(
        times=[0, 10 ** (-20)], average_over=10
    )
    # nickel_sim.plot_unpertubed_material_energy_states()
    nickel_sim.plot_material_energy_states(
        subplot_lims=[
            [-6 * 10 ** (-25), 6 * 10 ** (-25)],
            [-5 * 10 ** (-25), 5 * 10 ** (-25)],
        ]
    )

    class TempSim(TwoBandMaterialSimulator):
        @property
        def block_factors_for_simulation(self):
            M = super().block_factors_for_simulation
            return [
                [0, M[0][1]],
                [M[1][0], 0],
            ]

    # Two large Band
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        TempSim,
        temperature=150,
        number_of_states_per_band=5,
        number_of_electrons=5,
        target_frequency=150 * scipy.constants.Boltzmann * 0.1 / scipy.constants.hbar,
    )
    # nickel_sim.plot_average_electron_distribution(
    #     times=[0, 10 ** (-20)], average_over=10
    # )
    # nickel_sim.plot_unpertubed_material_energy_states()
    nickel_sim.plot_material_energy_states()

    # Two large Band
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        TwoBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=5,
        number_of_electrons=5,
        target_frequency=150 * scipy.constants.Boltzmann * 0.1 / scipy.constants.hbar,
    )
    # nickel_sim.plot_average_electron_distribution(
    #     times=[0, 10 ** (-20)], average_over=10
    # )
    # nickel_sim.plot_unpertubed_material_energy_states()
    nickel_sim.plot_material_energy_states()

    class DegenerateTwoBandMaterialSimulator(TwoBandMaterialSimulator):
        hydrogen_energies_for_simulation = [0, 0]

    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        DegenerateTwoBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=5,
        number_of_electrons=5,
        target_frequency=1 * 10 ** (9),
    )
    # nickel_sim.plot_average_electron_distribution(
    #     times=[0, 10 ** (-20)], average_over=10
    # )
    # nickel_sim.plot_unpertubed_material_energy_states()
    nickel_sim.plot_material_energy_states()

    # nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
    #     OneBandMaterialSimulator,
    #     temperature=150,
    #     number_of_states_per_band=10,
    #     number_of_electrons=5,
    #     target_frequency=1 * 10 ** (9),
    # )
    # print("No Hydrogen Energies")
    # nickel_sim.plot_material_energy_states()

    # rough sim with hydrogen energies
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        OneBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=10,
        number_of_electrons=5,
        target_frequency=150 * scipy.constants.Boltzmann * 8 / scipy.constants.hbar,
    )
    print("Realstic Hydrogen Energies")
    nickel_sim.hydrogen_energies_for_simulation = NICKEL_MATERIAL_PROPERTIES.hydrogen_energies  # type: ignore
    nickel_sim.plot_material_energy_states()

    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        OneBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=10,
        number_of_electrons=5,
        target_frequency=1 * 10 ** (9),
    )
    # add small hydrogen energy levels
    print("Small Hydrogen Energies")
    nickel_sim.hydrogen_energies_for_simulation = [0, nickel_sim._get_energy_spacing()]  # type: ignore
    nickel_sim.plot_material_energy_states()

    # add large hydrogen energy levels
    nickel_sim.hydrogen_energies_for_simulation = [0, 1000 * nickel_sim._get_energy_spacing()]  # type: ignore
    nickel_sim.plot_material_energy_states()


def plot_rough_simulation_with_electron_densities():
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        OneBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=10,
        number_of_electrons=5,
        target_frequency=150
        * scipy.constants.Boltzmann
        * 20
        * 2
        / scipy.constants.hbar,
    )

    nickel_sim.plot_electron_densities(
        times=np.linspace(0, 5 * 10 ** -4, 1000).tolist(),
        jitter_electrons=True,
    )


def plot_rough_simulation_without_hydrogen_energies():
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        OneBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=10,
        number_of_electrons=5,
        target_frequency=150 * scipy.constants.Boltzmann * 8 / scipy.constants.hbar,
    )
    print(nickel_sim._get_interaction_prefactor())
    print(nickel_sim.electron_energies)
    print(150 * scipy.constants.Boltzmann)

    fig, ax = plt.subplots()
    nickel_sim.plot_average_material(
        times=np.linspace(0, 0.2 * 10 ** -8, 1000),
        average_over=10,
        jitter_electrons=True,
        ax=ax,
    )
    ax.set_title(
        "Plot of Electron Denstity against time\n"
        + r"showing a tunnelling time of around $10^{-9}$ seconds"
    )
    ax.legend()
    fig.tight_layout()
    plt.show()

    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        OneBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=10,
        number_of_electrons=5,
        target_frequency=150 * scipy.constants.Boltzmann * 16 / scipy.constants.hbar,
    )

    fig, ax = plt.subplots()
    nickel_sim.plot_average_electron_distribution(
        times=np.linspace(0, 2 * 10 ** -8, 1000),
        average_over=10,
        ax=ax,
    )
    ax.set_title(
        "Plot of Electron Denstity against time\n"
        + r"showing a tunnelling time of around $10^{-9}$ seconds"
    )
    ax.set_ylim([0, 1])
    fig.tight_layout()
    plt.show()


def plot_rough_simulation_with_hydrogen_energies():
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        OneBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=10,
        number_of_electrons=5,
        target_frequency=150 * scipy.constants.Boltzmann * 10 / scipy.constants.hbar,
    )
    print(nickel_sim._get_interaction_prefactor())
    print(nickel_sim.electron_energies)
    nickel_sim.hydrogen_energies_for_simulation = np.array(  # type: ignore
        NICKEL_MATERIAL_PROPERTIES.hydrogen_energies
    )  # * 10 ** (-2)

    fig, ax = plt.subplots()
    nickel_sim.plot_average_material(
        times=np.linspace(0, 2 * 10 ** -9, 1000),
        average_over=10,
        jitter_electrons=True,
        ax=ax,
    )
    ax.set_title(
        "Plot of Electron Denstity against time\n"
        + r"with different hydrogen energies showing no tunnelling"
    )
    ax.legend()
    fig.tight_layout()
    plt.show()


def demonstrate_temperature_inversion():
    # class ReducedPrefactorOneBandMaterialSimulator(OneBandMaterialSimulator):
    #     def _get_interaction_prefactor(self):
    #         return 0.01 * super()._get_interaction_prefactor()

    # nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
    #     ReducedPrefactorOneBandMaterialSimulator,
    #     temperature=150,
    #     number_of_states_per_band=10,
    #     number_of_electrons=4,
    #     target_frequency=150 * scipy.constants.Boltzmann * 20 / scipy.constants.hbar,
    # )
    # print(nickel_sim._get_interaction_prefactor())
    # print(nickel_sim.electron_energies)

    # fig, ax = plt.subplots()
    # nickel_sim.plot_average_electron_distribution(
    #     times=np.linspace(1 * 10 ** -8, 2 * 10 ** -8, 1000),
    #     average_over=10,
    #     jitter_electrons=True,
    #     ax=ax,
    # )
    # ax.set_title(
    #     "Plot of Electron density against energy\n"
    #     + "showing agreement with the fermi-dirac distribution"
    # )
    # ax.legend()
    # plt.show()

    # nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
    #     OneBandMaterialSimulator,
    #     temperature=150,
    #     number_of_states_per_band=10,
    #     number_of_electrons=5,
    #     target_frequency=150 * scipy.constants.Boltzmann * 20 / scipy.constants.hbar,
    # )
    # print(nickel_sim._get_interaction_prefactor())
    # print(nickel_sim.electron_energies)

    # fig, ax = plt.subplots()
    # nickel_sim.plot_average_electron_distribution(
    #     times=np.linspace(1 * 10 ** -8, 2 * 10 ** -8, 1000),
    #     average_over=10,
    #     jitter_electrons=True,
    #     ax=ax,
    # )
    # ax.set_title(
    #     "Plot of Electron density against energy\n"
    #     + "showing disagreement with the fermi-dirac distribution"
    # )
    # ax.legend()
    # plt.show()

    # nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
    #     OneBandMaterialSimulator,
    #     temperature=150,
    #     number_of_states_per_band=10,
    #     number_of_electrons=5,
    #     target_frequency=150 * scipy.constants.Boltzmann * 20 / scipy.constants.hbar,
    # )
    # print(nickel_sim._get_interaction_prefactor())
    # print(nickel_sim.electron_energies)
    # nickel_sim.hydrogen_energies_for_simulation = np.array(  # type: ignore
    #     [nickel_sim.electron_energies[2], nickel_sim.electron_energies[-2]]
    # )

    # fig, ax = plt.subplots()
    # nickel_sim.plot_average_electron_distribution(
    #     times=np.linspace(1 * 10 ** -8, 2 * 10 ** -8, 1000),
    #     average_over=10,
    #     jitter_electrons=True,
    #     ax=ax,
    # )
    # ax.set_title(
    #     "Plot of Electron density against energy with hydrogen energy\n"
    #     + "showing disagreement with the fermi-dirac distribution"
    # )
    # ax.legend()
    # plt.show()

    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        TwoBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=5,
        number_of_electrons=5,
        target_frequency=150 * scipy.constants.Boltzmann * 0.1 / scipy.constants.hbar,
    )
    print(nickel_sim._get_interaction_prefactor())
    print(nickel_sim.electron_energies)

    fig, ax = plt.subplots()
    nickel_sim.plot_average_electron_distribution(
        times=np.linspace(1 * 10 ** -8, 2 * 10 ** -8, 1000),
        average_over=10,
        ax=ax,
    )
    ax.set_title(
        "Plot of Electron density against energy\n"
        + "showing difference in temperature in the HCP site"
    )
    ax.legend()
    plt.show()


def print_hamiltonian():
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        OneBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=2,
        number_of_electrons=1,
        target_frequency=1 * 10 ** (9),
    )
    sim = nickel_sim._create_simulation(jitter_electrons=False)
    print(sim.hamiltonian)


def plot_material_with_and_without_diagonal():
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        OneBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=10,
        number_of_electrons=5,
        target_frequency=1 * 10 ** (9),
    )
    nickel_sim.plot_average_densities(
        times=np.linspace(0, 5e-04, 1000).tolist(),
        average_over=100,
        jitter_electrons=True,
        initial_occupancy=1,
    )

    nickel_sim.remove_diagonal_block_factors_for_simulation()
    nickel_sim.plot_average_densities(
        times=np.linspace(0, 5e-04, 1000).tolist(),
        average_over=100,
        jitter_electrons=True,
        initial_occupancy=1,
    )


def plot_two_band_sim_example():

    number_of_electrons = 5
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        TwoBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=5,
        number_of_electrons=number_of_electrons,
        target_frequency=1 * 10 ** (5)
        # target_frequency=150
        # * scipy.constants.Boltzmann
        # * 0.0000001
        # / scipy.constants.hbar,
    )

    # initial_densities = nickel_sim.get_initial_electron_densities(average_over=100)
    upper_n = 0.6920113340410092  # np.average(initial_densities[:5])
    lower_n = 0.30798866595899077  # np.average(initial_densities[5:])
    print(upper_n, lower_n)
    gamma1 = upper_n * (1 - lower_n)
    gamma2 = lower_n * (1 - upper_n)

    fig, ax = nickel_sim.plot_average_material(
        times=np.linspace(0, 1 * 10 ** (1), 1000).tolist(),
        average_over=10,
        jitter_electrons=True,
        initial_occupancy=0.6920113340410092,
    )
    ax.set_ylim([0, 5])
    ax.set_title(
        "Plot of Average Electron Density Against Time\nWith an Initial Occuapancy Close To Equilibrium"
    )

    upper_line = number_of_electrons * gamma1 / (gamma1 + gamma2)
    ax.axhline(
        y=upper_line,
        linestyle="dashed",
        color="black",
        alpha=0.3,
    )

    lower_line = number_of_electrons * gamma2 / (gamma1 + gamma2)
    ax.axhline(
        y=lower_line,
        linestyle="dashed",
        color="black",
        alpha=0.3,
    )
    plt.show()

    class TwoBandSmallJumpMaterialSimulator(TwoBandMaterialSimulator):
        @property
        def hydrogen_energies_for_simulation(self):
            return [0.001 * x for x in super().hydrogen_energies_for_simulation]

    number_of_electrons = 5
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        TwoBandSmallJumpMaterialSimulator,
        temperature=150,
        number_of_states_per_band=5,
        number_of_electrons=number_of_electrons,
        target_frequency=150 * scipy.constants.Boltzmann * 0.1 / scipy.constants.hbar,
    )

    initial_densities = nickel_sim.get_initial_electron_densities()
    upper_n = np.average(initial_densities[:5])
    lower_n = np.average(initial_densities[5:])
    print(upper_n, lower_n)
    gamma1 = upper_n * (1 - lower_n)
    gamma2 = lower_n * (1 - upper_n)

    fig, ax = nickel_sim.plot_average_material(
        times=np.linspace(0, 0.5 * 10 ** -6, 1000).tolist(),
        average_over=10,
        jitter_electrons=True,
        initial_occupancy=1,
    )

    upper_line = number_of_electrons * gamma1 / (gamma1 + gamma2)
    ax.axhline(
        y=upper_line,
        linestyle="dashed",
        color="black",
        alpha=0.3,
    )

    lower_line = number_of_electrons * gamma2 / (gamma1 + gamma2)
    ax.axhline(
        y=lower_line,
        linestyle="dashed",
        color="black",
        alpha=0.3,
    )
    plt.show()

    number_of_electrons = 5
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        TwoBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=5,
        number_of_electrons=number_of_electrons,
        target_frequency=150 * scipy.constants.Boltzmann * 0.1 / scipy.constants.hbar,
    )

    initial_densities = nickel_sim.get_initial_electron_densities()
    upper_n = np.average(initial_densities[:5])
    lower_n = np.average(initial_densities[5:])
    gamma1 = upper_n * (1 - lower_n)
    gamma2 = lower_n * (1 - upper_n)

    fig, ax = nickel_sim.plot_average_material(
        times=np.linspace(0, 1 * 10 ** -8, 1000).tolist(),
        average_over=10,
        jitter_electrons=True,
    )

    upper_line = number_of_electrons * gamma1 / (gamma1 + gamma2)
    ax.axhline(
        y=upper_line,
        linestyle="dashed",
        color="black",
        alpha=0.3,
    )

    lower_line = number_of_electrons * gamma2 / (gamma1 + gamma2)
    ax.axhline(
        y=lower_line,
        linestyle="dashed",
        color="black",
        alpha=0.3,
    )
    plt.show()


def plot_one_band_non_degenerate_sim_example():
    pass


def plot_density_matrix_demonstration():
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        OneBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=8,
        number_of_electrons=4,
        target_frequency=1 * 10 ** (9),
    )
    nickel_sim.plot_average_material(
        times=np.linspace(0, 2e-4, 1000), jitter_electrons=True
    )
    plt.show()
    fig, ax = nickel_sim.plot_average_off_diagonal_density_matrix(
        initial_time=1e-3,
        average_over_times=np.linspace(0, 0.5e-7, 100).tolist(),
        average_over=100,
    )
    ax.set_title(
        "Average off Diagonal Against Time With a\n"
        + r"Characteristic decay time of $\sim{}10^{-8}s$"
    )
    plt.show()
    fig, ax = nickel_sim.plot_off_diagonal_density_matrix(
        times=np.linspace(0, 10e-3, 5000)
    )
    plt.show()
    fig, ax = nickel_sim.plot_time_average_density_matrix(
        times=np.linspace(1e-3, 1.01e-3, 5000)
    )
    ax.set_title("Full Density Matrix Averaged Over $10^{-6}s$")
    plt.show()
    fig, ax = nickel_sim.plot_density_matrix(time=1e-3)
    plt.show()
    fig, ax = nickel_sim.plot_time_average_electron_density_matrix(
        times=np.linspace(1e-3, 10e-3, 5000)
    )
    plt.show()
    fig, ax = nickel_sim.plot_electron_density_matrix(time=1e-3)
    plt.show()


if __name__ == "__main__":
    # plot_density_matrix_demonstration()
    # print_hamiltonian()
    # plot_simulation_energy_levels()
    # plot_rough_simulation_with_electron_densities()
    # plot_rough_simulation_without_hydrogen_energies()
    plot_rough_simulation_with_hydrogen_energies()
    # demonstrate_temperature_inversion()
    # plot_material_with_and_without_diagonal()
    # plot_two_band_sim_example()
