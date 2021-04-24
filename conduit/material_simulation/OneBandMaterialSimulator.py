from matplotlib.pyplot import title
from material_simulation.MultiBandMaterialSimulator import (
    MultiBandMaterialSimulator,
    MultiBandNickelMaterialSimulatorUtil,
)
from properties.MaterialProperties import MaterialProperties, NICKEL_MATERIAL_PROPERTIES
import numpy as np
import scipy.constants

# Simulates the electron sysytem using a single
# Closely packed band, and uses the theroetical
# electron occupation of the lower band to come to
# an approximate tunnelling rate


class OneBandMaterialSimulator(MultiBandMaterialSimulator):
    def __init__(
        self,
        material_properties: MaterialProperties,
        temperature: float,
        number_of_states_per_band: int,
        bandwidth: float,
        number_of_electrons: int,
    ) -> None:
        super().__init__(
            material_properties,
            temperature,
            number_of_states_per_band,
            number_of_electrons,
            bandwidth,
        )

    hydrogen_energies_for_simulation = [0, 0]

    def _generate_electron_energies(self):
        return self._get_band_energies()

    def _get_fraction_of_occupation(self):
        return self.number_of_electrons / self.number_of_states_per_band


def plot_simulation_energy_levels():
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        OneBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=10,
        number_of_electrons=5,
        target_frequency=1 * 10 ** (9),
    )

    nickel_sim.plot_material_energy_states()

    # add small hydrogen energy levels
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

    nickel_sim.simulate_material(
        times=np.linspace(0, 5 * 10 ** -4, 1000).tolist(),
        jitter_electrons=True,
    )


def plot_rough_simulation_without_hydrogen_energies():
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        OneBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=10,
        number_of_electrons=5,
        target_frequency=150 * scipy.constants.Boltzmann * 10 / scipy.constants.hbar,
    )
    print(nickel_sim._get_interaction_prefactor())
    print(nickel_sim.electron_energies)

    nickel_sim.plot_average_material(
        times=np.linspace(0, 0.2 * 10 ** -8, 1000),
        average_over=10,
        jitter_electrons=True,
        title="Plot of Electron Denstity against time\n"
        + r"showing a tunnelling time of around $10^{-10}$ seconds",
    )


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
    nickel_sim.hydrogen_energies_for_simulation = (  # type: ignore
        1 * np.array(NICKEL_MATERIAL_PROPERTIES.hydrogen_energies) * 10 ** (-2)
    )

    nickel_sim.plot_average_material(
        times=np.linspace(0, 2 * 10 ** -9, 1000),
        average_over=10,
        jitter_electrons=True,
        title="Plot of Electron Denstity against time\n"
        + r"with different hydrogen energies showing no tunnelling",
    )


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
    nickel_sim.simulate_average_material(
        times=np.linspace(0, 5e-04, 1000).tolist(),
        average_over=100,
        jitter_electrons=True,
        initial_occupancy=1,
    )

    nickel_sim.remove_diagonal_block_factors_for_simulation()
    nickel_sim.simulate_average_material(
        times=np.linspace(0, 5e-04, 1000).tolist(),
        average_over=100,
        jitter_electrons=True,
        initial_occupancy=1,
    )


if __name__ == "__main__":
    # print_hamiltonian()
    # plot_simulation_energy_levels()
    # plot_rough_simulation_with_hydrogen_energies()
    # plot_rough_simulation_with_hydrogen_energies()
    # plot_material_with_and_without_diagonal()
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        OneBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=97,
        number_of_electrons=1,
        target_frequency=1 * 10 ** (9),
    )

    # nickel_sim.simulate_material(times=np.linspace(0, 0.001 * 10 ** -12, 1000))

    # nickel_sim.simulate_average_material(
    #     times=np.linspace(0, 3 * 10 ** -10, 500), average_over=20
    # )
    nickel_sim.remove_diagonal_block_factors_for_simulation()
    nickel_sim.simulate_average_material(
        times=np.linspace(0, 20e-04, 1000).tolist(),
        average_over=100,
        jitter_electrons=True,
        initial_occupancy=1,
    )
