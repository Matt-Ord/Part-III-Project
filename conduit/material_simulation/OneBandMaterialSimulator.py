from material_simulation.MultiBandMaterialSimulator import (
    MultiBandMaterialSimulator,
    MultiBandNickelMaterialSimulatorUtil,
)
from properties.MaterialProperties import MaterialProperties, NICKEL_MATERIAL_PROPERTIES
import numpy as np

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

    @property
    def number_of_bands(self):
        return 1

    def _generate_electron_energies(self) -> np.ndarray:
        return self._get_band_energies()


if __name__ == "__main__":
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        OneBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=15,
        number_of_electrons=5,
        target_frequency=1 * 10 ** (9),
    )

    # nickel_sim.simulate_material(times=np.linspace(0, 0.001 * 10 ** -12, 1000))

    # nickel_sim.simulate_average_material(
    #     times=np.linspace(0, 3 * 10 ** -10, 500), average_over=20
    # )
    # nickel_sim.remove_diagonal_block_factors_for_simulation()
    nickel_sim.plot_average_densities(
        times=np.linspace(0, 7e-04, 1000).tolist(),
        average_over=10,
        jitter_electrons=True,
        initial_occupancy=1,
    )
