from material_simulation.MultiBandMaterialSimulator import (
    MultiBandMaterialSimulator,
    MultiBandNickelMaterialSimulatorUtil,
)
from properties.MaterialProperties import MaterialProperties
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

    @property
    def hydrogen_energies_for_simulation(self):
        return [0, 0]

    def _generate_electron_energies(self):
        return self._get_band_energies()

    def _get_fraction_of_occupation(self):
        return self.number_of_electrons / self.number_of_states_per_band


if __name__ == "__main__":
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        OneBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=13,
        number_of_electrons=1,
        target_frequency=1 * 10 ** (9),
    )

    # nickel_sim.simulate_material(times=np.linspace(0, 0.001 * 10 ** -12, 1000))

    # nickel_sim.simulate_average_material(
    #     times=np.linspace(0, 3 * 10 ** -10, 500), average_over=20
    # )
    nickel_sim.simulate_average_material(
        times=np.linspace(0, 60 * 10 ** -5, 1000),
        average_over=10,
        jitter_electrons=True,
    )
