import numpy as np
import scipy.constants
from matplotlib.pyplot import title
from properties.MaterialProperties import NICKEL_MATERIAL_PROPERTIES, MaterialProperties

from material_simulation.MultiBandMaterialSimulator import (
    MultiBandMaterialSimulator,
    MultiBandNickelMaterialSimulatorUtil,
)

# Simulates the electron sysytem using a single
# Closely packed band, and uses the theroetical
# electron occupation of the lower band to come to
# an approximate tunnelling rate


class OneBandTwoSpinMaterialSimulator(MultiBandMaterialSimulator):
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
    def number_of_bands(self):
        return 2

    hydrogen_energies_for_simulation = [0, 0]

    def _generate_electron_energies(self) -> np.ndarray:
        return np.tile(self._get_band_energies(), 2)


if __name__ == "__main__":
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        OneBandTwoSpinMaterialSimulator,
        temperature=150,
        number_of_states_per_band=5,
        number_of_electrons=4,
        target_frequency=1 * 10 ** (9),
    )

    nickel_sim.simulate_average_material(
        times=np.linspace(0, 1000e-04, 1000).tolist(),
        average_over=50,
        jitter_electrons=True,
        initial_occupancy=1,
    )
