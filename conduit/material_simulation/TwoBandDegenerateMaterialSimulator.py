import numpy as np
import scipy.constants
from material_simulation.MultiBandMaterialSimulator import (
    MultiBandMaterialSimulator,
    MultiBandNickelMaterialSimulatorUtil,
)

from properties.MaterialProperties import (
    MaterialProperties,
)

# Simulates a material using the two band
# approach, which alllows for nearly
# degenerate hopping to be seen and the
# difference in hydrogen energy to be incorperated
# but ignores any offset in the initial and final
# states.

# Note the energy difference is given in units of k_t


class TwoBandDegenerateMaterialSimulator(MultiBandMaterialSimulator):
    def __init__(
        self,
        material_properties: MaterialProperties,
        temperature: float,
        number_of_states_per_band: int,
        bandwidth: float,
        energy_difference: float,
    ) -> None:
        self.energy_difference = energy_difference
        super().__init__(
            material_properties, temperature, number_of_states_per_band, bandwidth
        )

    @property
    def hydrogen_energies_for_simulation(self):
        return [0, 0]

    @property
    def energy_difference_in_joules(self):
        return self.energy_difference * self.boltzmann_energy

    def _generate_electron_energies(self):
        energy_difference = self.energy_difference_in_joules

        lower_band_energies = self._get_band_energies() - energy_difference / 2
        upper_band_energies = self._get_band_energies() + energy_difference / 2

        energies = np.concatenate([lower_band_energies, upper_band_energies])
        return energies


if __name__ == "__main__":
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        TwoBandDegenerateMaterialSimulator,
        temperature=150,
        number_of_states_per_band=4,
        target_frequency=1 * 10 ** (11),  # 1 * 10 ** (9)
        energy_difference=1,
        number_of_electrons=7,
    )

    nickel_sim.simulate_average_material(
        times=np.linspace(0, 2 * 10 ** -5, 1000),
        average_over=10,
        jitter_electrons=True,
        period_of_noise_fluctuation=1 * 10 ** (-6),
    )
