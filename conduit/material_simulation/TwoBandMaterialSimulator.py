import numpy as np
from material_simulation.MultiBandMaterialSimulator import (
    MultiBandMaterialSimulator,
    MultiBandNickelMaterialSimulatorUtil,
)

# Simulates a material using the two band
# approach, which alllows for nearly
# degenerate hopping to be seen and the
# difference in hydrogen energy to be incorperated


class TwoBandMaterialSimulator(MultiBandMaterialSimulator):
    def _generate_electron_energies(self):
        hydrogen_energies = self.hydrogen_energies

        lower_band_energies = self._get_band_energies() - hydrogen_energies[0]
        upper_band_energies = self._get_band_energies() - hydrogen_energies[1]

        energies = np.sort(np.concatenate([lower_band_energies, upper_band_energies]))
        return energies

    @property
    def number_of_bands(self):
        return 2


if __name__ == "__main__":

    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        TwoBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=5,
        number_of_electrons=5,
        target_frequency=1 * 10 ** (2),
    )

    # nickel_sim.simulate_material(
    #     times=np.linspace(0, 1 * 10 ** -5, 1000).tolist(), initial_occupancy=0.5
    # )

    nickel_sim.simulate_average_material(
        times=np.linspace(0, 5 * 10 ** 2, 100).tolist(),
        average_over=50,
        jitter_electrons=True,
        initial_occupancy=1,
    )
