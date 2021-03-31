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
        hydrogen_energies = self.material_properties.hydrogen_energies

        lower_band_energies = self._get_band_energies() - hydrogen_energies[0]
        upper_band_energies = self._get_band_energies() - hydrogen_energies[1]

        energies = np.concatenate([lower_band_energies, upper_band_energies])
        return energies

    # @property
    # def hydrogen_energies_for_simulation(self):
    #     return [0, 0]

    @property
    def block_factors_for_simulation(self):
        M = self.hydrogen_overlaps
        d_factor = 500
        return [
            [M[0][0], d_factor * M[0][1]],
            [d_factor * M[1][0], M[1][1]],
        ]


if __name__ == "__main__":
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        TwoBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=6,
        number_of_electrons=4,
        target_frequency=100 * 10 ** (9),
    )

    nickel_sim.simulate_material(
        times=np.linspace(0, (4 / 500) * 10 ** -7, 1000),
    )

    # nickel_sim.simulate_average_material(
    #     times=np.linspace(0, 4 * 10 ** -4, 1), average_over=1, jitter_electrons=True
    # )
