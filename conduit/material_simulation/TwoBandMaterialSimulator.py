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


if __name__ == "__main__":
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        TwoBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=6,
        target_frequency=1 * 10 ** (9),
    )

    # nickel_sim.simulate_material(
    #     times=np.linspace(0, 4 * 10 ** -5, 1000),
    #     initial_electron_state=[1, 1, 1, 1, 0, 0, 0, 0],
    # )

    nickel_sim.simulate_average_material(
        times=np.linspace(0, 4 * 10 ** -4, 1), average_over=1, jitter_electrons=True
    )
