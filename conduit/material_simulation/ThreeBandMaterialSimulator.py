import numpy as np
from material_simulation.MultiBandMaterialSimulator import (
    MultiBandMaterialSimulator,
    MultiBandNickelMaterialSimulatorUtil,
)

# Simulates a material using the three band
# approach, which alllows for nearly
# degenerate hopping to be seen and the
# difference in hydrogen energy to be incorperated


class ThreeBandMaterialSimulator(MultiBandMaterialSimulator):
    @property
    def number_of_bands(self):
        return 3

    def _generate_electron_energies(self):
        hydrogen_energies = self.material_properties.hydrogen_energies

        first_band_energies = (
            self._get_band_energies() - hydrogen_energies[0] + hydrogen_energies[1]
        )
        second_band_energies = self._get_band_energies()
        third_band_energies = (
            self._get_band_energies() - hydrogen_energies[1] + hydrogen_energies[0]
        )

        energies = np.concatenate(
            [
                first_band_energies,
                second_band_energies,
                third_band_energies,
            ]
        )
        return energies


if __name__ == "__main__":
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        ThreeBandMaterialSimulator,
        temperature=10000,
        number_of_states_per_band=3,
        target_frequency=1 * 10 ** (9),
        number_of_electrons=3,
    )

    # nickel_sim.simulate_material(
    #     times=np.linspace(0, 4 * 10 ** -5, 1000),
    #     initial_electron_state=[1, 1, 1, 1, 0, 0, 0, 0],
    # )

    nickel_sim.plot_average_densities(
        times=np.linspace(0, 4 * 10 ** -5, 1000).tolist(),
        average_over=20,
        jitter_electrons=True,
    )
