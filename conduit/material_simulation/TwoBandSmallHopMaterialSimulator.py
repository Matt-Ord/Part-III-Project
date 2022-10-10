import numpy as np
from material_simulation.MultiBandMaterialSimulator import (
    MultiBandMaterialSimulator,
    MultiBandNickelMaterialSimulatorUtil,
)
from properties.MaterialProperties import MaterialProperties

# Simulates a material using the two band
# approach, which alllows for nearly
# degenerate hopping to be seen and the
# difference in hydrogen energy to be incorperated


class TwoBandSmallHopMaterialSimulator(MultiBandMaterialSimulator):
    def __init__(
        self,
        material_properties: MaterialProperties,
        temperature: float,
        number_of_states_per_band: int,
        number_of_electrons: int,
        bandwidth: float,
        hop_energy_factor: float,
    ) -> None:
        self.hop_energy = hop_energy_factor * bandwidth
        super().__init__(
            material_properties,
            temperature,
            number_of_states_per_band,
            number_of_electrons,
            bandwidth,
        )

    @property
    def hydrogen_energies_for_simulation(self):
        return [0, self.hop_energy]

    def _generate_electron_energies(self):

        middle_band_energies = self._get_band_energies()
        lower_band_energies = [middle_band_energies[0] - self.boltzmann_energy]

        upper_band_energies = self._get_band_energies() + self.hop_energy
        lower_band_energies2 = [upper_band_energies[0] + self.boltzmann_energy]

        energies = np.concatenate(
            [
                lower_band_energies,
                middle_band_energies,
                upper_band_energies,
                lower_band_energies2,
            ]
        )
        return energies


if __name__ == "__main__":
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        TwoBandSmallHopMaterialSimulator,
        temperature=150,
        number_of_states_per_band=3,
        number_of_electrons=2,
        target_frequency=1 * 10 ** (9),
        hop_energy_factor=0.1,
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
