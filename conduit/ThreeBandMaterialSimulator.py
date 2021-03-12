import numpy as np
import scipy.constants

from properties.MaterialProperties import (
    MaterialProperties,
    NICKEL_MATERIAL_PROPERTIES,
)

from MaterialSimulator import MaterialSimulator

# Simulates a material using the three band
# approach, which alllows for nearly
# degenerate hopping to be seen and the
# difference in hydrogen energy to be incorperated


class ThreeBandMaterialSimulator(MaterialSimulator):
    def __init__(
        self,
        material_properties: MaterialProperties,
        temperature: float,
        number_of_states_per_band: int,
        bandwidth: float,
    ) -> None:
        self.number_of_states_per_band = number_of_states_per_band
        self.bandwidth = bandwidth
        super().__init__(material_properties, temperature)

    # @property
    # def hydrogen_energies(self):
    #     return [0, 0]

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
        print(energies)
        return energies

    def _get_band_energies(self):
        return np.linspace(
            -self.bandwidth / 2,
            self.bandwidth / 2,
            self.number_of_states_per_band,
        )

    def _get_energy_spacing(self):
        return self.bandwidth / self.number_of_states_per_band

    def _get_energy_jitter(self):
        return 0.01 * self._get_energy_spacing()


class ThreeBandNickelMaterialSimulator(ThreeBandMaterialSimulator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(NICKEL_MATERIAL_PROPERTIES, *args, **kwargs)


class ThreeBandNickelMaterialSimulatorUtil:
    @staticmethod
    def _calculate_bandwidth(target_frequency):
        return scipy.constants.hbar * target_frequency / 2

    @classmethod
    def create(
        cls,
        sim: ThreeBandMaterialSimulator,
        temperature,
        number_of_states_per_band,
        target_frequency,
    ) -> ThreeBandMaterialSimulator:
        bandwidth = cls._calculate_bandwidth(target_frequency)
        print("bandwidth", bandwidth)
        return sim(temperature, number_of_states_per_band, bandwidth)


if __name__ == "__main__":
    nickel_sim = ThreeBandNickelMaterialSimulatorUtil.create(
        ThreeBandNickelMaterialSimulator,
        temperature=10000,
        number_of_states_per_band=3,
        target_frequency=1 * 10 ** (9),
    )

    # nickel_sim.simulate_material(
    #     times=np.linspace(0, 4 * 10 ** -5, 1000),
    #     initial_electron_state=[1, 1, 1, 1, 0, 0, 0, 0],
    # )

    nickel_sim.simulate_average_material(
        times=np.linspace(0, 4 * 10 ** -5, 1000), average_over=20, jitter_electrons=True
    )
