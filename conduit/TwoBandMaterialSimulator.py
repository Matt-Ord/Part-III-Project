import numpy as np
import scipy.constants

from properties.MaterialProperties import (
    MaterialProperties,
    NICKEL_MATERIAL_PROPERTIES,
)

from MaterialSimulator import MaterialSimulator

# Simulates a material using the two band
# approach, which alllows for nearly
# degenerate hopping to be seen and the
# difference in hydrogen energy to be incorperated


class TwoBandMaterialSimulator(MaterialSimulator):
    def __init__(
        self,
        material_properties: MaterialProperties,
        temperature: float,
        number_of_states_per_band: int,
        bandwidth: float,
    ) -> None:
        self.temperature = temperature
        self.number_of_states_per_band = number_of_states_per_band
        self.bandwidth = bandwidth
        super().__init__(material_properties)

    # @property
    # def hydrogen_energies(self):
    #     return [0, 0]

    def _generate_electron_energies(self):
        hydrogen_energies = self.material_properties.hydrogen_energies

        lower_band_energies = self._get_band_energies() - hydrogen_energies[0]
        upper_band_energies = self._get_band_energies() - hydrogen_energies[1]

        energies = np.concatenate([lower_band_energies, upper_band_energies])
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


class TwoBandNickelMaterialSimulator(TwoBandMaterialSimulator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(NICKEL_MATERIAL_PROPERTIES, *args, **kwargs)


class TwoBandNickelMaterialSimulatorUtil:
    @staticmethod
    def _calculate_bandwidth(target_frequency):
        return scipy.constants.hbar * target_frequency / 2

    @classmethod
    def create(
        cls,
        sim: TwoBandMaterialSimulator,
        temperature,
        number_of_states_per_band,
        target_frequency,
    ) -> TwoBandMaterialSimulator:
        bandwidth = cls._calculate_bandwidth(target_frequency)
        print("bandwidth", bandwidth)
        return sim(temperature, number_of_states_per_band, bandwidth)


if __name__ == "__main__":
    nickel_sim = TwoBandNickelMaterialSimulatorUtil.create(
        TwoBandNickelMaterialSimulator,
        temperature=200,
        number_of_states_per_band=3,
        target_frequency=1 * 10 ** (9),
    )

    nickel_sim.simulate_average_material(times=np.linspace(0, 4 * 10 ** -6, 1000))
