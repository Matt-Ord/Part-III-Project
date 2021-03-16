from abc import abstractmethod
import numpy as np
import scipy.constants

from properties.MaterialProperties import (
    MaterialProperties,
    NICKEL_MATERIAL_PROPERTIES,
)

from material_simulation.MaterialSimulator import MaterialSimulator

# Simulates a material using the multi band
# approach, which alllows for nearly
# degenerate hopping to be seen


class MultiBandMaterialSimulator(MaterialSimulator):
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
    @abstractmethod
    def _generate_electron_energies(self):
        pass

    def _get_band_energies(self):
        return np.linspace(
            -self.bandwidth / 2,
            self.bandwidth / 2,
            self.number_of_states_per_band,
        )

    def _get_energy_spacing(self):
        return self.bandwidth / self.number_of_states_per_band

    def _get_energy_jitter(self):
        return 0.1 * self._get_energy_spacing()


class MultiBandMaterialSimulatorUtil:
    @staticmethod
    def _calculate_bandwidth(target_frequency):
        return scipy.constants.hbar * target_frequency / 2

    @classmethod
    def create(
        cls,
        sim: MultiBandMaterialSimulator,
        material_properties: MaterialProperties,
        temperature,
        number_of_states_per_band,
        target_frequency,
        *args,
        **kwargs
    ) -> MultiBandMaterialSimulator:
        bandwidth = cls._calculate_bandwidth(target_frequency)
        return sim(
            material_properties,
            temperature,
            number_of_states_per_band,
            bandwidth,
            *args,
            **kwargs
        )


class MultiBandNickelMaterialSimulatorUtil(MultiBandMaterialSimulatorUtil):
    @classmethod
    def create(
        cls,
        sim: MultiBandMaterialSimulator,
        temperature,
        number_of_states_per_band,
        target_frequency,
        *args,
        **kwargs
    ) -> MultiBandMaterialSimulator:
        return super().create(
            sim,
            NICKEL_MATERIAL_PROPERTIES,
            temperature,
            number_of_states_per_band,
            target_frequency,
            *args,
            **kwargs
        )
