from abc import abstractmethod
from typing import Any, Generic, Type, TypeVar
import numpy as np
import scipy.constants

from properties.MaterialProperties import (
    MaterialProperties,
    NICKEL_MATERIAL_PROPERTIES,
)

from material_simulation.MaterialSimulator import MaterialSimulator

T = TypeVar("T")

# Simulates a material using the multi band
# approach, which alllows for nearly
# degenerate hopping to be seen


class MultiBandMaterialSimulator(MaterialSimulator):
    def __init__(
        self,
        material_properties: MaterialProperties,
        temperature: float,
        number_of_states_per_band: int,
        number_of_electrons: int,
        bandwidth: float,
    ) -> None:
        print(number_of_electrons)
        self.number_of_states_per_band = number_of_states_per_band
        self.number_of_electrons = number_of_electrons
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
        return 0.5 * self._get_energy_spacing()


class MultiBandMaterialSimulatorUtil(Generic[T]):
    @staticmethod
    def calculate_bandwidth(target_frequency):
        return scipy.constants.hbar * target_frequency / 2

    @classmethod
    def create(
        cls,
        sim: Any,
        material_properties: MaterialProperties,
        temperature,
        number_of_states_per_band,
        number_of_electrons,
        target_frequency,
        *args,
        **kwargs
    ) -> T:
        bandwidth = cls.calculate_bandwidth(target_frequency)
        return sim(
            material_properties=material_properties,
            temperature=temperature,
            number_of_states_per_band=number_of_states_per_band,
            number_of_electrons=number_of_electrons,
            bandwidth=bandwidth,
            *args,
            **kwargs
        )


class MultiBandNickelMaterialSimulatorUtil(MultiBandMaterialSimulatorUtil):
    @classmethod
    def create(
        cls,
        sim: type[T],
        temperature,
        number_of_states_per_band,
        number_of_electrons,
        target_frequency,
        *args,
        **kwargs
    ) -> T:
        return super().create(
            sim,
            material_properties=NICKEL_MATERIAL_PROPERTIES,
            temperature=temperature,
            number_of_states_per_band=number_of_states_per_band,
            number_of_electrons=number_of_electrons,
            target_frequency=target_frequency,
            *args,
            **kwargs
        )
