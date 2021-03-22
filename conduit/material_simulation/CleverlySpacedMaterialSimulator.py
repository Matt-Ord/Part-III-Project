from typing import Type, TypeVar
import numpy as np
from properties.MaterialProperties import MaterialProperties, NICKEL_MATERIAL_PROPERTIES
from material_simulation.MaterialSimulator import MaterialSimulator

T = TypeVar("T")


class CleverlySpacedMaterialSimulator(MaterialSimulator):
    def __init__(
        self,
        material_properties: MaterialProperties,
        temperature: float,
        number_of_bands: int,
        number_of_electrons: int,
    ) -> None:
        self.number_of_bands = number_of_bands
        self.number_of_electrons = number_of_electrons
        super().__init__(material_properties, temperature)

    def _generate_electron_energies(self):
        return np.arange(self.number_of_bands) * self.hydrogen_energy_difference

    def _get_energy_spacing(self):
        return self.hydrogen_energy_difference

    @property
    def hydrogen_energies(self):
        return [0, 0.01 * self.boltzmann_energy]

    def _get_energy_jitter(self):
        return 0.01 * self._get_energy_spacing()


class CleverlySpacedNickelMaterialSimulatorUtil:
    @staticmethod
    def create(
        sim: Type[CleverlySpacedMaterialSimulator],
        temperature,
        number_of_bands,
        number_of_electrons,
        *args,
        **kwargs
    ) -> CleverlySpacedMaterialSimulator:
        return sim(
            material_properties=NICKEL_MATERIAL_PROPERTIES,
            temperature=temperature,
            number_of_bands=number_of_bands,
            number_of_electrons=number_of_electrons,
            *args,
            **kwargs
        )


if __name__ == "__main__":
    nickel_sim = CleverlySpacedNickelMaterialSimulatorUtil.create(
        CleverlySpacedMaterialSimulator,
        temperature=150,
        number_of_bands=97,
        number_of_electrons=96,
    )

    nickel_sim.simulate_average_material(
        times=np.linspace(0, 6 * 10 ** (1), 1000),
        average_over=10,
        jitter_electrons=False,
    )
