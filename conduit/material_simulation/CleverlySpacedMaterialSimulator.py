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
        super().__init__(material_properties, temperature, number_of_electrons)

    def _generate_electron_energies(self):
        return np.arange(self.number_of_bands) * self._get_energy_spacing()

    def _get_energy_spacing(self):
        return self.hydrogen_energy_difference

    @property
    def hydrogen_energies_for_simulation(self):
        return [0, 0]

    @property
    def block_factors_for_simulation(self):
        M = self.hydrogen_overlaps
        d_factor = 1
        return [
            [M[0][0], d_factor * M[0][1]],
            [d_factor * M[1][0], M[1][1]],
        ]

    def _get_energy_jitter(self):
        return 0.2 * self._get_energy_spacing()


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
        number_of_bands=10,
        number_of_electrons=5,
    )
    nickel_sim.plot_electron_densities(
        times=np.linspace(0, 4 * 10 ** (-13), 1000).tolist(),
    )

    nickel_sim.plot_average_densities(
        times=np.linspace(0, 2 * 10 ** (-12), 1000).tolist(),
        average_over=20,
        jitter_electrons=True,
    )
