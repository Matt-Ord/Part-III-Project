import numpy as np
import scipy.constants

from properties.MaterialProperties import (
    NICKEL_MATERIAL_PROPERTIES,
)

from material_simulation.MaterialSimulator import MaterialSimulator

# Simulates a material using the basic approach,
# without considering the interaction of closely
# packed electron states and ignoring any offset
# in the hydrogen energy


class SimpleMaterialElectronSimulator(MaterialSimulator):
    def __init__(
        self,
        material_properties,
        temperature,
        number_of_states,
        number_of_electrons,
        **kwargs
    ) -> None:
        self.number_of_states = number_of_states
        super().__init__(material_properties, temperature, number_of_electrons)
        print(
            self.electron_energies,
            self.hydrogen_overlaps,
            self._get_interaction_prefactor(),
        )

    @property
    def hydrogen_energies_for_simualtion(self):
        return [0, 0]

    def _generate_electron_energies(self):
        d_e = 0.02 * self.boltzmann_energy

        e_states = np.linspace(-d_e, d_e, self.number_of_states)
        return e_states

    def _get_energy_spacing(self):
        return self.electron_energies[1] - self.electron_energies[0]

    def _get_energy_jitter(self):
        return 0.3 * self._get_energy_spacing()


class SimpleNickelElectronSimulator(SimpleMaterialElectronSimulator):
    def __init__(self, temperature, number_of_states, number_of_electrons) -> None:
        super().__init__(
            NICKEL_MATERIAL_PROPERTIES,
            temperature,
            number_of_states,
            number_of_electrons,
        )


if __name__ == "__main__":
    nickel_sim = SimpleNickelElectronSimulator(
        temperature=150,
        number_of_states=8,
        number_of_electrons=4,
    )
    nickel_sim.plot_electron_densities(times=np.linspace(0, 1 * 10 ** 8, 1000).tolist())

    # nickel_sim.simulate_average_material(
    #     times=np.linspace(0, 3 * 10 ** -10, 500), average_over=20
    # )
    nickel_sim.plot_average_densities(
        times=np.linspace(0, 3 * 10 ** 8, 500).tolist(),
        average_over=40,
        jitter_electrons=True,
    )
