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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def hydrogen_energies_for_simualtion(self):
        return [0, 0]

    def _generate_electron_energies(self, number_of_states):
        d_k = 4 * (
            self.boltzmann_energy
            * scipy.constants.electron_mass
            / (self.material_properties.fermi_wavevector * (scipy.constants.hbar ** 2))
        )

        # k_states = np.linspace(
        #     self.material_properties.fermi_wavevector - d_k,
        #     self.material_properties.fermi_wavevector + d_k,
        #     number_of_states,
        # )
        k_states = np.linspace(
            self.fermi_wavevector - d_k,
            self.fermi_wavevector + d_k,
            number_of_states,
        )

        energies = self._calculate_electron_energies(k_states)
        return energies - self._calculate_electron_energies(self.fermi_wavevector)

    def _get_energy_spacing(self):
        return self.electron_energies[1] - self.electron_energies[0]

    def _get_energy_jitter(self):
        return 0.3 * self._get_energy_spacing()


class SimpleNickelElectronSimulator(SimpleMaterialElectronSimulator):
    def __init__(self, temperature, number_of_states) -> None:
        super().__init__(
            NICKEL_MATERIAL_PROPERTIES,
            temperature,
            number_of_states,
        )


if __name__ == "__main__":
    nickel_sim = SimpleNickelElectronSimulator(
        temperature=150,
        number_of_states=8,
    )
    nickel_sim.simulate_material(times=np.linspace(0, 0.001 * 10 ** -12, 1000))

    # nickel_sim.simulate_average_material(
    #     times=np.linspace(0, 3 * 10 ** -10, 500), average_over=20
    # )
    nickel_sim.simulate_average_material(
        times=np.linspace(0, 3 * 10 ** -10, 500), average_over=40, jitter_electrons=True
    )
