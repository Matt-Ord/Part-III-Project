import numpy as np
import scipy.constants

from properties.MaterialProperties import (
    NICKEL_MATERIAL_PROPERTIES,
)

from MaterialSimulator import MaterialSimulator

# Simulates a material using the basic approach,
# without considering the interaction of closely
# packed electron states and ignoring any offset
# in the hydrogen energy


class SimpleMaterialElectronSimulator(MaterialSimulator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def hydrogen_energies(self):
        return [0, 0]

    def _generate_electron_energies(self, number_of_states, temperature):
        d_k = (
            scipy.constants.Boltzmann
            * temperature
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


class SimpleNickelElectronSimulator(SimpleMaterialElectronSimulator):
    def __init__(self, number_of_states, temperature) -> None:
        super().__init__(
            NICKEL_MATERIAL_PROPERTIES,
            number_of_states,
            temperature,
        )


if __name__ == "__main__":
    nickel_sim = SimpleNickelElectronSimulator(
        temperature=150,
        number_of_states=8,
    )
    nickel_sim.simulate_material(times=np.linspace(0, 3 * 10 ** -10, 1000))

    # nickel_sim.simulate_average_material(times=np.linspace(0, 3 * 10 ** -10, 500))
