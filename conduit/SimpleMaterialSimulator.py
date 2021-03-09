import numpy as np
import scipy.constants

from simulation.ElectronSimulator import (
    ElectronSimulator,
    ElectronSimulatorConfig
)
from properties.MaterialProperties import (
    MaterialProperties,
    NICKEL_MATERIAL_PROPERTIES
)


class SimpleMaterialElectronSimulator():

    def __init__(self, material_properties: MaterialProperties) -> None:
        self.material_properties = material_properties

    def _get_interaction_prefactor(self, wavevector_spacing):
        implied_volume = self._get_implied_volume(wavevector_spacing)
        # Ignoring q dependance the interaction
        # takes the form -2e^2 / epsilon_0 alpha^2 (see 6.9 on lab book)
        alpha = 3.77948796 * 10 ** 10  # m^-1
        potential_factor = - 2 * (scipy.constants.e ** 2) / \
            (scipy.constants.epsilon_0 * (alpha ** 2))
        prefactor = (4 * np.pi * potential_factor
                     / implied_volume)

        return prefactor

    def _get_interaction_q_factor(self, wavevector_spacing):
        # return lambda q: 1
        return lambda q: self._get_interaction_prefactor(wavevector_spacing)

    def _get_implied_volume(self, wavevector_spacing):
        # Electron states are quantised as k = 2npi / L
        # For a reduced system we have an implied lenght
        # L = 2 pi / wavevector_spacing
        return ((scipy.constants.pi ** 2) /
                (wavevector_spacing *
                 self.material_properties.fermi_wavevector ** 2))

# The implied length method is no longert thought to work
    # @staticmethod
    # def _get_implied_volume(wavevector_spacing):
    #     # Electron states are quantised as k = 2npi / L
    #     # For a reduced system we have an implied lenght
    #     # L = 2 pi / wavevector_spacing
    #     return (2 * scipy.constants.pi / wavevector_spacing)**3

    def simulate_material(self, number_of_states, k_width, times):
        fermi_k = self.material_properties.fermi_wavevector
        k_states = np.linspace(
            fermi_k-k_width,
            fermi_k + k_width,
            number_of_states)

        print(fermi_k, k_width)

        block_factors = self.material_properties.hydrogen_overlaps
        print(block_factors)

        wavevector_spacing = k_states[1] - k_states[0]
        q_factor = self._get_interaction_q_factor(wavevector_spacing)

        print('q_factor', q_factor(0))

        sim = ElectronSimulator(ElectronSimulatorConfig(
            hbar=scipy.constants.hbar,
            electron_mass=scipy.constants.electron_mass
        ))

        sim.simulate_random_system_coherently(
            k_states,
            times,
            block_factors,
            q_factor
        )


class SimpleNickelElectronSimulator(SimpleMaterialElectronSimulator):

    def __init__(self) -> None:
        super().__init__(NICKEL_MATERIAL_PROPERTIES)

    def run_default_sim(self):
        temperature = 200
        d_k = (scipy.constants.Boltzmann * temperature
               * scipy.constants.electron_mass
               / (NICKEL_MATERIAL_PROPERTIES.fermi_wavevector * (
                   scipy.constants.hbar ** 2)))

        self.simulate_material(
            number_of_states=6,
            k_width=2 * d_k,
            times=np.linspace(
                0, 10 ** -10, 1000),
        )


if __name__ == '__main__':
    nickel_sim = SimpleNickelElectronSimulator()
    nickel_sim.run_default_sim()
