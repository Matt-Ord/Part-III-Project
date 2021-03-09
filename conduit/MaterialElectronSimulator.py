from typing import List, NamedTuple
from matplotlib.pyplot import sci
import numpy as np
from ElectronSimulator import ElectronSimulator, ElectronSimulatorConfig
import scipy.constants


class MaterialProperties(NamedTuple):
    # Material properties as in SI units
    fermi_wavevector: float
    hydrogen_energies: List[float]
    hydrogen_overlaps: List[List[complex]]


class MaterialElectronSimulator():

    def __init__(self, material_properties: MaterialProperties) -> None:
        self.material_properties = material_properties

    def _get_interaction_prefactor(self, wavevector_spacing):
        implied_volume = self._get_implied_volume(wavevector_spacing)
        # Ignoring q dependance the interaction
        # takes the form -2e^2 / epsilon_0 alpha^2 (see 6.9 on lab book)
        alpha = 3.77948796 * 10 ** 10  # m^-1
        potential_factor = - 2 * (scipy.constants.e ** 2) / \
            (scipy.constants.epsilon_0 * (alpha ** 2))
        # prefactor = 4 * np.pi * self.material_properties.fermi_wavevector ** 2 * \ noqa  E501
        #     potential_factor / (implied_length ** 3)
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


class MaterialElectronSimulator1D():

    def __init__(self, material_properties: MaterialProperties) -> None:
        self.material_properties = material_properties

    def _get_interaction_prefactor(self, wavevector_spacing):
        implied_volume = self._get_implied_volume(wavevector_spacing)
        # Ignoring q dependance the interaction
        # takes the form -2e^2 / epsilon_0 alpha^2 (see 6.9 on lab book)
        alpha = 3.77948796 * 10 ** 10  # m^-1
        potential_factor = - 2 * (scipy.constants.e ** 2) / \
            (scipy.constants.epsilon_0 * (alpha ** 2))

        prefactor = (4 * np.pi * potential_factor / implied_volume)

        return prefactor

    def _get_interaction_q_factor(self, wavevector_spacing):
        # return lambda q: 1
        return lambda q: self._get_interaction_prefactor(wavevector_spacing)

    def _get_implied_volume(self, wavevector_spacing):
        return scipy.constants.pi / (2 * wavevector_spacing)

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


NICKEL_FCC_ENERGY = .1668191  # eV
NICKEL_HCP_ENERGY = .1794517  # eV

NICKEL_MATERIAL_PROPERTIES = MaterialProperties(
    fermi_wavevector=1.175 * 10 ** (10),  # m^-1, \cite{PhysRev.131.2469}
    hydrogen_energies=[0, 0],
    hydrogen_overlaps=[[1, 0.004], [0.004, 1]]  # Rough overlap!
)


class NickelElectronSimulator(MaterialElectronSimulator):

    def __init__(self) -> None:
        super().__init__(NICKEL_MATERIAL_PROPERTIES)


if __name__ == '__main__':
    nickel_sim = NickelElectronSimulator()

    temperature = 200
    d_k = (scipy.constants.Boltzmann * temperature
           * scipy.constants.electron_mass
           / (NICKEL_MATERIAL_PROPERTIES.fermi_wavevector * (
               scipy.constants.hbar ** 2)))

    nickel_sim.simulate_material(
        number_of_states=6,
        k_width=2 * d_k,
        times=np.linspace(
            0, 10 ** -10, 1000),
    )
