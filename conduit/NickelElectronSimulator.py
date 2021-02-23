from typing import List, NamedTuple
import numpy as np


class MaterialProperties(NamedTuple):
    fermi_wavevector: float
    hydrogen_energies: List[float]
    hydrogen_overlaps: List[List[complex]]


class MaterialElectronSimulator():

    def __init__(self, material_properties: MaterialProperties) -> None:
        self.material_properties = material_properties

    def _get_interaction_values(self, wavevector_spacing):
        interaction_prefactor = self._get_interaction_prefactor(
            wavevector_spacing)
        return (interaction_prefactor *
                np.array(self.material_properties.hydrogen_overlaps))

    def _get_interaction_prefactor(self, wavevector_spacing):
        implied_length = self._get_implied_length(wavevector_spacing)
        # Ignoring q dependance the interaction
        # takes the form -2e^2 / epsilon_0 alpha^2 (see 6.9 on lab book)
        alpha = 14.894
        potential_factor = - 2 / (alpha ** 2)
        return potential_factor / (implied_length ** 3)

    def _get_implied_length(self, wavevector_spacing):
        # Electron states are quantised as k = 2npi / L
        # For a reduced system we have an implied lenght
        # L = 2 pi / wavevector_spacing
        return 2 * np.pi / wavevector_spacing

    def simulate_material(number_of_states, ):
        pass
