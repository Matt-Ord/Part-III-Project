from abc import ABC, abstractmethod
import numpy as np
from simulation.ElectronSimulation import (
    ElectronSimulation,
    ElectronSimulationConfig,
)
import scipy.constants

from properties.MaterialProperties import (
    MaterialProperties,
)
from simulation.ElectronSimulationPlotter import ElectronSimulationPlotter


class MaterialSimulator(ABC):
    def __init__(
        self,
        material_properties: MaterialProperties,
        temperature: float,
        *args,
        **kwargs,
    ) -> None:
        self.material_properties = material_properties
        self.temperature = temperature
        self.electron_energies = self._generate_electron_energies(*args, **kwargs)

    number_of_electrons = None

    @property
    def hydrogen_energies(self):
        return self.material_properties.hydrogen_energies

    @property
    def hydrogen_energies_for_simulation(self):
        return self.hydrogen_energies

    @property
    def hydrogen_energy_difference(self):
        return self.hydrogen_energies[1] - self.hydrogen_energies[0]

    @property
    def fermi_wavevector(self):
        return self.material_properties.fermi_wavevector

    @property
    def boltzmann_energy(self):
        return self.temperature * scipy.constants.Boltzmann

    @abstractmethod
    def _generate_electron_energies(self, *args, **kwargs):
        pass

    @abstractmethod
    def _get_energy_spacing(self):
        pass

    @staticmethod
    def _calculate_electron_energies(k_states):
        return (scipy.constants.hbar * k_states) ** 2 / (2 * scipy.constants.m_e)

    def _get_fermi_energy(self):
        return self._calculate_electron_energies(
            self.material_properties.fermi_wavevector
        )

    # @abstractmethod
    # def _get_wavevector_spacing(self):
    #     pass

    # def _get_implied_volume_old(self):
    #     # old method using wavevector spacing instead of energy
    #     wavevector_spacing = self._get_wavevector_spacing()

    #     return ((scipy.constants.pi ** 2) /
    #             (wavevector_spacing *
    #              self.material_properties.fermi_wavevector ** 2))

    def _get_implied_volume(self):
        energy_spacing = self._get_energy_spacing()

        prefactor = (scipy.constants.pi ** 2) * (
            scipy.constants.hbar ** 2 / scipy.constants.m_e
        ) ** (3 / 2)
        energy_factor = (2 * self._get_fermi_energy()) ** (-1 / 2)

        return prefactor * energy_factor / energy_spacing

    def _get_interaction_prefactor(self):
        implied_volume = self._get_implied_volume()
        # Ignoring q dependance the interaction
        # takes the form -2e^2 / epsilon_0 alpha^2 (see 6.9 on lab book)
        alpha = 3.77948796 * 10 ** 10  # m^-1
        potential_factor = (
            -2 * (scipy.constants.e ** 2) / (scipy.constants.epsilon_0 * (alpha ** 2))
        )
        prefactor = 4 * np.pi * potential_factor / implied_volume
        return prefactor

    def _get_energy_jitter(self):
        return 0

    def _create_simulation(self, jitter_electrons=False):
        electron_energy_jitter = self._get_energy_jitter() if jitter_electrons else 0
        sim = ElectronSimulation(
            ElectronSimulationConfig(
                hbar=scipy.constants.hbar,
                boltzmann_energy=self.boltzmann_energy,
                electron_energies=self.electron_energies,
                hydrogen_energies=self.hydrogen_energies_for_simulation,
                block_factors=self.material_properties.hydrogen_overlaps,
                q_prefactor=self._get_interaction_prefactor(),
                electron_energy_jitter=electron_energy_jitter,
                number_of_electrons=self.number_of_electrons,
            )
        )
        return sim

    def simulate_material(self, times, jitter_electrons=False):

        sim = self._create_simulation(jitter_electrons)

        ElectronSimulationPlotter.plot_random_system_evolved_coherently(
            sim,
            times,
            thermal=True,
        )

    def simulate_average_material(
        self, times, average_over=10, jitter_electrons=False, **kwargs
    ):

        sim = self._create_simulation(jitter_electrons)

        ElectronSimulationPlotter.plot_average_densities_of_system_evolved_coherently(
            sim,
            times,
            average_over=average_over,
            thermal=True,
            jitter_for_each=jitter_electrons,
            **kwargs,
        )

        # ElectronSimulationPlotter.plot_tunneling_overlaps(sim)
