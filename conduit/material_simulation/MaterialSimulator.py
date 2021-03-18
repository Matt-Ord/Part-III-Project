from abc import ABC, abstractmethod
from typing import List
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
    def _generate_electron_energies(self, *args, **kwargs) -> List[float]:
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

    def _occupation_curve_fit_function(self, time, omega, decay_time):
        return ElectronSimulationPlotter.occupation_curve_fit_function(
            self.number_of_electrons, time, omega, decay_time
        )

    def _fit_electron_occupation_curve(self, times, initially_occupied_densities):
        initial_omega_guess = 10 / (times[-1] - times[0])
        initial_decay_time_guess = (times[-1] - times[0]) / 4

        return scipy.optimize.curve_fit(
            lambda t, w, d: self._occupation_curve_fit_function,
            times,
            initially_occupied_densities,
            p0=[initial_omega_guess, initial_decay_time_guess],
        )

    def simulate_average_densities(
        self, times, average_over=10, jitter_electrons=False, **kwargs
    ):
        sim = self._create_simulation(jitter_electrons)
        electron_densities_for_each = sim.simulate_random_system_coherently_for_each(
            times,
            average_over=average_over,
            thermal=True,
            jitter_for_each=jitter_electrons,
            **kwargs,
        )

        average_densities = ElectronSimulationPlotter.calculate_average_density(
            electron_densities_for_each
        )
        return average_densities

        initially_occupied_densities = [d[0] for d in average_densities]
        initially_unoccupied_densities = [d[1] for d in average_densities]

        (optimal_omega, optimal_decay_time), pcov = self._fit_electron_occupation_curve(
            times, initially_occupied_densities
        )

        print("omega", optimal_omega)
        print("decay_time", optimal_decay_time)
        print(pcov)
