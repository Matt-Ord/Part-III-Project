from __future__ import annotations

from typing import Callable, List, NamedTuple

from simulation.Hamiltonian import Hamiltonian
import numpy as np
from simulation.ElectronSystem import (
    ElectronSystem,
    ElectronSystemHamiltonianFactory,
    ElectronSystemUtil,
)


def randomise_electron_energies(energies: np.ndarray, scale):
    return energies + np.random.normal(loc=0.0, scale=scale, size=energies.size)


class ElectronSimulationConfig(NamedTuple):
    hbar: float
    boltzmann_energy: float
    electron_energies: List[float]
    hydrogen_energies: List[float]
    block_factors: List[List[complex]] = [[0, 0], [0, 0]]
    q_prefactor: float = 1
    electron_energy_jitter: Callable[[np.ndarray], np.ndarray] = lambda x: x
    number_of_electrons: int | None = None
    initial_occupancy: float = 1


class ElectronSimulation:
    def __init__(self, config: ElectronSimulationConfig) -> None:
        self.config = config

        self.hamiltonian = self._create_hamiltonian()

    @property
    def number_of_electron_states(self):
        return len(self.config.electron_energies)

    @property
    def number_of_electron_basis_states(self):
        dummy_sim = self._setup_random_initial_system()
        return dummy_sim.get_number_of_electron_states()

    @property
    def number_of_electrons(self):
        if self.config.number_of_electrons is None:
            if self.number_of_electron_states % 2 == 0:
                return int(self.number_of_electron_states / 2)
            else:  # choose to round up or down
                return np.floor(self.number_of_electron_states / 2) + np.random.choice(
                    [0, 1]
                )

        return self.config.number_of_electrons

    @property
    def electron_energies(self):
        return self.config.electron_energies

    @property
    def hydrogen_energies(self):
        return self.config.hydrogen_energies

    @property
    def block_factors(self):
        return self.config.block_factors

    @property
    def q_prefactor(self):
        return self.config.q_prefactor

    @property
    def electron_energy_jitter(self):
        return self.config.electron_energy_jitter

    @property
    def hbar(self):
        return self.config.hbar

    @property
    def boltzmann_energy(self):
        return self.config.boltzmann_energy

    @property
    def initial_occupancy(self):
        return self.config.initial_occupancy

    @staticmethod
    def _get_electron_densities(electron_systems: List[ElectronSystem]):
        electron_densities = [
            system.get_electron_density_for_each_hydrogen()
            for system in electron_systems
        ]
        return electron_densities

    def get_electron_systems(
        self,
        initial_system: ElectronSystem,
        times: List[float],
        new_hamiltonian: bool = False,
    ):
        if new_hamiltonian:
            self.hamiltonian = self._create_hamiltonian()

        evolved_systems = initial_system.evolve_system_vectorised(
            self.hamiltonian, times, self.hbar
        )
        return evolved_systems

    def _create_kinetic_hamiltonian(self) -> Hamiltonian:
        dummy_system = self._setup_random_initial_system()
        electron_energies = self.electron_energy_jitter(
            np.array(self.electron_energies)
        )

        kinetic_hamiltonian = ElectronSystemHamiltonianFactory(
            dummy_system
        ).create_kinetic(
            Hamiltonian,
            electron_energies,
            self.hydrogen_energies,
        )
        return kinetic_hamiltonian

    def _create_interaction_hamiltonian(self) -> Hamiltonian:
        dummy_system = self._setup_random_initial_system()

        interaction_hamiltonian = ElectronSystemHamiltonianFactory(
            dummy_system
        ).create_constant_interaction(Hamiltonian, self.block_factors, self.q_prefactor)
        return interaction_hamiltonian

    def _create_hamiltonian(self) -> Hamiltonian:

        kinetic_hamiltonian = self._create_kinetic_hamiltonian()
        interaction_hamiltonian = self._create_interaction_hamiltonian()

        # print("kinetic_energy", kinetic_hamiltonian[0, 0])
        # print("interaction_energy", interaction_hamiltonian[0, 0])

        hamiltonian = kinetic_hamiltonian + interaction_hamiltonian

        return hamiltonian

    def _setup_explicit_initial_system(self, initial_electron_state_vector=None):
        if initial_electron_state_vector is None:
            initial_electron_state_vector = np.zeros(self.number_of_electron_states)
            initial_electron_state_vector[: self.number_of_electrons] = 1

        initial_system = ElectronSystemUtil.create_explicit(
            ElectronSystem, initial_electron_state_vector, 0
        )
        return initial_system

    def _setup_random_initial_system(self, thermal=False):
        electron_boltzmann_factors = None
        hydrogen_boltzmann_factors = None
        if thermal:
            electron_energy_offsets = self.electron_energies - np.average(
                self.electron_energies
            )
            electron_boltzmann_factors = electron_energy_offsets / self.boltzmann_energy

            hydrogen_energy_offsets = self.hydrogen_energies - np.average(
                self.hydrogen_energies
            )
            hydrogen_boltzmann_factors = hydrogen_energy_offsets / self.boltzmann_energy

        initial_system = ElectronSystemUtil.create_random(
            ElectronSystem,
            number_of_electron_states=self.number_of_electron_states,
            number_of_electrons=self.number_of_electrons,
            electron_boltzmann_factors=electron_boltzmann_factors,
            hydrogen_boltzmann_factors=hydrogen_boltzmann_factors,
            initial_occupancy=self.initial_occupancy,
        )
        return initial_system

    def get_electron_densities(
        self,
        times: List[float],
        thermal=False,
        initial_electron_state_vector=None,
        new_hamiltonian=False,
    ):

        if initial_electron_state_vector is not None:
            initial_system = self._setup_explicit_initial_system(
                initial_electron_state_vector
            )
        else:
            initial_system = self._setup_random_initial_system(thermal)

        electron_densities = self._get_electron_densities(
            self.get_electron_systems(initial_system, times, new_hamiltonian)
        )

        return electron_densities

    @staticmethod
    def _get_normalisations(electron_systems: List[ElectronSystem]):
        electron_densities = [system.get_normalisation() for system in electron_systems]
        return electron_densities

    def get_normalisations(self, times: List[float], thermal=False):
        initial_system = self._setup_random_initial_system(thermal)

        normalisation = self._get_normalisations(
            self.get_electron_systems(initial_system, times)
        )

        return normalisation

    def _calculate_densities_for_each(self, initial_systems, times, jitter_for_each):
        electron_densities = [
            self._get_electron_densities(
                self.get_electron_systems(initial_system, times, jitter_for_each)
            )
            for initial_system in initial_systems
        ]
        return np.array(electron_densities)

    def get_electron_densities_for_each(
        self,
        times: List[float],
        average_over: int = 5,
        thermal: bool = False,
        jitter_for_each: bool = False,
    ):
        initial_systems = [
            self._setup_random_initial_system(thermal) for _ in range(average_over)
        ]

        electron_densities_for_each = self._calculate_densities_for_each(
            initial_systems, times, jitter_for_each
        )

        return electron_densities_for_each

    def get_energies_and_summed_overlaps(self):
        energies = self.hamiltonian.eigenvalues
        dummy_system = self._setup_random_initial_system()
        overlaps = dummy_system.get_summed_overlap_fraction_of_eigenstates(
            self.hamiltonian
        )
        return energies, overlaps

    def get_energies_and_overlaps(self):
        energies = self.hamiltonian.eigenvalues
        dummy_system = self._setup_random_initial_system()
        overlaps = dummy_system.get_overlap_fraction_of_eigenstates(self.hamiltonian)
        return energies, overlaps

    def get_energies_without_interaction(self):
        kinetic_hamiltonian = self._create_kinetic_hamiltonian()
        return np.diagonal(kinetic_hamiltonian.as_matrix())

    @staticmethod
    def _get_density_matricies(electron_systems: List[ElectronSystem]):
        density_matricies = [system.get_density_matrix() for system in electron_systems]
        return density_matricies

    def get_density_matricies(
        self, times: List[float], thermal: bool = False
    ) -> list[np.ndarray]:
        initial_system = self._setup_random_initial_system(thermal)

        density_matricies = self._get_density_matricies(
            self.get_electron_systems(initial_system, times)
        )

        return density_matricies

    @staticmethod
    def _get_electron_density_matricies(electron_systems: List[ElectronSystem]):
        density_matricies = [
            system.get_electron_density_matrix() for system in electron_systems
        ]
        return density_matricies

    def get_electron_density_matricies(
        self, times: List[float], thermal: bool = False
    ) -> list[np.ndarray]:
        initial_system = self._setup_random_initial_system(thermal)

        density_matricies = self._get_electron_density_matricies(
            self.get_electron_systems(initial_system, times)
        )

        return density_matricies


if __name__ == "__main__":
    config = ElectronSimulationConfig(
        hbar=1,
        electron_energies=np.linspace(0, 100, 12).tolist(),
        hydrogen_energies=[0, 0],
        boltzmann_energy=200,
        block_factors=[[1, 0.001], [0.001, 1]],
        q_prefactor=1,
    )
    simulator = ElectronSimulation(config)
    simulator.get_electron_densities(np.linspace(0, 1000, 5000).tolist())
    print("done")
