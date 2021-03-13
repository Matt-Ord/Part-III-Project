from __future__ import annotations

from typing import Dict, List, NamedTuple

from simulation.Hamiltonian import Hamiltonian, HamiltonianUtil
import numpy as np
import matplotlib.pyplot as plt
from simulation.ElectronSystem import ElectronSystem, ElectronSystemUtil


class ElectronSimulationConfig(NamedTuple):
    hbar: float
    boltzmann_energy: float
    electron_energies: List[float]
    hydrogen_energies: List[float]
    block_factors: List[List[float]] = [[0, 0], [0, 0]]
    q_prefactor: float = 1
    electron_energy_jitter: float = 0
    number_of_electrons: int = None


class ElectronSimulation:
    def __init__(self, config: ElectronSimulationConfig) -> None:
        self.config = config

        self.hamiltonian = self._create_hamiltonian()

    @property
    def number_of_electron_states(self):
        return len(self.config.electron_energies)

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

    @staticmethod
    def get_electron_densities(electron_states):
        electron_densities = [
            state.get_electron_density_for_each_hydrogen() for state in electron_states
        ]
        return electron_densities

    def get_electron_states(
        self,
        initial_system: ElectronSystem,
        times: List[float],
        new_hamiltonian: bool = False,
    ):
        if new_hamiltonian:
            self.hamiltonian = self._create_hamiltonian()

        evolved_states = [
            initial_system.evolve_system(self.hamiltonian, time, self.hbar)
            for time in times
        ]
        return evolved_states

    def get_electron_states_decoherently(
        self, initial_system: ElectronSystem, times: List[float]
    ):
        timesteps = [end - start for (start, end) in zip(times[:-1], times[1:])]

        evolved_states = [initial_system]
        for t in timesteps:
            evolved_states.append(
                evolved_states[-1].evolve_system_decoherently(
                    self.hamiltonian,
                    t,
                    self.hbar,
                )
            )
        return evolved_states

    @staticmethod
    def _randomise_energies(energies: np.ndarray, scale):
        return energies + np.random.normal(loc=0.0, scale=scale, size=energies.size)

    def _create_hamiltonian(self) -> Hamiltonian:
        dummy_system = self._setup_random_initial_system()
        electron_energies = self._randomise_energies(
            self.electron_energies, self.electron_energy_jitter
        )

        kinetic_hamiltonian = ElectronSystemUtil.given(dummy_system).create_kinetic(
            Hamiltonian,
            electron_energies,
            self.hydrogen_energies,
        )

        interaction_hamiltonian = ElectronSystemUtil.given(
            dummy_system
        ).create_constant_interaction(Hamiltonian, self.block_factors, self.q_prefactor)

        print("kinetic_energy", kinetic_hamiltonian[0, 0])
        print("interaction_energy", interaction_hamiltonian[0, 0])

        hamiltonian = kinetic_hamiltonian + interaction_hamiltonian
        hamiltonian.save_as_csv("hamiltonian.csv")
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
        hydrogen_state = 0

        boltzmann_factors = None
        if thermal:
            energy_offsets = self.electron_energies - np.average(self.electron_energies)
            boltzmann_factors = energy_offsets / self.boltzmann_energy
            print(boltzmann_factors)

        initial_system = ElectronSystemUtil.create_random(
            ElectronSystem,
            self.number_of_electron_states,
            self.number_of_electrons,
            hydrogen_state,
            boltzmann_factors,
        )
        return initial_system

    def simulate_system_coherently(
        self, times: List[float], initial_electron_state_vector=None
    ):
        initial_system = self._setup_explicit_initial_system(
            initial_electron_state_vector
        )

        electron_densities = self.get_electron_densities(
            self.get_electron_states(initial_system, times)
        )

        return electron_densities

    def simulate_random_system_coherently(self, times: List[float], thermal=False):
        initial_system = self._setup_random_initial_system(thermal)

        electron_densities = self.get_electron_densities(
            self.get_electron_states(initial_system, times)
        )

        return electron_densities

    def simulate_system_decoherently(
        self, times: List[float], initial_electron_state_vector=None
    ):
        initial_system = self._setup_explicit_initial_system(
            initial_electron_state_vector
        )

        electron_densities = self.get_electron_densities(
            self.get_electron_states_decoherently(initial_system, times)
        )

        return electron_densities

    def _calculate_densities_for_each(self, initial_systems, times, jitter_for_each):
        electron_densities = [
            self.get_electron_densities(
                self.get_electron_states(initial_system, times, jitter_for_each)
            )
            for initial_system in initial_systems
        ]
        return np.array(electron_densities)

    def simulate_random_system_coherently_for_each(
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

    def characterise_tunnelling_overlaps(self):
        dummy_system = self._setup_random_initial_system()
        overlaps = ElectronSystemUtil.given(
            dummy_system
        ).characterise_tunnelling_overlaps(hamiltonian=self.hamiltonian)
        return overlaps
