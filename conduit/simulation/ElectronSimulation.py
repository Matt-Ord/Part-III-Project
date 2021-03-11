from __future__ import annotations

from typing import Dict, List, NamedTuple
from simulation.Hamiltonian import Hamiltonian, HamiltonianUtil
import numpy as np
import matplotlib.pyplot as plt
from simulation.ElectronSystem import ElectronSystem, ElectronSystemUtil
import matplotlib as mpl


class ElectronSimulationConfig(NamedTuple):
    hbar: float
    electron_energies: List[float]
    hydrogen_energies: List[float]
    block_factors: List[List[float]] = [[0, 0], [0, 0]]
    q_prefactor: float = 1


class ElectronSimulation:
    def __init__(self, config: ElectronSimulationConfig) -> None:
        self.config = config

        self.hamiltonian = self._create_hamiltonian()

    @property
    def number_of_electron_states(self):
        return len(self.config.electron_energies)

    @property
    def number_of_electrons(self):
        return int(self.number_of_electron_states / 2)

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
    def hbar(self):
        return self.config.hbar

    @staticmethod
    def get_electron_densities(electron_states):
        electron_densities = [
            state.get_electron_density_for_each_hydrogen() for state in electron_states
        ]
        return electron_densities

    def get_electron_states(self, initial_system: ElectronSystem, times: List[float]):
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

    def _create_hamiltonian(self) -> Hamiltonian:
        dummy_system = self._setup_random_initial_state()

        kinetic_hamiltonian = ElectronSystemUtil.given(dummy_system).create_kinetic(
            Hamiltonian,
            self.electron_energies,
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

    def _setup_explicit_initial_state(self):
        initial_electron_state_vector = np.zeros(self.number_of_electron_states)
        initial_electron_state_vector[: self.number_of_electrons] = 1

        initial_state = ElectronSystemUtil.create_explicit(
            ElectronSystem, initial_electron_state_vector, 0
        )
        return initial_state

    def _setup_random_initial_state(self):
        hydrogen_state = 0

        initial_state = ElectronSystemUtil.create_random(
            ElectronSystem,
            self.number_of_electron_states,
            self.number_of_electrons,
            hydrogen_state,
        )
        return initial_state

    def simulate_system_coherently(self, times: List[float]):
        initial_state = self._setup_explicit_initial_state()

        electron_densities = self.get_electron_densities(
            self.get_electron_states(initial_state, times)
        )

        return electron_densities

    def simulate_random_system_coherently(self, times: List[float]):
        initial_state = self._setup_random_initial_state()

        electron_densities = self.get_electron_densities(
            self.get_electron_states(initial_state, times)
        )

        return electron_densities

    def simulate_system_decoherently(self, times: List[float]):
        initial_state = self._setup_explicit_initial_state()

        electron_densities = self.get_electron_densities(
            self.get_electron_states_decoherently(initial_state, times)
        )

        return electron_densities

    def _calculate_densities_for_each(self, initial_states, times):
        electron_densities = [
            self.get_electron_densities(self.get_electron_states(initial_state, times))
            for initial_state in initial_states
        ]
        return np.array(electron_densities)

    def simulate_random_system_coherently_for_each(
        self,
        times: List[float],
        average_over: int = 5,
    ):
        initial_states = [
            self._setup_random_initial_state() for _ in range(average_over)
        ]

        electron_densities_for_each = self._calculate_densities_for_each(
            initial_states, times
        )

        return electron_densities_for_each

    def get_tunnelling_overlaps(self):
        empty_state = np.zeros(self.number_of_electron_states * )
        state_1 = 
        HamiltonianUtil.characterise_overlap(
            self.hamiltonian,

        )
        self.hamiltonian.
