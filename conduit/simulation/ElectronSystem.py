from typing import Callable, List, final
import numpy as np
import scipy.constants
from simulation.Hamiltonian import Hamiltonian, HamiltonianUtil


class ElectronSystem:
    def __init__(self, system_vector: np.ndarray, electron_basis_states) -> None:
        self.electron_basis_states = np.array(electron_basis_states)
        self.system_vector = system_vector

    def evolve_system(self, hamiltonian: Hamiltonian, time, hbar=1):
        evolved_system_vector = hamiltonian.evolve_system_vector(
            self.system_vector, time, hbar
        )
        return type(self)(evolved_system_vector, self.electron_basis_states)

    def evolve_system_vectorised(self, hamiltonian: Hamiltonian, times, hbar=1):
        evolved_system_vectors = hamiltonian.evolve_system_vector_vectorised(
            self.system_vector, times, hbar
        )
        return [
            type(self)(evolved_system_vector, self.electron_basis_states)
            for evolved_system_vector in evolved_system_vectors
        ]

    def get_electron_density_for_each_hydrogen(self) -> np.ndarray:
        return self.get_probability_for_each_hydrogen().dot(
            self.electron_basis_states
        )  # type:ignore

    def get_probability_for_each_hydrogen(self):
        return np.array(self._get_state_probabilities()).reshape(2, -1)

    def _get_state_probabilities(self):
        return np.abs(self.system_vector) ** 2

    def get_normalisation(self):
        return np.sum(self._get_state_probabilities())

    def get_number_of_electron_states(self):
        return self.electron_basis_states.shape[0]

    def get_number_of_electrons(self):
        return self.electron_basis_states.shape[1]

    def get_density_matrix(self) -> np.ndarray:
        vec = self.system_vector[np.newaxis]
        return np.matmul(vec.conj().T, vec)

    def get_electron_density_matrix(self) -> np.ndarray:
        density_matrix = self.get_density_matrix()
        # Take the trace
        electron_density_matrix = (
            density_matrix[
                : self.get_number_of_electron_states(),
                : self.get_number_of_electron_states(),
            ]
            + density_matrix[
                self.get_number_of_electron_states() :,
                self.get_number_of_electron_states() :,
            ]
        )
        return electron_density_matrix

    @staticmethod
    def _overlap_of_state(state_1, state_2):
        product = np.multiply(state_1, np.conj(state_2))
        overlap_amplitude = np.sum(product)
        overlap = np.abs(overlap_amplitude) ** 2
        return overlap

    def get_summed_overlap_fraction_of_eigenstates(
        self, hamiltonian: Hamiltonian
    ) -> List[float]:

        summed_overlaps = [
            np.sum([state[0] for state in eigenstate])
            for eigenstate in self.get_overlap_fraction_of_eigenstates(hamiltonian)
        ]
        return summed_overlaps  # type: ignore

    def get_overlap_fraction_of_eigenstates(self, hamiltonian: Hamiltonian):

        initial_states = np.identity(self.get_number_of_electron_states() * 2)[
            0 : self.get_number_of_electron_states()
        ]
        final_states = np.identity(self.get_number_of_electron_states() * 2)[
            self.get_number_of_electron_states() :
        ]

        overlaps = [
            [
                (
                    self._overlap_of_state(eigenstate, initial_state),
                    self._overlap_of_state(eigenstate, final_state),
                )
                for (initial_state, final_state) in zip(initial_states, final_states)
            ]
            for eigenstate in hamiltonian.eigenvectors.T
        ]
        return overlaps

    def __str__(self) -> str:
        return (
            f"state vector:\n{self.system_vector}\n"
            f"basis:\n{self.electron_basis_states}\n"
        )


class ElectronSystemUtil:
    @classmethod
    def _generate_electron_basis(cls, number_of_states, number_of_electrons):
        if number_of_states < number_of_electrons or number_of_electrons < 0:
            return []
        if number_of_states == 0:
            return [[]]
        return [
            [0] + x
            for x in cls._generate_electron_basis(
                number_of_states - 1, number_of_electrons
            )
        ] + [
            [1] + x
            for x in cls._generate_electron_basis(
                number_of_states - 1, number_of_electrons - 1
            )
        ]

    @classmethod
    def _create_explicit_state_vector(
        cls, electron_basis_states, electron_state, hydrogen_state
    ):
        number_of_electron_states = len(electron_state)

        electron_state_index = electron_basis_states.tolist().index(
            list(electron_state)
        )
        number_of_basis_states = 2 * len(electron_basis_states)

        overall_index = (
            electron_state_index + number_of_electron_states * hydrogen_state
        )

        state_vector = np.zeros(number_of_basis_states)
        state_vector[overall_index] = 1
        return state_vector

    @classmethod
    def create_explicit(cls, system, electron_state, hydrogen_state):
        number_of_electron_states = len(electron_state)
        number_of_electrons = sum(electron_state)
        electron_basis_states = np.array(
            cls._generate_electron_basis(number_of_electron_states, number_of_electrons)
        )

        state_vector = cls._create_explicit_state_vector(
            electron_basis_states, electron_state, hydrogen_state
        )

        return system(state_vector, electron_basis_states)

    # Creates a random state with probabilities weighted
    # exponentially by the relevant boltzmann factors
    @staticmethod
    def _create_random_state_vector(
        electron_boltzmann_factors, hydrogen_boltzmann_factors, initial_occupancy
    ):
        boltzmann_probabilities = np.exp(-electron_boltzmann_factors / 2)
        hydrogen_boltzmann_probabilities = np.exp(-hydrogen_boltzmann_factors / 2)
        state_probabilities = np.array(
            [
                probability * np.random.normal() * np.exp(2j * np.pi * np.random.rand())
                for probability in boltzmann_probabilities
            ]
        )

        overall_state = np.concatenate(
            [
                initial_occupancy
                * state_probabilities
                * hydrogen_boltzmann_probabilities[0],
                (1 - initial_occupancy)
                * state_probabilities
                * hydrogen_boltzmann_probabilities[1],
            ]
        )
        normalised_overall_state = overall_state / np.linalg.norm(overall_state)
        return normalised_overall_state

    # Calculates the boltzmann factors for the
    # multi electron states
    @staticmethod
    def _calculate_multi_electron_boltzmann_factors(
        single_electron_boltzmann_factors, electron_basis_states
    ):
        multi_electron_boltzmann_factors = np.array(
            [
                sum(single_electron_boltzmann_factors * state)
                for state in electron_basis_states
            ]
        )

        return multi_electron_boltzmann_factors

    @classmethod
    def create_random(
        cls,
        system,
        number_of_electron_states,
        number_of_electrons,
        electron_boltzmann_factors=None,
        hydrogen_boltzmann_factors=None,
        initial_occupancy=1,
    ):
        if electron_boltzmann_factors is None:
            electron_boltzmann_factors = np.zeros(number_of_electron_states)

        if hydrogen_boltzmann_factors is None:
            hydrogen_boltzmann_factors = np.zeros(2)

        electron_basis_states = cls._generate_electron_basis(
            number_of_electron_states, number_of_electrons
        )

        multi_electron_boltzmann_factors = (
            cls._calculate_multi_electron_boltzmann_factors(
                electron_boltzmann_factors, np.array(electron_basis_states)
            )
        )

        state_vector = cls._create_random_state_vector(
            multi_electron_boltzmann_factors,
            hydrogen_boltzmann_factors,
            initial_occupancy,
        )

        return system(state_vector, electron_basis_states)


class ElectronSystemHamiltonianFactory:
    def __init__(self, system: ElectronSystem) -> None:
        self.system = system

    def get_number_of_electron_states(self):
        return self.system.get_number_of_electron_states()

    def get_number_of_electrons(self):
        return self.system.get_number_of_electrons()

    def get_electron_basis_states(self):
        return self.system.electron_basis_states

    def create_kinetic(self, hamiltonian, electron_energies, hydrogen_energies):
        basis_states = self.get_electron_basis_states()
        basis_electron_energies = np.sum(
            np.array(electron_energies) * np.array(basis_states), axis=1
        )

        basis_energies = np.concatenate(
            [
                basis_electron_energies + hydrogen_energy
                for hydrogen_energy in hydrogen_energies
            ]
        )

        return HamiltonianUtil.create_diagonal(hamiltonian, basis_energies)

    def create_block_identity(self, hamiltonian, block_factors):
        states_in_each_block = self.get_number_of_electron_states()
        return HamiltonianUtil.create_block_identity(
            hamiltonian, states_in_each_block, block_factors
        )

    def create_constant_interaction(self, hamiltonian, block_factors, q_prefactor=1):
        k_values = np.zeros(self.get_number_of_electrons())

        def q_factor(x):
            return q_prefactor

        return self.create_q_dependent_interaction(
            hamiltonian, block_factors, k_values, q_factor
        )

    def create_q_dependent_interaction(
        self, hamiltonian, block_factors, k_values, q_factor
    ):
        base_matrix = self._generate_single_hop_q_dependant_base_matrix(
            np.array(k_values), q_factor
        )

        return HamiltonianUtil.create_block(hamiltonian, base_matrix, block_factors)

    @classmethod
    def _calculate_q_dependant_single_hop_strength(
        cls,
        state_a: np.ndarray,
        state_b: np.ndarray,
        k_values: np.ndarray,
        q_factor: Callable[[float], float],
    ) -> float:
        differences = state_a - state_b
        if np.count_nonzero(differences) > 2:
            # More than 1 hop not allowed
            return 0
        k_differences = k_values * differences
        q = np.abs(sum(k_differences))
        if np.count_nonzero(differences) != 2:
            return q_factor(q)
        return q_factor(q) * cls._exchange_sign(state_a, state_b)

    # calculates the fermion exchange sign
    # assuming only one exchange
    @staticmethod
    def _exchange_sign(initial_electron_state, final_electron_state):
        difference = initial_electron_state - final_electron_state

        exchanged_index = np.argwhere(difference != 0)[:, 0]
        number_of_exchanged_electrons = np.count_nonzero(
            initial_electron_state[exchanged_index[0] + 1 : exchanged_index[1]]
        )
        if number_of_exchanged_electrons % 2 == 0:
            return 1
        return -1

    @classmethod
    def _has_at_most_one_hop(cls, state_a, state_b):
        return cls._calculate_q_dependant_single_hop_strength(
            state_a, state_b, np.zeros_like(state_a), lambda x: 1
        )

    def _generate_single_hop_constant_base_matrix(self):
        return self._generate_base_matrix(strength_function=self._has_at_most_one_hop)

    def _generate_single_hop_q_dependant_base_matrix(self, k_values, q_factor):
        def strength_function(a, b):
            return self._calculate_q_dependant_single_hop_strength(
                a, b, k_values, q_factor
            )

        return self._generate_base_matrix(strength_function)

    def _generate_base_matrix(
        self, strength_function: Callable[[np.ndarray, np.ndarray], float]
    ) -> np.ndarray:
        states_in_each_block = self.get_number_of_electron_states()
        electron_basis_states = self.get_electron_basis_states()
        a = np.zeros((states_in_each_block, states_in_each_block), dtype=np.double)
        for x in range(states_in_each_block):
            for y in range(states_in_each_block):
                a[x, y] = strength_function(
                    electron_basis_states[x], electron_basis_states[y]
                )
        return a