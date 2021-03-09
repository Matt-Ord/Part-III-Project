from typing import Callable, Tuple
import numpy as np
from Hamiltonian import Hamiltonian, HamiltonianUtil


class ElectronSystem():

    def __init__(self, system_vector, electron_basis_states) -> None:
        self.electron_basis_states = np.array(electron_basis_states)
        self.system_vector = system_vector

    def evolve_system(self, hamiltonian: Hamiltonian, time, hbar=1):
        evolved_system_vector = hamiltonian.evolve_system_vector(
            self.system_vector,
            time,
            hbar
        )
        return type(self)(evolved_system_vector, self.electron_basis_states)

    def evolve_system_decoherently(
        self,
        hamiltonian: Hamiltonian,
        time,
        hbar=1
    ):
        evolved_system_vector = hamiltonian.evolve_system_vector(
            self.system_vector,
            time,
            hbar
        )
        evolved_system_vector = evolved_system_vector * \
            np.exp(2j * np.pi * np.random.rand(*evolved_system_vector.shape))

        return type(self)(evolved_system_vector, self.electron_basis_states)

    def get_electron_density_for_each_hydrogen(self):
        return self.get_probability_for_each_hydrogen() \
            .dot(self.electron_basis_states)

    def get_probability_for_each_hydrogen(self):
        return np.array(
            self._get_state_probabilities()).reshape(2, -1)

    def _get_state_probabilities(self):
        return np.abs(self.system_vector) ** 2

    def get_number_of_electron_states(self):
        return self.electron_basis_states.shape[0]

    def get_number_of_electrons(self):
        return self.electron_basis_states.shape[1]

    def __str__(self) -> str:
        return (f"state vector:\n{self.system_vector}\n"
                f"basis:\n{self.electron_basis_states}\n")


class ElectronSystemUtil():

    def __init__(self, system: ElectronSystem) -> None:
        self.system = system

    @ classmethod
    def _generate_electron_basis(cls, number_of_states, number_of_electrons):
        if (number_of_states < number_of_electrons or number_of_electrons < 0):
            return []
        if number_of_states == 0:
            return [[]]
        return ([[0] + x for x in
                 cls._generate_electron_basis(number_of_states - 1,
                                              number_of_electrons)]
                + [[1] + x for x in
                   cls._generate_electron_basis(number_of_states - 1,
                                                number_of_electrons - 1)]
                )

    def _create_explicit_state_vector(
        cls,
        electron_basis_states,
        electron_state,
        hydrogen_state
    ):
        number_of_electron_states = len(electron_state)

        electron_state_index = electron_basis_states.index(
            list(electron_state))
        overall_index = electron_state_index + \
            number_of_electron_states * hydrogen_state

        state_vector = np.zeros(2*number_of_electron_states)
        state_vector[overall_index] = 1
        return state_vector

    @classmethod
    def _create_random_state_vector(cls, electron_basis_states):
        random_state = np.concatenate(
            [np.random.normal(size=len(electron_basis_states))
             * np.exp(2j * np.pi * np.random.rand(len(electron_basis_states))),
             np.zeros(len(electron_basis_states))])

        normalised_random_state = random_state / \
            np.linalg.norm(random_state)
        return normalised_random_state

    @ classmethod
    def create_explicit(cls, system, electron_state, hydrogen_state):
        number_of_electron_states = len(electron_state)
        number_of_electrons = sum(electron_state)
        electron_basis_states = cls._generate_electron_basis(
            number_of_electron_states,
            number_of_electrons
        )

        state_vector = cls._create_explicit_state_vector(
            electron_basis_states,
            electron_state,
            hydrogen_state
        )

        return system(state_vector, electron_basis_states)

    @ classmethod
    def create_random(
        cls,
        system,
        number_of_electron_states,
        number_of_electrons,
        hydrogen_state
    ):

        electron_basis_states = cls._generate_electron_basis(
            number_of_electron_states,
            number_of_electrons
        )

        state_vector = cls._create_random_state_vector(
            electron_basis_states
        )

        return system(state_vector, electron_basis_states)

    def create_kinetic(
        self,
        hamiltonian,
        electron_energies,
        hydrogen_energies
    ):
        basis_states = self.system.electron_basis_states

        basis_electron_energies = np.sum(np.array(electron_energies) *
                                         np.array(basis_states), axis=1)

        basis_energies = np.concatenate(
            [basis_electron_energies + hydrogen_energy
             for hydrogen_energy in hydrogen_energies])

        return HamiltonianUtil.create_diagonal(
            hamiltonian,
            basis_energies
        )

    def create_block_identity(self, hamiltonian, block_factors):
        states_in_each_block = self.system.get_number_of_electron_states()
        return HamiltonianUtil.create_block_identity(
            hamiltonian,
            states_in_each_block,
            block_factors
        )

    @staticmethod
    def _calculate_q_dependant_single_hop_strength(
            state_a: np.ndarray,
            state_b: np.ndarray,
            k_values: np.ndarray,
            q_factor: Callable[[float], float]
    ) -> float:
        differences = state_a - state_b
        if(np.count_nonzero(differences) > 2):
            # More than 1 hop not allowed
            return 0
        k_differences = k_values * differences
        q = np.abs(sum(k_differences))
        return q_factor(q)

    @classmethod
    def _has_at_most_one_hop(cls, state_a, state_b):
        return cls._calculate_q_dependant_single_hop_strength(
            state_a,
            state_b,
            np.zeros_like(state_a),
            lambda x: 1
        )

    def _generate_single_hop_constant_base_matrix(self):
        return self._generate_base_matrix(
            strength_function=self._has_at_most_one_hop
        )

    def _generate_single_hop_q_dependant_base_matrix(self, k_values, q_factor):
        def strength_function(a, b):
            return self._calculate_q_dependant_single_hop_strength(
                a, b, k_values, q_factor)
        return self._generate_base_matrix(
            strength_function
        )

    def _generate_base_matrix(
        self,
        strength_function: Callable[[np.ndarray, np.ndarray], bool]
    ) -> np.ndarray:
        states_in_each_block = self.system.get_number_of_electron_states()
        electron_basis_states = self.system.electron_basis_states
        a = np.zeros((states_in_each_block, states_in_each_block),
                     dtype=np.double)
        for x in range(states_in_each_block):
            for y in range(states_in_each_block):
                a[x, y] = strength_function(
                    electron_basis_states[x],
                    electron_basis_states[y])
        return a

    def create_constant_interaction(self, hamiltonian, block_factors):
        k_values = np.zeros(self.system.get_number_of_electrons())
        def q_factor(x): return 1
        return self.create_q_dependent_interaction(
            hamiltonian,
            block_factors,
            k_values,
            q_factor
        )

    def create_q_dependent_interaction(
            self,
            hamiltonian,
            block_factors,
            k_values,
            q_factor
    ):
        base_matrix = self._generate_single_hop_q_dependant_base_matrix(
            np.array(k_values), q_factor)

        return HamiltonianUtil.create_block(
            hamiltonian,
            base_matrix,
            block_factors
        )

    @classmethod
    def given(cls, system: ElectronSystem):
        return cls(system)
