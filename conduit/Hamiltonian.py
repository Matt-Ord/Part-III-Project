from __future__ import annotations
import numpy as np


class Hamiltonian():
    _eigenvalues = None
    _eigenvectors = None

    def __init__(self, matrix_representation: np.ndarray) -> None:
        if not self._is_valid_matrix_shape(matrix_representation):
            raise Exception(
                f'matrix representation has the wrong shape:\
                    actual {matrix_representation.shape}, \
                    expected a square matrix')
        if not self._is_valid_matrix(matrix_representation):
            raise Exception(
                f'matrix representation is not hermitian')
        self._matrix_representation = matrix_representation.copy()

    @property
    def eigenvalues(self):
        if self._eigenvalues is None:
            self._calculate_eigenvalues_and_vectors()
        return self._eigenvalues

    @property
    def eigenvectors(self):
        if self._eigenvectors is None:
            self._calculate_eigenvalues_and_vectors()
        return self._eigenvectors

    def _calculate_eigenvalues_and_vectors(self):
        self._eigenvalues, self._eigenvectors = \
            np.linalg.eigh(self._matrix_representation)

    def _get_number_of_states(self):
        return self._matrix_representation.shape[0]

    def _is_valid_state_vector(self, vector):
        return vector.shape == (self._get_number_of_states(),)

    @staticmethod
    def _is_valid_matrix_shape(matrix_representation):
        return (len(matrix_representation.shape) == 2 and
                matrix_representation.shape[0] ==
                matrix_representation.shape[1])

    @staticmethod
    def _is_valid_matrix(matrix_representation):
        return np.all(
            np.conj(matrix_representation.T) == matrix_representation)

    def evolve_system_vector(self, intial_state_vector, time, hbar=1):
        initial_state_decomposition = self.get_eigen_decomposition_of_vector(
            intial_state_vector
        )

        final_state_decompositon = self.get_decomposition_after_time(
            initial_state_decomposition,
            time,
            hbar,
        )

        final_state = self.get_vector_of_eigen_decomposition(
            final_state_decompositon
        )
        return final_state

    def get_decomposition_after_time(self, decomposition, time, hbar=1):
        eigenvector_phase_shift = np.exp(1j * self.eigenvalues * time / hbar)
        return np.multiply(
            eigenvector_phase_shift, decomposition)

    def get_eigen_decomposition_of_vector(self, vector) -> np.ndarray:
        if not self._is_valid_state_vector(vector):
            raise Exception(
                f'state vector shape is wrong: actual {vector.shape}, \
                expected({self._get_number_of_states()},)')
        return np.linalg.solve(self.eigenvectors, vector)
        return np.linalg.inv(self.eigenvectors.T).dot(vector)

    def get_vector_of_eigen_decomposition(self, decomposition):
        return np.dot(self.eigenvectors, decomposition)
        return self.eigenvectors.T.dot(decomposition)

    def __add__(self, other: Hamiltonian) -> Hamiltonian:
        return type(self)(self._matrix_representation
                          + other._matrix_representation)

    def __mul__(self, other: complex) -> Hamiltonian:
        return type(self)(self._matrix_representation.copy() * other)

    __rmul__ = __mul__

    def __getitem__(self, key):
        # only allow access to the values in _matrix_representation
        # not the sub-array itself!
        if type(key) is not tuple or len(key) != 2:
            raise IndexError
        val = self._matrix_representation[key]
        return val

    def __str__(self) -> str:
        return self._matrix_representation.__str__()


class HamiltonianUtil():

    @staticmethod
    def create_random(cls, number_of_states: int) -> Hamiltonian:
        random_matrix = np.random.rand(
            number_of_states,
            number_of_states) \
            + np.random.rand(
                number_of_states,
                number_of_states) * 1j
        return cls(random_matrix)

    @staticmethod
    def create_random_hermitian(cls, number_of_states: int) -> Hamiltonian:
        random_matrix = np.random.rand(
            number_of_states,
            number_of_states) \
            + np.random.rand(
                number_of_states,
                number_of_states) * 1j
        hermitian_matrix = random_matrix + np.conj(random_matrix).T
        return cls(hermitian_matrix)

    @staticmethod
    def create_diagonal(cls, energies: list):
        matrix = np.diag(energies)
        return cls(matrix)

    @classmethod
    def create_block_identity(
            cls,
            hamiltonian_cls,
            states_in_each_block,
            block_factors):
        base_matrix = np.ones((states_in_each_block, states_in_each_block))
        return cls.create_block(hamiltonian_cls, base_matrix, block_factors)

    @staticmethod
    def create_block(cls, base_matrix, block_factors):
        matrix_parts = [[block_value * base_matrix
                         for block_value in r]
                        for r in block_factors]
        return cls(np.block(matrix_parts))
