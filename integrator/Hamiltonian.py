from __future__ import annotations
import numpy as np
from functools import cached_property

from numpy.linalg import eig


class Hamiltonian():
    _eigenvalues = None
    _eigenvectors = None

    def __init__(self, matrix_representation: np.ndarray) -> None:
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
            np.linalg.eig(self._matrix_representation)

    def _get_number_of_states(self):
        return self._matrix_representation.shape[0]

    def _is_valid_state_vector(self, vector):
        return vector.shape == (self._get_number_of_states(),)

    def evolve_state(self, intial_state_vector, time):
        initial_state_decomposition = \
            self.get_eigen_decomposition_of_vector(intial_state_vector)
        eigenvector_phase_shift = np.exp(1j * self._eigenvalues * time)
        final_state_decompositon = np.multiply(
            eigenvector_phase_shift, initial_state_decomposition)
        final_state = \
            self.get_vector_of_eigen_decomposition(final_state_decompositon)
        return final_state / np.linalg.norm(final_state)

    def get_eigen_decomposition_of_vector(self, vector):
        if not self._is_valid_state_vector(vector):
            raise Exception(
                f'state vector shape is wrong: actual {vector.shape}, \
                expected({self._get_number_of_states()},)')
        return np.linalg.solve(self.eigenvectors.T, vector)
        return np.linalg.inv(self.eigenvectors.T).dot(vector)

    def get_vector_of_eigen_decomposition(self, decomposition):
        return self.eigenvectors.T.dot(decomposition)

    @classmethod
    def create_random_hamiltonian(cls, number_of_states: int) -> Hamiltonian:
        random_matrix = np.random.rand(
            number_of_states,
            number_of_states) \
            + np.random.rand(
                number_of_states,
                number_of_states) * 1j
        return cls(random_matrix)

    @classmethod
    def create_diagonal_hamiltonian(cls, energies: list) -> Hamiltonian:
        matrix = np.diag(energies)
        return cls(matrix)

    def __add__(self, other: Hamiltonian) -> Hamiltonian:
        return type(self)(self._matrix_representation
                          + other._matrix_representation)

    def __str__(self) -> str:
        return self._matrix_representation.__str__()
