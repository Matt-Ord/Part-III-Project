import unittest
from Hamiltonian import Hamiltonian
import numpy as np


class TestHamiltonian(unittest.TestCase):

    def test_addition(self):
        h1 = Hamiltonian.create_random_hamiltonian(5)
        h2 = Hamiltonian.create_random_hamiltonian(5)
        h3 = h1 + h2
        self.assertCountEqual(
            h3._matrix_representation,
            h1._matrix_representation + h2._matrix_representation)

    def test_decomposition(self):
        a = Hamiltonian([[1, 0], [0, 2]])
        b = a.get_eigen_decomposition_of_vector(np.array([1, 1]))
        self.assertCountEqual(b, [1., 1.])

    def test_decomposition_of_eigenstates(self):
        h = np.random.rand(5, 5)
        hamiltonian = Hamiltonian(h)
        _, eigenvectors = np.linalg.eig(h)

        for vec in range(len(eigenvectors)):
            expected = [1 if (x == vec) else 0
                        for x in range(len(eigenvectors))]
            self.assertTrue(np.allclose(
                hamiltonian.get_eigen_decomposition_of_vector(
                    eigenvectors[vec]),
                expected))

    def test_decomposition_of_product_state(self):
        h = np.random.rand(5, 5)
        hamiltonian = Hamiltonian(h)
        _, eigenvectors = np.linalg.eig(h)
        true_decompositon = np.random.rand(len(eigenvectors))

        vector = np.zeros_like(eigenvectors[0])
        for i in range(len(eigenvectors)):
            vector += true_decompositon[i] * eigenvectors[i]

        calculated_decomposition = \
            hamiltonian.get_eigen_decomposition_of_vector(vector)

        self.assertTrue(np.allclose(
            calculated_decomposition,
            true_decompositon))

    def test_recomposition_gives_initial_vector(self):
        hamiltonian = Hamiltonian(np.random.rand(5, 5))
        vector = np.random.rand(5)
        # hamiltonian = Hamiltonian([[1, 1], [1, 1]])
        # vector = np.array([1., 2])
        self.assertTrue(np.allclose(
            vector,
            hamiltonian.get_vector_of_eigen_decomposition(
                hamiltonian.get_eigen_decomposition_of_vector(
                    vector))))

    def test_no_timestep_evolution_returns_initial_state(self):
        hamiltonian = Hamiltonian(np.random.rand(5, 5))
        initial_vector = np.random.rand(5)
        final_vector = hamiltonian.evolve_state(initial_vector, 0)

        self.assertTrue(np.allclose(
            initial_vector,
            final_vector))


if __name__ == '__main__':
    unittest.main()
