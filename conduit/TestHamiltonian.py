import unittest
from Hamiltonian import Hamiltonian, HamiltonianUtil
import numpy as np


class TestHamiltonian(unittest.TestCase):

    def test_addition(self):
        h1 = HamiltonianUtil.create_random(Hamiltonian, 5)
        h2 = HamiltonianUtil.create_random(Hamiltonian, 5)
        h3 = h1 + h2
        self.assertTrue(np.allclose(
            h3._matrix_representation,
            h1._matrix_representation + h2._matrix_representation))

    def test_decomposition(self):
        a = Hamiltonian(np.array([[1, 0], [0, 2]]))
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
                    eigenvectors[:, vec]),
                expected))

    def test_decomposition_maintains_norm_if_hermitian(self):

        hamiltonian = HamiltonianUtil.create_random_hermitian(
            Hamiltonian,
            10,
        )
        vector = np.random.rand(10)

        decomposition = hamiltonian.get_eigen_decomposition_of_vector(
            vector)
        self.assertAlmostEqual(
            np.linalg.norm(decomposition),
            np.linalg.norm(vector)
        )

    def test_decomposition_of_product_state(self):
        h = np.random.rand(5, 5)
        hamiltonian = Hamiltonian(h)
        _, eigenvectors = np.linalg.eig(h)
        # To get a list of eigenvectors we must transpose!
        eigenvectors = eigenvectors.T
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

    def test_get_decomposition_after_time_with_zero_timestep(self):
        hamiltonian = HamiltonianUtil.create_random(Hamiltonian, 5)
        initial_decomposition = np.random.rand(5)
        final_decomposition = hamiltonian.get_decomposition_after_time(
            initial_decomposition, 0)
        self.assertTrue(np.allclose(
            initial_decomposition,
            final_decomposition), msg=f'initial decomposition \
                                {initial_decomposition} != \
                                final decomposition {final_decomposition}')

    def test_no_timestep_evolution_returns_initial_state(self):
        hamiltonian = HamiltonianUtil.create_random(Hamiltonian, 5)
        initial_vector = np.random.rand(5)
        final_vector = hamiltonian.evolve_system_vector(initial_vector, 0)

        self.assertTrue(np.allclose(
            initial_vector,
            final_vector), msg=f'initial vector {initial_vector} != \
                                final vector {final_vector}')

    def test_timestep_evolution_retains_norm_if_hermitian(self):
        hamiltonian = HamiltonianUtil.create_random_hermitian(Hamiltonian, 5)
        initial_vector = np.random.rand(5)
        final_vector = hamiltonian.evolve_system_vector(
            initial_vector, 1000 * np.random.random())

        self.assertAlmostEqual(
            np.linalg.norm(initial_vector),
            np.linalg.norm(final_vector))


if __name__ == '__main__':
    unittest.main()
