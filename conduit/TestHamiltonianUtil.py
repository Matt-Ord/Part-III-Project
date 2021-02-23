import unittest
from Hamiltonian import HamiltonianUtil
import numpy as np


class TestHamiltonianUtil(unittest.TestCase):

    def test_create_block_hamiltonian_size_1(self):
        matrix = np.random.rand(5, 5)
        a = HamiltonianUtil.create_block_identity(
            lambda x: x, 1, matrix)
        self.assertTrue(np.allclose(a, matrix))

    def test_create_block_hamiltonian(self):
        a = HamiltonianUtil.create_block_identity(
            lambda x: x, 2, [[1, 2], [3, 4]])
        self.assertTrue(np.allclose(
            a, [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]))


if __name__ == '__main__':
    unittest.main()
