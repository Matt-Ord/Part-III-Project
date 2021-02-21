import unittest
from ElectronSimulator import ElectronSimulator


class TestHamiltonian(unittest.TestCase):

    def test_get_electron_densities(self):
        expected = [22, 28]
        actual = ElectronSimulator.get_electron_densities(
            [1, 2, 3], [[1, 2], [3, 4], [5, 6]])
        self.assertCountEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
