import unittest
from SystemState import SystemState


class TestHamiltonian(unittest.TestCase):

    def test_timestep(self):
        initial_state = []

    def test_permutations_for_2_1(self):

        generated_permutations = SystemState.generate_permutations(
            2,
            1)
        self.assertEquals([[0, 1], [1, 0]], sorted(generated_permutations))

    def test_permutations_for_4_2(self):

        generated_permutations = SystemState.generate_permutations(
            4,
            2)
        self.assertEquals([[0, 0, 1, 1], [0, 1, 0, 1],
                           [0, 1, 1, 0], [1, 0, 0, 1],
                           [1, 0, 1, 0], [1, 1, 0, 0]],
                          sorted(generated_permutations))

    def test_permutations_for_1_2(self):

        generated_permutations = SystemState.generate_permutations(
            1,
            2)
        self.assertEquals([], sorted(generated_permutations))


if __name__ == '__main__':
    unittest.main()
