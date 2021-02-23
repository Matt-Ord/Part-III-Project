import unittest
from ElectronSystem import ElectronSystem, ElectronSystemUtil
import numpy as np

from Hamiltonian import Hamiltonian, HamiltonianUtil


class TestElectronSystem(unittest.TestCase):

    def test_system_is_normalised(self):
        system = ElectronSystemUtil.create(
            ElectronSystem,
            [1, 0, 0, 0],
            0
        )
        self.assertEqual(1, np.linalg.norm(system.system_vector))

    def test_electron_density_one_way_hamiltonian(self):
        system = ElectronSystemUtil.create(
            ElectronSystem,
            [1, 0, 0, 0],
            0
        )

        hamiltonian: Hamiltonian = ElectronSystemUtil\
            .given(system)\
            .create_block_identity(
                Hamiltonian,
                block_factors=[[0, 1], [0, 0]],
            )

        a = ElectronSystemUtil\
            .given(system)\
            .create_kinetic(Hamiltonian, [1, 2, 3, 4], [10, 10.1])

        print(hamiltonian)
        print(a)

        print('before', system.system_vector)

        hamiltonian += a

        evolved_system = system.evolve_system(
            hamiltonian,
            40000
        )
        print(evolved_system._get_state_probabilities())
        print('after', np.abs(evolved_system.system_vector))
        print(evolved_system.get_electron_density_for_each_hydrogen())
        print()

    def test_get_electron_density_for_each_hydrogen(self):
        hydrogen_state = np.random.choice([0, 1])
        electron_state = [1, 1, 1, 0, 0]
        system = ElectronSystemUtil.create(
            ElectronSystem, electron_state, hydrogen_state)

        actual_density = system.get_electron_density_for_each_hydrogen()

        self.assertCountEqual(
            list(actual_density[hydrogen_state]),
            electron_state)
        self.assertCountEqual(
            list(actual_density[int(not hydrogen_state)]),
            [0 for _ in electron_state])

    def test_get_probability_for_each_hydrogen(self):
        hydrogen_state = np.random.choice([0, 1])
        electron_state = [1, 1, 1, 0, 0]
        system = ElectronSystemUtil.create(
            ElectronSystem, electron_state, hydrogen_state)

        actual_probabliliites = system.get_probability_for_each_hydrogen()
        self.assertCountEqual(
            list(actual_probabliliites[hydrogen_state]),
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        self.assertCountEqual(
            list(actual_probabliliites[int(not hydrogen_state)]),
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_has_at_most_one_hop(self):
        initial_state = np.array([0, 0, 1, 1])
        single_hop_state = np.array([0, 1, 0, 1])
        double_hop_state = np.array([1, 1, 0, 0])
        self.assertTrue(ElectronSystemUtil._has_at_most_one_hop(
            initial_state, initial_state))
        self.assertTrue(ElectronSystemUtil._has_at_most_one_hop(
            initial_state, single_hop_state))
        self.assertFalse(ElectronSystemUtil._has_at_most_one_hop(
            initial_state, double_hop_state))

    def test_calculate_q_dependant_single_hop_strength(self):
        actual_no_hop = ElectronSystemUtil\
            ._calculate_q_dependant_single_hop_strength(
                np.array([1, 0]),
                np.array([1, 0]),
                np.array([1, 2]),
                lambda x: x
            )
        expected_no_hop = 0
        self.assertEqual(actual_no_hop, expected_no_hop)

        actual_one_hop = ElectronSystemUtil\
            ._calculate_q_dependant_single_hop_strength(
                np.array([1, 0]),
                np.array([0, 1]),
                np.array([1, 2]),
                lambda x: x
            )
        expected_one_hop = 1
        self.assertEqual(actual_one_hop, expected_one_hop)

    def test_generate_diagonal_interaction_base_matrix(self):
        hydrogen_state = 0
        electron_state = [1, 0, 1, 0, 1, 1]
        system = ElectronSystemUtil.create(
            ElectronSystem, electron_state, hydrogen_state)

        def strength_function(x, y) -> bool: return (x == y).all()
        actual_base_matrix = ElectronSystemUtil\
            .given(system)\
            ._generate_base_matrix(strength_function)

        expected_base_matrix = np.eye(system.get_number_of_electron_states())
        self.assertCountEqual(expected_base_matrix.tolist(),
                              actual_base_matrix.tolist())

    def test_generate_single_hop_base_matrix(self):

        hydrogen_state = 0
        electron_state = [1, 1, 0, 0]
        system = ElectronSystemUtil.create(
            ElectronSystem, electron_state, hydrogen_state)

        actual_base_matrix = ElectronSystemUtil.given(
            system)._generate_single_hop_constant_base_matrix()
        expected_base_matrix = [[1, 1, 1, 1, 1, 0],
                                [1, 1, 1, 1, 0, 1],
                                [1, 1, 1, 0, 1, 1],
                                [1, 1, 0, 1, 1, 1],
                                [1, 0, 1, 1, 1, 1],
                                [0, 1, 1, 1, 1, 1]]
        self.assertCountEqual(expected_base_matrix,
                              actual_base_matrix.tolist())


if __name__ == '__main__':
    unittest.main()
