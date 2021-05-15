import unittest
from RotatingWaveLindbladSolver import RotatingWaveLindbladSolver
import numpy as np


class TestHamiltonian(unittest.TestCase):
    def test_save_no_result(self):
        solver = RotatingWaveLindbladSolver(
            times=np.linspace(0, 10 ** (-9), 10), temperature=150
        )
        solver.save_to_file("test2.npz")

        solver2 = RotatingWaveLindbladSolver.load_from_file("test2.npz")

        self.assertEqual(solver.temperature, solver2.temperature)
        self.assertTrue(np.all(solver.times == solver2.times))
        self.assertEqual(solver._soln, None)
        self.assertEqual(solver2._soln, None)

    def test_save_result(self):
        solver = RotatingWaveLindbladSolver(
            times=np.linspace(0, 10 ** (-9), 10), temperature=150
        )
        solver._solve_lindbald_equation()
        solver.save_to_file("test.npz")

        solver2 = RotatingWaveLindbladSolver.load_from_file("test.npz")

        self.assertNotEqual(solver._soln, None)
        self.assertNotEqual(solver2._soln, None)
        self.assertTrue(np.all(solver.times == solver2.times))
        self.assertEqual(solver.temperature, solver2.temperature)
        self.assertTrue(np.array_equal(solver._soln["y"], solver2._soln["y"]))


if __name__ == "__main__":
    unittest.main()
