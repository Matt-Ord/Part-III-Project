import unittest
from material_simulation.analysis.OneBandCoshResultsAnalyser import (
    OneBandCoshResultsAnalyserType3,
)
from material_simulation.MultiBandMaterialSimulator import (
    MultiBandMaterialSimulatorUtil,
)
from properties.MaterialProperties import NICKEL_MATERIAL_PROPERTIES
import numpy as np


class TestOneBandResultAnalyser(unittest.TestCase):
    def test_get_lower_band_occupation(self):
        analyser = OneBandCoshResultsAnalyserType3(
            12,
            13,
            temperature=200,
            material_properties=NICKEL_MATERIAL_PROPERTIES,
            simulation_energy_bandwidth=MultiBandMaterialSimulatorUtil.calculate_bandwidth(
                target_frequency=1 * 10 ** (9)
            ),
        )
        occupation = np.random.rand()
        actual_occupation = analyser.calculate_occupation_of_lower_band(occupation)
        expected_occupation = analyser.calculate_occupation(
            analyser.calculate_energy(occupation) - analyser.energy_difference
        )
        self.assertAlmostEqual(expected_occupation, actual_occupation)


if __name__ == "__main__":
    unittest.main()
