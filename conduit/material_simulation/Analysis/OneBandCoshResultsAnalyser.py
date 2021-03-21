from abc import abstractmethod
from typing import Type
from material_simulation.Analysis.OneBandResultsAnalyser import OneBandResultsAnalyser
from properties.MaterialProperties import NICKEL_MATERIAL_PROPERTIES
from material_simulation.MultiBandMaterialSimulator import (
    MultiBandMaterialSimulatorUtil,
)
import numpy as np


class OneBandCoshResultsAnalyser(OneBandResultsAnalyser):
    @property
    def energy_difference(self):
        return 10 * (
            self.material_properties.hydrogen_energies[1]
            - self.material_properties.hydrogen_energies[0]
        )

    pass


# Dont change either factor
class OneBandCoshResultsAnalyserType1(OneBandCoshResultsAnalyser):
    def different_occupancy_rate(
        self, initial_occupation_fraction, final_occupation_fraction
    ):
        return (
            4
            * initial_occupation_fraction
            * (1 - initial_occupation_fraction)
            * np.exp(self.offset)
            / np.cosh(0.5 * self.amplitude * (2 * initial_occupation_fraction - 1))
        )


# Only change N(N-1) factor
class OneBandCoshResultsAnalyserType2(OneBandCoshResultsAnalyser):
    def different_occupancy_rate(
        self, initial_occupation_fraction, final_occupation_fraction
    ):
        return (
            4
            * initial_occupation_fraction
            * (1 - final_occupation_fraction)
            * np.exp(self.offset)
            / np.cosh(0.5 * self.amplitude * (2 * initial_occupation_fraction - 1))
        )


# Only change exponential factor
class OneBandCoshResultsAnalyserType3(OneBandCoshResultsAnalyser):
    def different_occupancy_rate(
        self, initial_occupation_fraction, final_occupation_fraction
    ):
        return (
            4
            * initial_occupation_fraction
            * (1 - initial_occupation_fraction)
            * np.exp(self.offset)
            / np.cosh(
                0.5
                * self.amplitude
                * (final_occupation_fraction + initial_occupation_fraction - 1)
            )
        )


# Change both factors
class OneBandCoshResultsAnalyserType4(OneBandCoshResultsAnalyser):
    def different_occupancy_rate(
        self, initial_occupation_fraction, final_occupation_fraction
    ):
        return (
            4
            * initial_occupation_fraction
            * (1 - final_occupation_fraction)
            * np.exp(self.offset)
            / np.cosh(
                0.5
                * self.amplitude
                * (final_occupation_fraction + initial_occupation_fraction - 1)
            )
        )


def analyse_results(analyser_type: Type[OneBandCoshResultsAnalyser], data):
    analyser = analyser_type(
        **data,
        temperature=200,
        material_properties=NICKEL_MATERIAL_PROPERTIES,
        simulation_energy_bandwidth=MultiBandMaterialSimulatorUtil.calculate_bandwidth(
            target_frequency=1 * 10 ** (9)
        ),
    )
    # analyser.plot_adjusted_occupation_against_occupation()
    # analyser.plot_adjusted_log_rate_against_occupation()
    # analyser.plot_adjusted_rate_against_occupation()
    analyser.plot_adjusted_rate_against_energy()
    print("measured", analyser.calculate_measured_total_rate())
    print("adjusted", analyser.calculate_adjusted_total_rate())


data = {
    "auto_data": {
        "amplitude": -13.58439877,
        "offset": 12.93159321,
    },
    "auto_data_log_fit": {
        "amplitude": 13.47583827,
        "offset": 12.80287851,
    },
    "manual_data": {
        "amplitude": -12.46886531,
        "offset": 12.2826424,
    },
    "manual_data_log_fit": {
        "amplitude": 11.76861189,
        "offset": 12.15316137,
    },
}
if __name__ == "__main__":
    # analyse_results(OneBandCoshResultsAnalyserType3, data["auto_data"])
    analyse_results(OneBandCoshResultsAnalyserType4, data["manual_data_log_fit"])
