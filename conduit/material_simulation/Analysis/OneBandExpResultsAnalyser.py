from typing import Type
import numpy as np

from material_simulation.Analysis.OneBandResultsAnalyser import OneBandResultsAnalyser
from properties.MaterialProperties import NICKEL_MATERIAL_PROPERTIES
from material_simulation.MultiBandMaterialSimulator import (
    MultiBandMaterialSimulatorUtil,
)


class OneBandExpResultsAnalyser(OneBandResultsAnalyser):
    pass


# Looks at the potential to scale the exponent
# Ignoring the N(N-1) factor
class OneBandExponentialResultsAnalyser(OneBandExpResultsAnalyser):
    def different_occupancy_rate(
        self, initial_occupation_fraction, final_occupation_fraction
    ):
        return (
            4
            * initial_occupation_fraction
            * (1 - initial_occupation_fraction)
            * np.exp(self.offset)
            * np.exp(
                -0.5
                * self.amplitude
                * np.abs(final_occupation_fraction + initial_occupation_fraction - 1)
            )
        )


class OneBandUncorrectedResultsAnalyser(OneBandExpResultsAnalyser):
    def different_occupancy_rate(
        self, initial_occupation_fraction, final_occupation_fraction
    ):
        return (
            4
            * initial_occupation_fraction
            * (1 - initial_occupation_fraction)
            * np.exp(self.offset)
            * np.exp(
                -0.5
                * self.amplitude
                * np.abs(initial_occupation_fraction + initial_occupation_fraction - 1)
            )
        )


# Attempts to correct the rate using the
# N(N-1) prefactor only
class OneBandLinearResultsAnalyser(OneBandExpResultsAnalyser):
    def different_occupancy_rate(
        self, initial_occupation_fraction, final_occupation_fraction
    ):
        return (
            4
            * initial_occupation_fraction
            * (1 - final_occupation_fraction)
            * np.exp(self.offset)
            * np.exp(
                -0.5
                * self.amplitude
                * np.abs(initial_occupation_fraction + initial_occupation_fraction - 1)
            )
        )


class OneBandCombinedResultsAnalyser(OneBandExponentialResultsAnalyser):
    def different_occupancy_rate(
        self, initial_occupation_fraction, final_occupation_fraction
    ):
        return (
            4
            * initial_occupation_fraction
            * (1 - final_occupation_fraction)
            * np.exp(self.offset)
            * np.exp(
                -0.5
                * self.amplitude
                * np.abs(final_occupation_fraction + initial_occupation_fraction - 1)
            )
        )


def analyse_results(analyser_type: Type[OneBandExpResultsAnalyser]):
    analyser = analyser_type(
        amplitude=12.56,
        offset=13.25,
        temperature=150,
        material_properties=NICKEL_MATERIAL_PROPERTIES,
        simulation_energy_bandwidth=MultiBandMaterialSimulatorUtil.calculate_bandwidth(
            target_frequency=1 * 10 ** (9)
        ),
    )
    # analyser.plot_adjusted_occupation_against_occupation()
    analyser.plot_adjusted_log_rate_against_occupation()
    analyser.plot_adjusted_rate_against_occupation()
    analyser.plot_adjusted_rate_against_energy()
    # print(analyser.calculate_measured_total_rate())


if __name__ == "__main__":
    analyse_results(OneBandExponentialResultsAnalyser)
