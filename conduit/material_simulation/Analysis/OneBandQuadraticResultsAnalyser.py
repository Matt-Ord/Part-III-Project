from abc import abstractmethod
from typing import Type
from material_simulation.analysis.OneBandResultsAnalyser import OneBandResultsAnalyser
from properties.MaterialProperties import NICKEL_MATERIAL_PROPERTIES
from material_simulation.MultiBandMaterialSimulator import (
    MultiBandMaterialSimulatorUtil,
)
import numpy as np


class OneBandQuadraticResultsAnalyser(OneBandResultsAnalyser):
    # @property
    # def energy_difference(self):
    #     return 10 * (
    #         self.material_properties.hydrogen_energies[1]
    #         - self.material_properties.hydrogen_energies[0]
    #     )

    pass


# Dont change either factor
class OneBandQuadraticResultsAnalyserType1(OneBandQuadraticResultsAnalyser):
    def different_occupancy_rate(
        self, initial_occupation_fraction, final_occupation_fraction
    ):
        return (
            self.amplitude
            * initial_occupation_fraction
            * (1 - final_occupation_fraction)
        )


def analyse_results(
    analyser_type: Type[OneBandQuadraticResultsAnalyserType1], data, temperature
):
    analyser = analyser_type(
        **data,
        temperature=temperature,
        material_properties=NICKEL_MATERIAL_PROPERTIES,
        simulation_energy_bandwidth=MultiBandMaterialSimulatorUtil.calculate_bandwidth(
            target_frequency=1 * 10 ** (9)
        ),
    )
    # analyser.plot_adjusted_occupation_against_occupation()
    analyser.plot_adjusted_log_rate_against_occupation()
    # analyser.plot_adjusted_rate_against_occupation()
    analyser.plot_adjusted_rate_against_energy()
    print("measured", analyser.calculate_measured_total_rate())
    print("adjusted", analyser.calculate_adjusted_total_rate())


def calculate_adjusted_rate_against_temperature(
    analyser_type: Type[OneBandQuadraticResultsAnalyserType1], curve_data, temperatures
):
    simulation_energy_bandwidth = MultiBandMaterialSimulatorUtil.calculate_bandwidth(
        target_frequency=1 * 10 ** (9)
    )
    analysers = [
        analyser_type(
            **curve_data,
            temperature=temperature,
            material_properties=NICKEL_MATERIAL_PROPERTIES,
            simulation_energy_bandwidth=simulation_energy_bandwidth,
        )
        for temperature in temperatures
    ]
    adjusted_rates = [
        analyser.calculate_adjusted_total_rate() for analyser in analysers
    ]
    print(adjusted_rates)


data_150k = {
    "manual_data": {
        "amplitude": 7914,
        "offset": 0,
    },
    "half_filled_data": {
        "amplitude": 3706,
        "offset": 0,
    },
    "auto_no_diagonal_data": {
        "amplitude": 13755,
        "offset": 0,
    },
}

if __name__ == "__main__":
    analyse_results(
        OneBandQuadraticResultsAnalyserType1,
        data_150k["auto_no_diagonal_data"],
        temperature=150,
    )
    # analyse_results(
    #     OneBandQuadraticResultsAnalyserType1,
    #     data_150k["half_filled_data"],
    #     temperature=150,
    # )
    # calculate_adjusted_rate_against_temperature(
    #     OneBandCoshResultsAnalyserType9,
    #     data_150k["manual_data"],
    #     temperatures=np.linspace(350, 50, 30),
    # )
    # calculate_adjusted_rate_against_temperature(
    #     OneBandQuadraticResultsAnalyserType1,
    #     data_150k["manual_data"],
    #     temperatures=np.linspace(40, 0, 5),
    # )
