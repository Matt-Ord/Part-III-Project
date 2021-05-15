from abc import abstractmethod
from typing import Type
from material_simulation.analysis.OneBandResultsAnalyser import (
    OneBandResultsAnalyser,
    OneBandResultsAnalyserUtil,
)
from properties.MaterialProperties import NICKEL_MATERIAL_PROPERTIES
from material_simulation.MultiBandMaterialSimulator import (
    MultiBandMaterialSimulatorUtil,
)
import numpy as np
import matplotlib.pyplot as plt


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
            4
            * self.amplitude
            * initial_occupation_fraction
            * (1 - initial_occupation_fraction)
        )


# Dont change final occupation fraction
class OneBandQuadraticResultsAnalyserType2(OneBandQuadraticResultsAnalyser):
    def different_occupancy_rate(
        self, initial_occupation_fraction, final_occupation_fraction
    ):
        return (
            4
            * self.amplitude
            * initial_occupation_fraction
            * (1 - final_occupation_fraction)
        )


# Change both occupation fraction
class OneBandQuadraticResultsAnalyserType3(OneBandQuadraticResultsAnalyser):
    def different_occupancy_rate(
        self, initial_occupation_fraction, final_occupation_fraction
    ):
        return (
            4
            * self.amplitude
            * np.sqrt(initial_occupation_fraction * (1 - final_occupation_fraction))
            * np.sqrt(final_occupation_fraction * (1 - initial_occupation_fraction))
        )


def analyse_results(
    analyser_type: Type[OneBandQuadraticResultsAnalyser], data, temperature
):
    analyser = analyser_type(
        **data,
        temperature=temperature,
        material_properties=NICKEL_MATERIAL_PROPERTIES,
        simulation_energy_bandwidth=MultiBandMaterialSimulatorUtil.calculate_bandwidth(
            target_frequency=1 * 10 ** (9)
        ),
    )  # type: ignore
    # analyser.plot_adjusted_occupation_against_occupation()
    # analyser.plot_adjusted_log_rate_against_occupation()
    # analyser.plot_adjusted_rate_against_occupation()
    # analyser.plot_adjusted_rate_against_energy()
    # print("measured", analyser.calculate_measured_total_rate())
    # print("adjusted", analyser.calculate_adjusted_total_rate())
    print("measured continuous", analyser.integrate_measured_total_rate())
    print("adjusted continuous", analyser.integrate_adjusted_total_rate())


def plot_all_adjusted_rate_against_energy(data):
    analyser_kwargs = {
        "temperature": 150,
        "material_properties": NICKEL_MATERIAL_PROPERTIES,
        "simulation_energy_bandwidth": MultiBandMaterialSimulatorUtil.calculate_bandwidth(
            target_frequency=1 * 10 ** (9)
        ),
    }
    fig, ax = plt.subplots(1)
    OneBandResultsAnalyserUtil.plot_adjusted_rate_against_energy(
        analysers={
            r"$N(N-1)$": OneBandQuadraticResultsAnalyserType1(
                **data, **analyser_kwargs
            ),
            r"$N(N'-1)$": OneBandQuadraticResultsAnalyserType2(
                **data, **analyser_kwargs
            ),
            r"$\sqrt{N(N-1)N'(N'-1)}$": OneBandQuadraticResultsAnalyserType3(
                **data, **analyser_kwargs
            ),
        },
        ax=ax,
        energy_range=4,
    )
    plt.show()


def calculate_adjusted_rate_against_temperature(
    analyser_type: Type[OneBandQuadraticResultsAnalyser], curve_data, temperatures
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
        )  # type:ignore
        for temperature in temperatures
    ]
    adjusted_rates = [
        analyser.integrate_adjusted_total_rate() for analyser in analysers
    ]
    print(adjusted_rates)


data_150k = {
    "manual_data": {
        "amplitude": 7914,
        "offset": 0,
    },
    "half_filled_data": {
        "amplitude": 3864,
        "offset": 0,
    },
    "two_electron_data": {
        "amplitude": 8353,
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
        data_150k["two_electron_data"],
        temperature=150,
    )
    analyse_results(
        OneBandQuadraticResultsAnalyserType2,
        data_150k["two_electron_data"],
        temperature=150,
    )
    analyse_results(
        OneBandQuadraticResultsAnalyserType3,
        data_150k["two_electron_data"],
        temperature=150,
    )
    # plot_all_adjusted_rate_against_energy(data_150k["half_filled_data"])
    # analyse_results(
    #     OneBandQuadraticResultsAnalyserType1,
    #     data_150k["half_filled_data"],
    #     temperature=150,
    # )
    # calculate_adjusted_rate_against_temperature(
    #     OneBandQuadraticResultsAnalyserType1,
    #     data_150k["half_filled_data"],
    #     temperatures=np.linspace(350, 0, 100),
    # )
    # calculate_adjusted_rate_against_temperature(
    #     OneBandQuadraticResultsAnalyserType2,
    #     data_150k["half_filled_data"],
    #     temperatures=np.linspace(350, 0, 100),
    # )
    # calculate_adjusted_rate_against_temperature(
    #     OneBandQuadraticResultsAnalyserType3,
    #     data_150k["half_filled_data"],
    #     temperatures=np.linspace(350, 0, 100),
    # )
    # calculate_adjusted_rate_against_temperature(
    #     OneBandQuadraticResultsAnalyserType1,
    #     data_150k["two_electron_data"],
    #     temperatures=np.linspace(350, 0, 100),
    # )
    # calculate_adjusted_rate_against_temperature(
    #     OneBandQuadraticResultsAnalyserType2,
    #     data_150k["two_electron_data"],
    #     temperatures=np.linspace(350, 0, 100),
    # )
    # calculate_adjusted_rate_against_temperature(
    #     OneBandQuadraticResultsAnalyserType3,
    #     data_150k["two_electron_data"],
    #     temperatures=np.linspace(350, 0, 100),
    # )
