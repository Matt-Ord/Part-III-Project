from abc import abstractmethod
from typing import Type
from material_simulation.analysis.OneBandResultsAnalyser import OneBandResultsAnalyser
from properties.MaterialProperties import NICKEL_MATERIAL_PROPERTIES
from material_simulation.MultiBandMaterialSimulator import (
    MultiBandMaterialSimulatorUtil,
)
import numpy as np


class OneBandCoshResultsAnalyser(OneBandResultsAnalyser):
    # @property
    # def energy_difference(self):
    #     return 10 * (
    #         self.material_properties.hydrogen_energies[1]
    #         - self.material_properties.hydrogen_energies[0]
    #     )

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


# Change N(N-1) factor in a more symmetric way
class OneBandCoshResultsAnalyserType5(OneBandCoshResultsAnalyser):
    def different_occupancy_rate(
        self, initial_occupation_fraction, final_occupation_fraction
    ):
        return (
            4
            * np.sqrt(
                initial_occupation_fraction
                * final_occupation_fraction
                * (1 - initial_occupation_fraction)
                * (1 - final_occupation_fraction)
            )
            * np.exp(self.offset)
            / np.cosh(
                0.5
                * self.amplitude
                * (final_occupation_fraction + initial_occupation_fraction - 1)
            )
        )


# The following mess around with the definition
# of a backwards rate, and as such I dont think
# are reasonable choices.

# Change N(N-1) factor with reverse rate being
# a tunnelling process followed by a drop in
# energy, not a drop in energy fllowed by
# tunnelling process

# Note this should have no effect on the overall
# Tunnelling rate!
class OneBandCoshResultsAnalyserType6(OneBandCoshResultsAnalyser):
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

    def adjusted_rate_curve(self, occupation_fraction) -> np.ndarray:
        lower_occupation_fraction = self.calculate_occupation_of_lower_band(
            occupation_fraction
        )
        higher_occupation_fraction = self.calculate_occupation_of_higher_band(
            occupation_fraction
        )
        return 0.5 * (
            self.different_occupancy_rate(
                occupation_fraction, lower_occupation_fraction
            )
            + self.different_occupancy_rate(
                occupation_fraction, higher_occupation_fraction
            )
        )


# Change N(N-1) factor with reverse rate being
# a tunnelling process followed by a drop in
# energy, not a drop in energy fllowed by
# tunnelling process using a symmetric prefactor

# Not a physical curve shape
class OneBandCoshResultsAnalyserType7(OneBandCoshResultsAnalyser):
    def different_occupancy_rate(
        self, initial_occupation_fraction, final_occupation_fraction
    ):
        return (
            4
            * np.sqrt(
                initial_occupation_fraction
                * final_occupation_fraction
                * (1 - initial_occupation_fraction)
                * (1 - final_occupation_fraction)
            )
            * np.exp(self.offset)
            / np.cosh(
                0.5
                * self.amplitude
                * (final_occupation_fraction + initial_occupation_fraction - 1)
            )
        )

    def adjusted_rate_curve(self, occupation_fraction) -> np.ndarray:
        lower_occupation_fraction = self.calculate_occupation_of_lower_band(
            occupation_fraction
        )
        higher_occupation_fraction = self.calculate_occupation_of_higher_band(
            occupation_fraction
        )
        return 0.5 * (
            self.different_occupancy_rate(
                occupation_fraction, lower_occupation_fraction
            )
            + self.different_occupancy_rate(
                occupation_fraction, higher_occupation_fraction
            )
        )


# Change N(N-1) factor with reverse rate being
# a tunnelling process followed by a drop in
# energy, not a drop in energy fllowed by
# tunnelling process using a symmetric prefactor
# but don't modify exponential term

# Most premutations on
# these results can be ruled out simply by the
# shape of the curve!
class OneBandCoshResultsAnalyserType8(OneBandCoshResultsAnalyser):
    def different_occupancy_rate(
        self, initial_occupation_fraction, final_occupation_fraction
    ):
        return (
            4
            * np.sqrt(
                initial_occupation_fraction
                * final_occupation_fraction
                * (1 - initial_occupation_fraction)
                * (1 - final_occupation_fraction)
            )
            * np.exp(self.offset)
            / np.cosh(0.5 * self.amplitude * (2 * initial_occupation_fraction - 1))
        )

    def adjusted_rate_curve(self, occupation_fraction) -> np.ndarray:
        lower_occupation_fraction = self.calculate_occupation_of_lower_band(
            occupation_fraction
        )
        higher_occupation_fraction = self.calculate_occupation_of_higher_band(
            occupation_fraction
        )
        return 0.5 * (
            self.different_occupancy_rate(
                occupation_fraction, lower_occupation_fraction
            )
            + self.different_occupancy_rate(
                occupation_fraction, higher_occupation_fraction
            )
        )


# Change N(N-1) factor in a more symmetric way
# without the square root
# since degenerate rate does not have the
# final transition
class OneBandCoshResultsAnalyserType9(OneBandCoshResultsAnalyser):
    def different_occupancy_rate(
        self, initial_occupation_fraction, final_occupation_fraction
    ):
        return (
            16
            * (
                initial_occupation_fraction
                * final_occupation_fraction
                * (1 - initial_occupation_fraction)
                * (1 - final_occupation_fraction)
            )
            * np.exp(self.offset)
            / np.cosh(
                0.5
                * self.amplitude
                * (final_occupation_fraction + initial_occupation_fraction - 1)
            )
        )


def analyse_results(analyser_type: Type[OneBandCoshResultsAnalyser], data, temperature):
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
    analyser.plot_adjusted_rate_against_energy()
    print("measured", analyser.calculate_measured_total_rate())
    print("adjusted", analyser.calculate_adjusted_total_rate())


def calculate_adjusted_rate_against_temperature(
    analyser_type: Type[OneBandCoshResultsAnalyser], curve_data, temperatures
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

data_120k = {"auto_data": {"amplitude": -14.3556328, "offset": 13.01287323}}
if __name__ == "__main__":
    analyse_results(
        OneBandCoshResultsAnalyserType5, data_150k["auto_data"], temperature=150
    )
    # calculate_adjusted_rate_against_temperature(
    #     OneBandCoshResultsAnalyserType9,
    #     data_150k["manual_data"],
    #     temperatures=np.linspace(350, 50, 30),
    # )
    calculate_adjusted_rate_against_temperature(
        OneBandCoshResultsAnalyserType9,
        data_150k["manual_data"],
        temperatures=np.linspace(40, 0, 5),
    )
    calculate_adjusted_rate_against_temperature(
        OneBandCoshResultsAnalyserType8,
        data_150k["manual_data"],
        temperatures=np.linspace(40, 0, 5),
    )
    calculate_adjusted_rate_against_temperature(
        OneBandCoshResultsAnalyserType7,
        data_150k["manual_data"],
        temperatures=np.linspace(40, 0, 5),
    )
    calculate_adjusted_rate_against_temperature(
        OneBandCoshResultsAnalyserType6,
        data_150k["manual_data"],
        temperatures=np.linspace(40, 0, 5),
    )
    calculate_adjusted_rate_against_temperature(
        OneBandCoshResultsAnalyserType5,
        data_150k["manual_data"],
        temperatures=np.linspace(40, 0, 5),
    )
    calculate_adjusted_rate_against_temperature(
        OneBandCoshResultsAnalyserType4,
        data_150k["manual_data"],
        temperatures=np.linspace(40, 0, 5),
    )
    analyse_results(OneBandCoshResultsAnalyserType9, data_150k["manual_data"], 50)
