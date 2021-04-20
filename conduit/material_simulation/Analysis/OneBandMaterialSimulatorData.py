# Utility class used to collect data from the one band material
# simulator
import pickle
from typing import Any, Dict, List
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from simulation.ElectronSimulationPlotter import (
    ElectronSimulationPlotter,
    plot_average_densities_example,
)
import scipy.signal
from collections import defaultdict

from material_simulation.MultiBandMaterialSimulator import (
    MultiBandNickelMaterialSimulatorUtil,
)
from material_simulation.OneBandMaterialSimulator import OneBandMaterialSimulator

omega_150K = {
    7: {1: 0, 2: 2810000, 3: 5500000, 4: 5500000, 5: 2810000, 6: 0},
    8: {1: 0, 2: 2459000, 3: 4865000, 4: 7250000, 5: 4865000, 6: 2459000, 7: 0},
    9: {1: 0, 2: 2189000, 3: 4365000, 6: 4365000, 7: 2189000, 8: 0},
    10: {
        1: 0,
        2: 1974000,
        3: 3940000,
        4: 5800000,
        6: 5800000,
        7: 3940000,
        8: 1974000,
        9: 0,
    },
    11: {1: 0, 2: 1795000, 3: 3592000, 8: 3592000, 9: 1795000, 10: 0},
    12: {1: 0, 2: 1649000, 3: 3290000, 9: 3290000, 10: 1649000, 11: 0},
    13: {1: 0, 2: 1520000, 3: 3040000, 10: 3040000, 11: 1520000, 12: 0},
    16: {1: 0, 15: 0},
    17: {1: 0, 16: 0},
    18: {1: 0, 17: 0},
    23: {1: 0, 22: 0},
    41: {1: 0, 40: 0},
    67: {1: 0, 66: 0},
    97: {1: 0, 96: 0},
}

omega_120K = {
    7: {1: 0, 2: 3080000, 3: 6100000, 4: 6100000, 5: 3080000, 6: 0},
    8: {1: 0, 2: 2710000, 3: 5400000, 4: 8000000, 5: 5400000, 6: 2710000, 7: 0},
    9: {1: 0, 2: 2410000, 3: 4820000, 6: 4815000, 7: 2410000, 8: 0},
    11: {1: 0, 2: 1975000, 9: 1975000, 10: 0},
}

times_150K = {
    7: {
        2: 4.5e-05,
        3: 6e-06,
        4: 6e-06,
        5: 4.5e-05,
    },
    8: {
        1: 7.5558318058332e-05,
        2: 6e-05,
        3: 1.4e-05,
        4: 5e-06,
        5: 1.5e-05,
        6: 6e-05,
        7: 8.244901420131929e-05,
    },
    9: {
        2: 6.2e-05,
        3: 1.8e-05,
        6: 1.8e-05,
        7: 6.2e-05,
    },
    10: {
        2: 9e-05,
        3: 3e-05,
        4: 8e-06,
        6: 8e-06,
        7: 3e-05,
        8: 9e-05,
    },
    11: {
        1: 0.00013096863483106044,
        2: 11e-05,
        3: 4.5e-05,
        8: 4.5e-05,
        9: 11e-05,
        10: 0.0001294631504746427,
    },
    12: {
        2: 12e-05,
        3: 5e-05,
        9: 5e-05,
        10: 12e-05,
    },
    13: {
        1: 0.00017,
        2: 0.0001271751274220063,
        3: 6e-05,
        10: 6e-05,
        11: 0.00013,
        12: 0.00017,
    },
    16: {
        1: 0.00021031815445628774,
        15: 0.0002076017911176765,
    },
    17: {
        1: 0.00021808276462576535,
        16: 0.00023341955198895103,
    },
    18: {
        1: 0.00024762301585452303,
        17: 0.00024351771937875747,
    },
    23: {
        1: 0.00033080585725698083,
        22: 0.0003368310373444469,
    },
    41: {
        1: 0.0005699156687531629,
        40: 0.000652994710412042,
    },
    67: {
        1: 0.001117159878235939,
        66: 0.0010935980152674382,
    },
    97: {
        1: 0.0016705988337883816,
        96: 0.0016620298601939588,
    },
}


# rough_150K_corrected = {
#     7: {2: 3e-04, 3: 3e-04, 4: 3e-04, 5: 3e-04},
#     8: {1: 3e-04, 2: 3e-04, 3: 3e-04, 4: 3e-04, 5: 3e-04, 6: 3e-04, 7: 3e-04},
#     9: {2: 5e-04, 3: 3e-04, 4: 3e-04, 5: 3e-04, 6: 3e-04, 7: 5e-04},
#     10: {2: 5e-04, 3: 3e-04, 4: 3e-04, 5: 3e-04, 6: 3e-04, 7: 3e-04, 8: 5e-04},
#     11: {1: 5e-04, 2: 5e-04, 3: 5e-04, 8: 5e-04, 9: 5e-04, 10: 5e-04},
#     12: {2: 6e-04, 3: 5e-04, 9: 5e-04, 10: 6e-04},
#     13: {1: 8e-04, 2: 8e-04, 3: 4e-04, 10: 4e-04, 11: 8e-04, 12: 8e-04},
#     16: {1: 10e-04, 15: 10e-04},
#     17: {1: 10e-04, 16: 10e-04},
#     18: {1: 10e-04, 17: 10e-04},
#     23: {1: 12e-04, 22: 12e-04},
#     41: {1: 25e-04, 40: 25e-04},
#     67: {1: 50e-04, 66: 50e-04},
#     97: {1: 60e-04, 96: 60e-04},
# }
rough_150K_corrected_5000_average = {
    2: {1: 0.15e-04},
    4: {2: 0.5e-4},
    5: {2: 1.25e-04, 3: 1.25e-04},
    6: {2: 2e-04, 3: 1.5e-04, 4: 1.5e-04},
}
rough_150K_corrected_100_average = {
    11: {4: 4e-04, 5: 3e-04, 6: 3e-04, 7: 4e-04},
    12: {4: 4e-04, 5: 4e-04, 6: 4e-04, 7: 4e-04, 8: 4e-04},
    13: {4: 5e-04, 5: 4e-04, 8: 4e-04, 9: 5e-04},
}
rough_150K_corrected_50_average = {
    14: {7: 6e-04},
}
# Longer time, wt smaller w smaller
omega_150k_corrected = {
    # 7: {2: 0, 3: 0, 4: 0, 5: 0},
    # 8: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
    # 9: {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0},
    # 10: {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0},
    # 11: {1: 0, 2: 0, 3: 0, 8: 0, 9: 0, 10: 0},
    # 12: {2: 0, 3: 0, 9: 0, 10: 0},
    # 13: {1: 0, 2: 0, 3: 0, 10: 0, 11: 0, 12: 0},
    # 16: {1: 0, 15: 0},
    # 17: {1: 0, 16: 0},
    # 18: {1: 0, 17: 0},
    # 23: {1: 0, 22: 0},
    # 41: {1: 0, 40: 0},
    # 67: {1: 0, 66: 0},
    # 97: {1: 0, 96: 0},
}
times_150k_corrected = {
    7: {2: 1.25e-04, 3: 1.1e-04, 4: 1.1e-04, 5: 1.25e-04},
    8: {
        1: 1.2e-04,
        2: 1.6e-04,
        3: 1.1e-04,
        4: 1.25e-04,
        5: 1e-04,
        6: 1.4e-04,
        7: 2e-04,  # REDO??
    },
    9: {2: 1.7e-04, 3: 1.6e-04, 4: 1.4e-04, 5: 1.4e-04, 6: 1.6e-04, 7: 1.7e-04},
    10: {
        2: 2e-04,
        3: 1.6e-04,
        4: 1.5e-04,
        5: 1.4e-04,
        6: 1.3e-04,
        7: 1.5e-04,
        8: 1.9e-04,
    },
    11: {1: 1.9e-04, 2: 2.1e-04, 3: 2e-04, 8: 1.9e-04, 9: 2.1e-04, 10: 3e-04},
    12: {2: 2.5e-04, 3: 2.2e-04, 9: 2.2e-04, 10: 2.5e-04},  # redo 9
    13: {1: 2.5e-04, 2: 2.7e-04, 3: 2.2e-04, 10: 2.3e-04, 11: 2.7e-04, 12: 4e-04},
    16: {1: 3.1e-04, 15: 5.5e-04},
    17: {1: 3.1e-04, 16: 5.5e-04},
    18: {1: 3.5e-04, 17: 5.5e-04},
    23: {1: 4.8e-04, 22: 8e-04},
    41: {1: 9.5e-04, 40: 15e-04},
    67: {1: 15e-04, 66: 26e-04},
    97: {1: 24e-04, 96: 35e-04},
}


class OneBandMaterialSimualtorData:
    def __init__(self, data: Dict[int, Dict[int, List]] = {}) -> None:
        self._data = data

    def _insert_data_to_dict(self, number_of_states, number_of_electrons, data):
        if number_of_states not in self._data:
            self._data[number_of_states] = {}
        if number_of_electrons not in self._data[number_of_states]:
            self._data[number_of_states][number_of_electrons] = []

        old_data = self._data[number_of_states][number_of_electrons]
        old_data.append(data)
        self._data[number_of_states][number_of_electrons] = old_data

    def plot_data(self, number_of_states, number_of_electrons, index):
        data = self._data[number_of_states][number_of_electrons][index]
        ElectronSimulationPlotter._plot_total_number_against_time(
            number_in_each_state={
                "fcc": data[1],
            },
            times=data[0],
        )
        plt.show()

    def add_data(self, simulator: OneBandMaterialSimulator, average_densities, times):
        simulator.number_of_electrons
        simulator.number_of_states_per_band
        self._insert_data_to_dict(
            simulator.number_of_states_per_band,
            simulator.number_of_electrons,
            (times, average_densities),
        )

    def filter_data(self, filter):
        new_data = {}
        for number_of_states, data_for_number_of_states in self._data.items():
            new_data[number_of_states] = {}
            for (
                number_of_electrons,
                data_for_number_of_electrons,
            ) in data_for_number_of_states.items():
                if filter(number_of_states, number_of_electrons):
                    new_data[number_of_states][
                        number_of_electrons
                    ] = data_for_number_of_electrons
        return type(self)(new_data)

    def clear_data(self, number_of_states, number_of_electrons):
        self._data[number_of_states][number_of_electrons] = []

    @staticmethod
    def _ex_occupation_curve_fit_function(time, omega, decay_time):
        return 0.5 * np.exp(-time / decay_time) * np.cos(omega * time) + 0.5

    def _fit_ex_electron_occupation_curve(
        self, times, normalised_densities, omega
    ) -> Any:
        initial_decay_time_guess = (times[-1] - times[0]) / 4

        lower_bounds = [0.1 * initial_decay_time_guess]
        upper_bounds = [4 * initial_decay_time_guess]
        return scipy.optimize.curve_fit(
            lambda t, x: self._ex_occupation_curve_fit_function(t, omega, x),
            times,
            normalised_densities,
            p0=[initial_decay_time_guess],
            bounds=[lower_bounds, upper_bounds],
        )

    def generate_decay_time_data_fit_ex(self, omega, plot_each: bool = False):
        decay_time_data = {}
        for number_of_states, data_for_number_of_states in self._data.items():
            decay_time_data[number_of_states] = {}
            for (
                number_of_electrons,
                data_for_number_of_electrons,
            ) in data_for_number_of_states.items():
                decay_time_data[number_of_states][number_of_electrons] = []
                for i, (times, densities) in enumerate(data_for_number_of_electrons):
                    normalised_densities = np.array(densities) / number_of_electrons
                    if (
                        omega.get(number_of_states, {}).get(number_of_electrons, None)
                        is not None
                    ):
                        a, b = self._fit_ex_electron_occupation_curve(
                            times,
                            normalised_densities,
                            omega[number_of_states][number_of_electrons],
                        )
                        decay_time_data[number_of_states][number_of_electrons].append(
                            (a[0], b[0][0])
                        )

        if plot_each:
            self.plot_each_decay_curve(omega, decay_time_data)
        return OneBandDecayTimeData(decay_time_data)

    @staticmethod
    def _ex_squared_occupation_curve_fit_function(time, omega, decay_time):
        return 0.97 * (
            0.5 * np.exp(-((time / decay_time) ** 2)) * np.cos(omega * time) + 0.5
        )

    def _fit_ex_squared_electron_occupation_curve(
        self, times, normalised_densities, omega
    ) -> Any:
        initial_decay_time_guess = (times[-1] - times[0]) / 4

        lower_bounds = [0.1 * initial_decay_time_guess]
        upper_bounds = [4 * initial_decay_time_guess]
        return scipy.optimize.curve_fit(
            lambda t, x: self._ex_squared_occupation_curve_fit_function(t, omega, x),
            times,
            normalised_densities,
            p0=[initial_decay_time_guess],
            bounds=[lower_bounds, upper_bounds],
        )

    @staticmethod
    def _ex_squared_decay_curve_fit_function(time, decay_time, amplitude):
        return amplitude * np.exp(-((time / decay_time) ** 2)) + 0.5

    def _fit_ex_squared_decay_curve(
        self, times, normalised_densities, fixed_amplitude=False
    ) -> Any:
        initial_decay_time_guess = (times[-1] - times[0]) / 4

        # lower_bounds = [0.1 * initial_decay_time_guess]
        # upper_bounds = [4 * initial_decay_time_guess]
        if fixed_amplitude:
            return scipy.optimize.curve_fit(
                lambda t, x: self._ex_squared_decay_curve_fit_function(t, x, 0.5),
                times,
                normalised_densities,
                p0=[initial_decay_time_guess],
            )
        return scipy.optimize.curve_fit(
            self._ex_squared_decay_curve_fit_function,
            times,
            normalised_densities,
            p0=[initial_decay_time_guess, 0.5],
            bounds=[[-np.inf, 0.3], [np.inf, 0.5]],
        )

    def generate_decay_time_data_fit_ex_squared(self, omega, plot_each: bool = False):
        decay_time_data = {}
        for number_of_states, data_for_number_of_states in self._data.items():
            decay_time_data[number_of_states] = {}
            for (
                number_of_electrons,
                data_for_number_of_electrons,
            ) in data_for_number_of_states.items():
                decay_time_data[number_of_states][number_of_electrons] = []
                for i, (times, densities) in enumerate(data_for_number_of_electrons):
                    normalised_densities = np.array(densities) / number_of_electrons
                    if (
                        omega.get(number_of_states, {}).get(number_of_electrons, None)
                        is not None
                    ):
                        if 0 == omega[number_of_states][number_of_electrons]:
                            a, b = self._fit_ex_squared_electron_occupation_curve(
                                times,
                                normalised_densities,
                                omega[number_of_states][number_of_electrons],
                            )
                        else:
                            a, b = self._fit_ex_electron_occupation_curve(
                                times,
                                normalised_densities,
                                omega[number_of_states][number_of_electrons],
                            )
                        decay_time_data[number_of_states][number_of_electrons].append(
                            (a[0], b[0][0])
                        )

        if plot_each:
            self.plot_each_decay_curve(omega, decay_time_data)
        return OneBandDecayTimeData(decay_time_data)

    def plot_each_decay_curve(self, omega: Dict[int, Dict[int, float]], times):
        for number_of_states, time_for_number_of_states in times.items():
            for (
                number_of_electrons,
                time,
            ) in time_for_number_of_states.items():
                densities_data = self._data[number_of_states][number_of_electrons]
                print(number_of_states, number_of_electrons)
                if (
                    omega.get(number_of_states, {}).get(number_of_electrons, None)
                    is None
                ):
                    continue
                if 0 == omega[number_of_states][number_of_electrons]:
                    ElectronSimulationPlotter._plot_total_number_against_time(
                        number_in_each_state={
                            "actual": np.array(densities_data[0][1])
                            / number_of_electrons,
                            "average": self._ex_squared_occupation_curve_fit_function(
                                np.array(densities_data[0][0]),
                                omega[number_of_states][number_of_electrons],
                                time[0][0],
                            ),
                        },
                        times=densities_data[0][0],
                    )
                else:
                    ElectronSimulationPlotter._plot_total_number_against_time(
                        number_in_each_state={
                            "actual": np.array(densities_data[0][1])
                            / number_of_electrons,
                            "average": self._ex_occupation_curve_fit_function(
                                np.array(densities_data[0][0]),
                                omega[number_of_states][number_of_electrons],
                                time[0][0],
                            ),
                        },
                        times=densities_data[0][0],
                    )

                plt.show()

    def plot_each_decay_curve2(self, times, offset):
        for number_of_states, time_for_number_of_states in times.items():
            for (
                number_of_electrons,
                time,
            ) in time_for_number_of_states.items():
                densities_data = self._data[number_of_states][number_of_electrons]
                print(number_of_states, number_of_electrons)

                ElectronSimulationPlotter._plot_total_number_against_time(
                    number_in_each_state={
                        "actual": np.array(densities_data[0][1]) / number_of_electrons,
                        "average": self._ex_squared_decay_curve_fit_function(
                            np.array(densities_data[0][0]),
                            time[0][0],
                            offset[number_of_states][number_of_electrons][0],
                        ),
                    },
                    times=densities_data[0][0],
                )
                plt.show()

    def generate_decay_time_data_fit_ex_squared_wavepacket(
        self, plot_each: bool = False, fixed_amplitude: bool = False
    ):
        decay_time_data = {}
        amplitude = {}
        for number_of_states, data_for_number_of_states in self._data.items():
            decay_time_data[number_of_states] = {}
            amplitude[number_of_states] = {}
            for (
                number_of_electrons,
                data_for_number_of_electrons,
            ) in data_for_number_of_states.items():
                decay_time_data[number_of_states][number_of_electrons] = []
                amplitude[number_of_states][number_of_electrons] = []
                for i, (times, densities) in enumerate(data_for_number_of_electrons):
                    normalised_densities = np.array(densities) / number_of_electrons
                    K = 10
                    upper_filtered_data = np.array(
                        [
                            max(normalised_densities[j : j + K])
                            for j in range(len(normalised_densities))
                        ]
                    )
                    # ElectronSimulationPlotter._plot_total_number_against_time(
                    #     number_in_each_state={
                    #         "actual": np.array(filtered_data).tolist(),
                    #     },
                    #     times=times,
                    # )
                    # plt.show()
                    a1, b1 = self._fit_ex_squared_decay_curve(
                        times, upper_filtered_data, fixed_amplitude
                    )

                    decay_time_data[number_of_states][number_of_electrons].append(
                        (a1[0], b1[0][0])
                    )
                    amplitude[number_of_states][number_of_electrons].append(
                        0.5 if fixed_amplitude else a1[1]
                    )
                    if number_of_electrons != 1:
                        lower_filtered_data = 1 - np.array(
                            [
                                min(normalised_densities[j : j + K])
                                for j in range(len(normalised_densities))
                            ]
                        )
                        a2, b2 = self._fit_ex_squared_decay_curve(
                            times, lower_filtered_data, fixed_amplitude
                        )

                        decay_time_data[number_of_states][number_of_electrons].append(
                            (a2[0], b2[0][0])
                        )
                        amplitude[number_of_states][number_of_electrons].append(
                            0.5 if fixed_amplitude else a2[1]
                        )

        if plot_each:
            self.plot_each_decay_curve2(decay_time_data, amplitude)
        return OneBandDecayTimeData(decay_time_data)

    def generate_decay_time_data_fit_ex_squared_manually(
        self, omega, times, plot_each: bool = False
    ):
        decay_time_data = {}
        for number_of_states, time_for_number_of_states in times.items():
            decay_time_data[number_of_states] = {}
            for (
                number_of_electrons,
                time,
            ) in time_for_number_of_states.items():
                decay_time_data[number_of_states][number_of_electrons] = [[time, 0]]

        if plot_each:
            self.plot_each_decay_curve(omega, decay_time_data)

        return OneBandDecayTimeData(decay_time_data)

    def save_to_file(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self._data, f, pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_from_file(cls, filename):
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
        except FileNotFoundError:
            data = {}
        return cls(data)


class OneBandDecayTimeData:
    def __init__(self, data: Dict[int, Dict[int, List]]) -> None:
        self._data = data

    @staticmethod
    def _quadratic_decay_rate_curve(occupation, amplitude, offset):
        return -amplitude * 4 * occupation * (1 - occupation) + offset

    @staticmethod
    def _fixed_quadratic_decay_rate_curve(occupation, amplitude):
        return amplitude * 4 * occupation * (1 - occupation)

    @staticmethod
    def _sqrt_quadratic_decay_rate_curve(occupation, amplitude, offset):
        return -amplitude * np.sqrt(occupation * (occupation - 1) + 0.25) + offset

    @staticmethod
    def _linear_decay_rate_curve(occupation, amplitude, offset):
        return -amplitude * np.abs(occupation - 0.5) + offset

    @staticmethod
    def _linear_decay_rate_with_log_curve(occupation, amplitude, offset):
        return (
            -amplitude * np.abs(occupation - 0.5)
            + offset
            + np.log(4 * occupation * (1 - occupation))
        )

    @staticmethod
    def _cosh_decay_rate_curve(occupation, amplitude, offset):
        return offset - np.log(np.cosh(amplitude * (occupation - 0.5)))

    @staticmethod
    def _cosh_decay_rate_with_log_curve(occupation, amplitude, offset):
        return (
            offset
            - np.log(np.cosh(amplitude * (occupation - 0.5)))
            + np.log(4 * occupation * (1 - occupation))
        )

    @classmethod
    def fit_rate_curve(cls, all_occupations, all_decay_times, rate_fn) -> Any:
        return scipy.optimize.curve_fit(
            f=rate_fn,
            xdata=all_occupations,
            ydata=all_decay_times,
        )

    @staticmethod
    def _decay_N4_time_curve(occupation, amplitude):
        return amplitude * (occupation * (1 - occupation)) ** 4

    @classmethod
    def _fit_N4_time_curve(cls, all_occupations, all_decay_times) -> Any:
        return scipy.optimize.curve_fit(
            f=cls._decay_N4_time_curve,
            xdata=all_occupations,
            ydata=all_decay_times,
        )

    @staticmethod
    def _decay_exponential_time_curve(occupation, amplitude, period):
        return amplitude * np.exp(-((occupation - 0.5) ** 2) * period)

    @classmethod
    def _fit_exponential_time_curve(cls, all_occupations, all_decay_times) -> Any:
        return scipy.optimize.curve_fit(
            f=cls._decay_exponential_time_curve,
            xdata=all_occupations,
            ydata=all_decay_times,
        )

    def plot_decay_times_against_occupation(self):
        (fig, (ax, lax)) = plt.subplots(2)  # type: ignore

        all_occupations = []
        all_decay_times = []
        for number_of_states, data_for_number_of_states in self._data.items():
            plot_occupations = []
            plot_decay_times = []
            plot_decay_errors = []

            for (
                number_of_electrons,
                data_for_number_of_electrons,
            ) in data_for_number_of_states.items():
                if len(data_for_number_of_electrons) > 0:
                    plot_occupations.append(number_of_electrons / number_of_states)
                    plot_decay_times.append(
                        1 / np.average([d[0] for d in data_for_number_of_electrons])
                    )
                    plot_decay_errors.append(
                        np.var([1 / d[0] for d in data_for_number_of_electrons])
                    )
            ax.errorbar(
                plot_occupations,
                plot_decay_times,
                # yerr=plot_decay_errors,
                fmt="+",
                label=number_of_states,
            )

            all_occupations.extend(plot_occupations)
            all_decay_times.extend(plot_decay_times)

        a, b = self._fit_N4_time_curve(all_occupations, all_decay_times)
        ax.plot(
            np.linspace(0.01, 0.99, 1000),
            self._decay_N4_time_curve(np.linspace(0.1, 0.9, 1000), 256 * 350000),
            label="polynomial squared fit",
        )
        a, b = self._fit_exponential_time_curve(all_occupations, all_decay_times)
        ax.plot(
            np.linspace(0.01, 0.99, 1000),
            self._decay_exponential_time_curve(np.linspace(0.1, 0.9, 1000), a[0], a[1]),
            label="exponential fit",
        )
        # plt.legend(bbox_to_anchor=(1.04, 1))
        # fig.legend(loc=7)
        handles, labels = ax.get_legend_handles_labels()
        lax.legend(handles, labels, borderaxespad=0)
        lax.axis("off")

        plt.tight_layout()
        plt.show()
        print(a[0])

    def get_inverted_data(self):
        flipped = defaultdict(dict)
        for key, val in self._data.items():
            for subkey, subval in val.items():
                flipped[subkey][key] = subval

        return dict(flipped)

    def plot_rate_for_each_elecron(self, ignore_single, ax):
        all_occupations = []
        all_decay_rates = []
        data = self.get_inverted_data()
        for number_of_electrons, data_for_number_of_electrons in data.items():
            plot_occupations = []
            plot_decay_rates = []
            plot_decay_rate_errors = []

            for (
                number_of_states,
                data_for_number_of_states,
            ) in data_for_number_of_electrons.items():
                if not (
                    ignore_single
                    and (
                        number_of_electrons == 1
                        or number_of_electrons == number_of_states - 1
                    )
                ):
                    plot_occupations.append(number_of_electrons / number_of_states)
                    plot_decay_rates.append(
                        1 / np.average([d[0] for d in data_for_number_of_states])
                    )
                    plot_decay_rate_errors.append(
                        np.average([d[0] for d in data_for_number_of_states]) ** (-2)
                        * np.var([d[0] for d in data_for_number_of_states]) ** (0.5)
                    )
            ax.errorbar(
                plot_occupations,
                plot_decay_rates,
                yerr=np.array(plot_decay_rate_errors),
                fmt="+",
                label=number_of_electrons,
            )

            all_occupations.extend(plot_occupations)
            all_decay_rates.extend(plot_decay_rates)
        return all_occupations, all_decay_rates

    def plot_rate_for_each_state(self, ignore_single, ax):
        all_occupations = []
        all_decay_rates = []
        for number_of_states, data_for_number_of_states in self._data.items():
            plot_occupations = []
            plot_decay_rates = []
            plot_decay_rate_errors = []

            for (
                number_of_electrons,
                data_for_number_of_electrons,
            ) in data_for_number_of_states.items():
                if len(data_for_number_of_electrons) > 0 and (
                    not (
                        ignore_single
                        and (
                            number_of_electrons == 1
                            or number_of_electrons == number_of_states - 1
                        )
                    )
                ):
                    plot_occupations.append(number_of_electrons / number_of_states)
                    plot_decay_rates.append(
                        1 / np.average([d[0] for d in data_for_number_of_electrons])
                    )
                    plot_decay_rate_errors.append(
                        np.average([d[0] for d in data_for_number_of_electrons]) ** (-2)
                        * np.var([d[0] for d in data_for_number_of_electrons]) ** (0.5)
                    )
            ax.errorbar(
                plot_occupations,
                plot_decay_rates,
                yerr=np.array(plot_decay_rate_errors),
                fmt="+",
                label=number_of_states,
            )

            all_occupations.extend(plot_occupations)
            all_decay_rates.extend(plot_decay_rates)
        return all_occupations, all_decay_rates

    def plot_decay_rates_against_occupation(
        self,
        ignore_single=False,
        for_each="state",
        display=["cosh", "cosh exponential", "cosh with log", "linear"],
    ):
        fig = plt.figure(figsize=(10, 5))
        spec = gs.GridSpec(1, 2, width_ratios=[20, 1])
        ax = fig.add_subplot(spec[0])
        lax = fig.add_subplot(spec[1])
        if for_each == "state":
            all_occupations, all_decay_rates = self.plot_rate_for_each_state(
                ignore_single, ax
            )
        elif for_each == "electron":
            all_occupations, all_decay_rates = self.plot_rate_for_each_elecron(
                ignore_single, ax
            )
        else:
            all_occupations, all_decay_rates = [[], []]

        if "quadratic" in display:
            a, b = self.fit_rate_curve(
                all_occupations,
                all_decay_rates,
                self._quadratic_decay_rate_curve,
            )
            ax.plot(
                np.linspace(0.01, 0.99, 1000),
                self._quadratic_decay_rate_curve(
                    np.linspace(0.01, 0.99, 1000), a[0], a[1]
                ),
                label="quadratic fit",
                linestyle="-",
                marker="",
            )
        if "quadratic2" in display:
            a, b = self.fit_rate_curve(
                all_occupations,
                all_decay_rates,
                self._fixed_quadratic_decay_rate_curve,
            )
            ax.plot(
                np.linspace(0.01, 0.99, 1000),
                self._fixed_quadratic_decay_rate_curve(
                    np.linspace(0.01, 0.99, 1000), a[0]
                ),
                label="quadratic fit 2",
                linestyle="-",
                marker="",
            )
            print(
                "Quadratic2",
                "fitted aplitude",
                a[0],
                "error",
                b,
            )
        if "linear" in display:
            a, b = self.fit_rate_curve(
                all_occupations, all_decay_rates, self._linear_decay_rate_curve
            )
            ax.plot(
                np.linspace(0.01, 0.99, 1000),
                self._linear_decay_rate_curve(
                    np.linspace(0.01, 0.99, 1000), a[0], a[1]
                ),
                label="linear fit",
                linestyle="-",
                marker="",
            )
        if False:
            a, b = self.fit_rate_curve(
                all_occupations,
                np.log(all_decay_rates),
                self._linear_decay_rate_with_log_curve,
            )
            ax.plot(
                np.linspace(0.01, 0.99, 1000),
                np.exp(
                    self._linear_decay_rate_with_log_curve(
                        np.linspace(0.01, 0.99, 1000), a[0], a[1]
                    )
                ),
                label="linear fit with N(1-N) prefactor",
                linestyle="-",
                marker="",
            )
        if "cosh" in display:
            a, b = self.fit_rate_curve(
                all_occupations, np.log(all_decay_rates), self._cosh_decay_rate_curve
            )
            ax.plot(
                np.linspace(0.01, 0.99, 1000),
                np.exp(
                    self._cosh_decay_rate_curve(
                        np.linspace(0.01, 0.99, 1000), a[0], a[1]
                    )
                ),
                label="cosh fit",
                linestyle="-",
                marker="",
            )
        if "cosh exponential" in display:
            a, b = self.fit_rate_curve(
                all_occupations,
                all_decay_rates,
                lambda a, b, c: np.exp(self._cosh_decay_rate_curve(a, b, c)),
            )
            ax.plot(
                np.linspace(0.01, 0.99, 1000),
                np.exp(
                    self._cosh_decay_rate_curve(
                        np.linspace(0.01, 0.99, 1000), a[0], a[1]
                    )
                ),
                label="cosh fit exponential",
                linestyle="-",
                marker="",
            )

        handles, labels = ax.get_legend_handles_labels()
        lax.legend(handles, labels, borderaxespad=0)
        lax.axis("off")

        fig.tight_layout()
        plt.show()

    def plot_log_decay_rates_against_occupation(
        self,
        ignore_single=False,
        display=["cosh", "cosh exponential", "cosh with log", "linear"],
    ):
        fig = plt.figure(figsize=(10, 5))
        spec = gs.GridSpec(1, 2, width_ratios=[20, 1])
        ax = fig.add_subplot(spec[0])
        lax = fig.add_subplot(spec[1])
        all_occupations = []
        all_decay_rates = []
        for number_of_states, data_for_number_of_states in self._data.items():
            plot_occupations = []
            plot_decay_rates = []

            for (
                number_of_electrons,
                data_for_number_of_electrons,
            ) in data_for_number_of_states.items():
                if len(data_for_number_of_electrons) > 0 and (
                    not (
                        ignore_single
                        and (
                            number_of_electrons == 1
                            or number_of_electrons == number_of_states - 1
                        )
                    )
                ):
                    plot_occupations.append(number_of_electrons / number_of_states)
                    plot_decay_rates.append(
                        np.log(
                            1 / np.average([d[0] for d in data_for_number_of_electrons])
                        )
                    )
            ax.errorbar(
                plot_occupations,
                plot_decay_rates,
                fmt="+",
                label=number_of_states,
            )

            all_occupations.extend(plot_occupations)
            all_decay_rates.extend(plot_decay_rates)

        if False:
            a, b = self.fit_rate_curve(
                all_occupations, all_decay_rates, self._sqrt_quadratic_decay_rate_curve
            )
            ax.plot(
                np.linspace(0.01, 0.99, 1000),
                self._sqrt_quadratic_decay_rate_curve(
                    np.linspace(0.01, 0.99, 1000), a[0], a[1]
                ),
                label="sqrt quadratic fit",
                linestyle="-",
                marker="",
            )

        if "quadratic" in display:
            a, b = self.fit_rate_curve(
                all_occupations, all_decay_rates, self._quadratic_decay_rate_curve
            )
            ax.plot(
                np.linspace(0.01, 0.99, 1000),
                self._quadratic_decay_rate_curve(
                    np.linspace(0.01, 0.99, 1000), a[0], a[1]
                ),
                label="quadratic fit",
                linestyle="-",
                marker="",
            )
            print(
                "Quadratic",
                "fitted aplitude",
                a[0],
                "fitted offset",
                a[1],
                "error",
                b,
            )

        if "quadratic2" in display:
            a, b = self.fit_rate_curve(
                all_occupations,
                np.exp(all_decay_rates),
                self._fixed_quadratic_decay_rate_curve,
            )
            ax.plot(
                np.linspace(0.01, 0.99, 1000),
                np.log(
                    self._fixed_quadratic_decay_rate_curve(
                        np.linspace(0.1, 0.9, 1000), a[0]
                    )
                ),
                label="quadratic fit 2",
                linestyle="-",
                marker="",
            )

        if "linear" in display:
            a, b = self.fit_rate_curve(
                all_occupations, np.exp(all_decay_rates), self._linear_decay_rate_curve
            )
            ax.plot(
                np.linspace(0.01, 0.99, 1000),
                np.log(
                    self._linear_decay_rate_curve(
                        np.linspace(0.01, 0.99, 1000), a[0], a[1]
                    )
                ),
                label="linear fit",
                linestyle="-",
                marker="",
            )
            print("fitted aplitude", a[0], "fitted offset", a[1], "error", b)

        if False:
            a, b = self.fit_rate_curve(
                all_occupations, all_decay_rates, self._linear_decay_rate_with_log_curve
            )
            ax.plot(
                np.linspace(0.01, 0.99, 1000),
                self._linear_decay_rate_with_log_curve(
                    np.linspace(0.01, 0.99, 1000), a[0], a[1]
                ),
                label="linear fit with N(1-N) prefactor",
                linestyle="-",
                marker="",
            )
        if "cosh" in display:
            a, b = self.fit_rate_curve(
                all_occupations, all_decay_rates, self._cosh_decay_rate_curve
            )
            ax.plot(
                np.linspace(0.01, 0.99, 1000),
                self._cosh_decay_rate_curve(np.linspace(0.01, 0.99, 1000), a[0], a[1]),
                label="cosh fit in log space",
                linestyle="-",
                marker="",
            )

            print("fitted log", a, "error", b)

        if "cosh exponential" in display:
            a, b = self.fit_rate_curve(
                all_occupations,
                np.exp(all_decay_rates),
                lambda a, b, c: np.exp(self._cosh_decay_rate_curve(a, b, c)),
            )
            ax.plot(
                np.linspace(0.01, 0.99, 1000),
                self._cosh_decay_rate_curve(np.linspace(0.01, 0.99, 1000), *a),
                label="cosh fit",
                linestyle="-",
                marker="",
            )
            print("fitted exp", a, "error", b)

        if "cosh with log" in display:
            a, b = self.fit_rate_curve(
                all_occupations, all_decay_rates, self._cosh_decay_rate_with_log_curve
            )
            ax.plot(
                np.linspace(0.01, 0.99, 1000),
                self._cosh_decay_rate_with_log_curve(
                    np.linspace(0.01, 0.99, 1000), a[0], a[1]
                ),
                label="cosh with log fit",
                linestyle="-",
                marker="",
            )

        handles, labels = ax.get_legend_handles_labels()
        lax.legend(handles, labels, borderaxespad=0)
        lax.axis("off")

        fig.tight_layout()
        plt.show()


class OneBandMaterialSimualtorDataUtil:
    @classmethod
    def simulate_all_tunnelling_times(
        cls,
        number_of_states,
        stop_times: Dict[int, float],
        number_of_points: Dict[int, int],
        average_over,
        number_of_repeats,
        save_to,
    ):

        for number_of_electrons, stop_time in stop_times.items():
            nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
                OneBandMaterialSimulator,
                temperature=150,
                number_of_states_per_band=number_of_states,
                target_frequency=1 * 10 ** (9),
                number_of_electrons=number_of_electrons,
            )

            if number_of_points[number_of_electrons] != 0:
                cls.simulate_tunnelling_times(
                    simulator=nickel_sim,
                    times=np.linspace(
                        0, stop_time, number_of_points[number_of_electrons]
                    ).tolist(),
                    average_over=average_over,
                    number_of_repeats=number_of_repeats,
                    save_to=save_to,
                )

    @staticmethod
    def simulate_tunnelling_times(
        simulator: OneBandMaterialSimulator,
        times: List[float],
        average_over,
        number_of_repeats,
        save_to,
    ):

        data = OneBandMaterialSimualtorData.load_from_file(save_to)
        for i in range(number_of_repeats):
            average_densities = simulator.simulate_average_densities(
                times, average_over, jitter_electrons=True
            )
            initially_occupied_densities = [d[0] for d in average_densities]
            total_densities = [sum(x) for x in initially_occupied_densities]

            data.add_data(simulator, total_densities, times)
            print("done", simulator.number_of_electrons, i)
            data.save_to_file(save_to)

        data.save_to_file(save_to)

        return data

    @staticmethod
    def plot_decay_times_fitted_ex_squared(load_from):
        data = OneBandMaterialSimualtorData.load_from_file(load_from)
        # omega = {
        #     8: {6: 2189000},
        # }
        # plot_each = True
        # Longer time, wt smaller w smaller
        decay_data = data.generate_decay_time_data_fit_ex_squared(
            omega_150K, plot_each=False
        )
        decay_data.plot_decay_rates_against_occupation(ignore_single=True, display=[])
        decay_data.plot_log_decay_rates_against_occupation(
            ignore_single=True, display=[]
        )

    @staticmethod
    def plot_decay_times_fitted_ex_squared_manually(load_from):
        data = OneBandMaterialSimualtorData.load_from_file(load_from)

        plot_each = False
        # omega = {
        #     8: {6: 2189000},
        # }
        # Longer time, wt smaller w smaller
        decay_data = data.generate_decay_time_data_fit_ex_squared_manually(
            omega_150K, times_150K, plot_each=plot_each
        )
        decay_data.plot_decay_rates_against_occupation(ignore_single=True)
        decay_data.plot_log_decay_rates_against_occupation(ignore_single=True)

    @staticmethod
    def plot_decay_times_fitted_ex_squared_corrected_manually(load_from):
        data = OneBandMaterialSimualtorData.load_from_file(load_from)

        plot_each = True
        # omega = {
        #     8: {6: 2189000},
        # }
        # Longer time, wt smaller w smaller
        decay_data = data.generate_decay_time_data_fit_ex_squared_manually(
            omega_150k_corrected, times_150k_corrected, plot_each=plot_each
        )
        decay_data.plot_decay_rates_against_occupation(
            ignore_single=True, display=["linear", "quadratic2"]
        )
        decay_data.plot_log_decay_rates_against_occupation(
            ignore_single=True, display=["linear", "quadratic2"]
        )

    @staticmethod
    def plot_decay_times_fitted_ex_squared_corrected(load_from):
        data = OneBandMaterialSimualtorData.load_from_file(load_from)
        data = data.filter_data(
            filter=lambda n_states, n_electrons: True  # n_states == 8 and n_electrons == 5
        )
        plot_each = False
        decay_data = data.generate_decay_time_data_fit_ex_squared_wavepacket(
            plot_each=plot_each, fixed_amplitude=False
        )
        decay_data.plot_decay_rates_against_occupation(
            display=["linear", "quadratic2"], for_each="electron"
        )
        decay_data.plot_log_decay_rates_against_occupation(
            display=["linear", "quadratic2"]
        )

    @staticmethod
    def plot_decay_times_fitted_ex(load_from):
        data = OneBandMaterialSimualtorData.load_from_file(load_from)

        plot_each = False
        decay_data = data.generate_decay_time_data_fit_ex(
            omega_150K, plot_each=plot_each
        )
        decay_data.plot_decay_rates_against_occupation()
        decay_data.plot_log_decay_rates_against_occupation()

    @staticmethod
    def fix_data(stop_times, load_from, save_to):
        data = OneBandMaterialSimualtorData.load_from_file(load_from)._data
        for i, d in data.items():
            for j, old_data in d.items():
                new_data = []
                for d1 in old_data:
                    initially_occupied_densities = [d[0] for d in d1]
                    total_densities = [sum(x) for x in initially_occupied_densities]
                    times = np.linspace(0, stop_times[j], 1000).tolist()
                    new_data.append((times, total_densities))

                data[i][j] = new_data
        OneBandMaterialSimualtorData(data).save_to_file(save_to)

    @staticmethod
    def delete_data(load_from):
        data = OneBandMaterialSimualtorData.load_from_file(load_from)
        print(len(data._data[41][1]))
        print(len(data._data[41][1][:-10]))
        # _data = data._data
        # _data[41][1] = data._data[41][1][:-10]
        # # data.clear_data(16, 1)
        # # data.clear_data(23, 22)
        # data._data = _data
        print(len(data._data[41][1]))
        data.save_to_file(load_from)


# def simulate_times():
#     print("7")
#     OneBandMaterialSimualtorDataUtil.simulate_all_tunnelling_times(
#         number_of_states=7,
#         stop_times={
#             2: 15 * 10 ** -5,
#             3: 2.5 * 10 ** -5,
#             4: 2.5 * 10 ** -5,
#             5: 50 * 10 ** -5,
#         },
#         number_of_points={
#             2: 1000,
#             3: 1000,
#             4: 1000,
#             5: 1000,
#         },
#         average_over=20,
#         number_of_repeats=10,
#         save_to="conduit data/150K data with exchange.pkl",
#     )
#     print("8")
#     OneBandMaterialSimualtorDataUtil.simulate_all_tunnelling_times(
#         number_of_states=8,
#         stop_times={
#             2: 20 * 10 ** -5,
#             3: 5 * 10 ** -5,
#             4: 2 * 10 ** -5,
#             5: 5 * 10 ** -5,
#             6: 20 * 10 ** -5,
#         },
#         number_of_points={
#             2: 1000,
#             3: 1000,
#             4: 1000,
#             5: 1000,
#             6: 1000,
#         },
#         average_over=20,
#         number_of_repeats=10,
#         save_to="conduit data/150K data with exchange.pkl",
#     )
#     print("9")
#     OneBandMaterialSimualtorDataUtil.simulate_all_tunnelling_times(
#         number_of_states=9,
#         stop_times={
#             2: 20 * 10 ** -5,
#             3: 6 * 10 ** -5,
#             4: 2.5 * 10 ** -5,
#             5: 2.5 * 10 ** -5,
#             6: 6 * 10 ** -5,
#             7: 20 * 10 ** -5,
#         },
#         number_of_points={
#             2: 1000,
#             3: 1000,
#             4: 1000,
#             5: 1000,
#             6: 1000,
#             7: 1000,
#         },
#         average_over=20,
#         number_of_repeats=10,
#         save_to="conduit data/150K data with exchange.pkl",
#     )
#     print("10")
#     OneBandMaterialSimualtorDataUtil.simulate_all_tunnelling_times(
#         number_of_states=10,
#         stop_times={
#             2: 20 * 10 ** -5,
#             3: 5 * 10 ** -4,
#             4: 4 * 10 ** -4,
#             5: 4 * 10 ** -4,
#             6: 4 * 10 ** -4,
#             7: 5 * 10 ** -4,
#             8: 20 * 10 ** -5,
#         },
#         number_of_points={
#             2: 2000,
#             3: 1000,
#             4: 1000,
#             5: 1000,
#             6: 1000,
#             7: 1000,
#             8: 2000,
#         },
#         average_over=20,
#         number_of_repeats=10,
#         save_to="conduit data/150K data with exchange.pkl",
#     )
#     print("11")
#     OneBandMaterialSimualtorDataUtil.simulate_all_tunnelling_times(
#         number_of_states=11,
#         stop_times={
#             1: 30 * 10 ** -5,
#             2: 25 * 10 ** -5,
#             3: 20 * 10 ** -5,
#             8: 20 * 10 ** -5,
#             9: 25 * 10 ** -5,
#             10: 30 * 10 ** -5,
#         },
#         number_of_points={
#             1: 1000,
#             2: 2000,
#             3: 1000,
#             8: 1000,
#             9: 2000,
#             10: 1000,
#         },
#         average_over=20,
#         number_of_repeats=10,
#         save_to="conduit data/150K data with exchange.pkl",
#     )
#     print("12")
#     OneBandMaterialSimualtorDataUtil.simulate_all_tunnelling_times(
#         number_of_states=12,
#         stop_times={
#             2: 35 * 10 ** -5,
#             3: 15 * 10 ** -5,
#             9: 15 * 10 ** -5,
#             10: 35 * 10 ** -5,
#         },
#         number_of_points={
#             2: 1000,
#             3: 1000,
#             9: 1000,
#             10: 1000,
#         },
#         average_over=20,
#         number_of_repeats=10,
#         save_to="conduit data/150K data with exchange.pkl",
#     )
#     print("13")
#     OneBandMaterialSimualtorDataUtil.simulate_all_tunnelling_times(
#         number_of_states=13,
#         stop_times={
#             1: 40 * 10 ** -5,
#             2: 35 * 10 ** -5,
#             11: 35 * 10 ** -5,
#             12: 40 * 10 ** -5,
#         },
#         number_of_points={
#             1: 1000,
#             2: 1000,
#             11: 1000,
#             12: 1000,
#         },
#         average_over=20,
#         number_of_repeats=10,
#         save_to="conduit data/150K data with exchange.pkl",
#     )
#     print("18")
#     OneBandMaterialSimualtorDataUtil.simulate_all_tunnelling_times(
#         number_of_states=18,
#         stop_times={
#             1: 100 * 10 ** -5,
#             17: 100 * 10 ** -5,
#         },
#         number_of_points={
#             1: 1000,
#             17: 1000,
#         },
#         average_over=20,
#         number_of_repeats=10,
#         save_to="conduit data/150K data with exchange.pkl",
#     )
#     print("21")
#     OneBandMaterialSimualtorDataUtil.simulate_all_tunnelling_times(
#         number_of_states=21,
#         stop_times={
#             1: 100 * 10 ** -5,
#             20: 100 * 10 ** -5,
#         },
#         number_of_points={
#             1: 1000,
#             20: 1000,
#         },
#         average_over=20,
#         number_of_repeats=10,
#         save_to="conduit data/150K data with exchange.pkl",
#     )
#     print("41")
#     OneBandMaterialSimualtorDataUtil.simulate_all_tunnelling_times(
#         number_of_states=41,
#         stop_times={
#             1: 125 * 10 ** -5,
#             40: 125 * 10 ** -5,
#         },
#         number_of_points={
#             1: 1000,
#             40: 1000,
#         },
#         average_over=20,
#         number_of_repeats=10,
#         save_to="conduit data/150K data with exchange.pkl",
#     )
#     print("67")
#     OneBandMaterialSimualtorDataUtil.simulate_all_tunnelling_times(
#         number_of_states=67,
#         stop_times={
#             1: 300 * 10 ** -5,
#             66: 300 * 10 ** -5,
#         },
#         number_of_points={
#             1: 1000,
#             66: 1000,
#         },
#         average_over=20,
#         number_of_repeats=10,
#         save_to="conduit data/150K data with exchange.pkl",
#     )
#     print("97")
#     OneBandMaterialSimualtorDataUtil.simulate_all_tunnelling_times(
#         number_of_states=97,
#         stop_times={
#             1: 400 * 10 ** -5,
#             96: 400 * 10 ** -5,
#         },
#         number_of_points={
#             1: 1000,
#             96: 1000,
#         },
#         average_over=20,
#         number_of_repeats=10,
#         save_to="conduit data/150K data with exchange.pkl",
#     )


def simulate_times():
    for n_states, stop_times in rough_150K_corrected_50_average.items():
        print(n_states)
        n_points = {n_electrons: 2000 for n_electrons in stop_times.keys()}
        OneBandMaterialSimualtorDataUtil.simulate_all_tunnelling_times(
            number_of_states=n_states,
            stop_times=stop_times,
            number_of_points=n_points,
            average_over=50,
            number_of_repeats=4,
            save_to="conduit data/150K data with exchange2.pkl",
        )


if __name__ == "__main__":
    simulate_times()

    # OneBandMaterialSimualtorDataUtil.delete_data("conduit data/150K data.pkl")
    # OneBandMaterialSimualtorDataUtil.plot_decay_times_fitted_ex(
    #     "conduit data/150K data.pkl"
    # )
    OneBandMaterialSimualtorDataUtil.plot_decay_times_fitted_ex_squared_corrected(
        "conduit data/150K data with exchange2.pkl"
    )
    # OneBandMaterialSimualtorDataUtil.plot_decay_times_fitted_ex_squared_corrected_manually(
    #     "conduit data/150K data with exchange.pkl"
    # )
    # OneBandMaterialSimualtorDataUtil.plot_decay_times_fitted_ex_squared(
    #     "conduit data/150K data.pkl"
    # )
