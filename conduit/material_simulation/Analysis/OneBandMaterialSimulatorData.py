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
        return -amplitude * occupation * (1 - occupation) + offset

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

    def plot_decay_rates_against_occupation(
        self,
        ignore_single=False,
        display=["cosh", "cosh exponential", "cosh with log"],
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
                        1 / np.average([d[0] for d in data_for_number_of_electrons])
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
                all_occupations,
                np.log(all_decay_rates),
                self._quadratic_decay_rate_curve,
            )
            ax.plot(
                np.linspace(0.01, 0.99, 1000),
                np.exp(
                    self._quadratic_decay_rate_curve(
                        np.linspace(0.01, 0.99, 1000), a[0], a[1]
                    )
                ),
                label="polynomial fit",
                linestyle="-",
                marker="",
            )
        if False:
            a, b = self.fit_rate_curve(
                all_occupations, np.log(all_decay_rates), self._linear_decay_rate_curve
            )
            ax.plot(
                np.linspace(0.01, 0.99, 1000),
                np.exp(
                    self._linear_decay_rate_curve(
                        np.linspace(0.01, 0.99, 1000), a[0], a[1]
                    )
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
        display=["cosh", "cosh exponential", "cosh with log"],
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

        if False:
            a, b = self.fit_rate_curve(
                all_occupations, all_decay_rates, self._quadratic_decay_rate_curve
            )
            ax.plot(
                np.linspace(0.01, 0.99, 1000),
                self._quadratic_decay_rate_curve(
                    np.linspace(0.01, 0.99, 1000), a[0], a[1]
                ),
                label="polynomial fit",
                linestyle="-",
                marker="",
            )

        if False:
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
                temperature=120,
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
        decay_data.plot_decay_rates_against_occupation(ignore_single=True)
        decay_data.plot_log_decay_rates_against_occupation(ignore_single=True)

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


def simulate_times():
    print("7")
    OneBandMaterialSimualtorDataUtil.simulate_all_tunnelling_times(
        number_of_states=7,
        stop_times={
            2: 15 * 10 ** -5,
            3: 2.5 * 10 ** -5,
            4: 2.5 * 10 ** -5,
            5: 50 * 10 ** -5,
        },
        number_of_points={
            2: 1000,
            3: 1000,
            4: 1000,
            5: 1000,
        },
        average_over=40,
        number_of_repeats=10,
        save_to="conduit data/120K data.pkl",
    )
    print("8")
    OneBandMaterialSimualtorDataUtil.simulate_all_tunnelling_times(
        number_of_states=8,
        stop_times={
            2: 20 * 10 ** -5,
            3: 5 * 10 ** -5,
            4: 2 * 10 ** -5,
            5: 5 * 10 ** -5,
            6: 20 * 10 ** -5,
        },
        number_of_points={
            2: 1000,
            3: 1000,
            4: 1000,
            5: 1000,
            6: 1000,
        },
        average_over=30,
        number_of_repeats=10,
        save_to="conduit data/120K data.pkl",
    )
    print("9")
    OneBandMaterialSimualtorDataUtil.simulate_all_tunnelling_times(
        number_of_states=9,
        stop_times={
            2: 20 * 10 ** -5,
            3: 6 * 10 ** -5,
            6: 6 * 10 ** -5,
            7: 20 * 10 ** -5,
        },
        number_of_points={
            2: 1000,
            3: 1000,
            6: 1000,
            7: 1000,
        },
        average_over=20,
        number_of_repeats=10,
        save_to="conduit data/120K data.pkl",
    )
    print("12")
    OneBandMaterialSimualtorDataUtil.simulate_all_tunnelling_times(
        number_of_states=12,
        stop_times={
            2: 35 * 10 ** -5,
            10: 35 * 10 ** -5,
        },
        number_of_points={
            2: 1000,
            10: 1000,
        },
        average_over=20,
        number_of_repeats=10,
        save_to="conduit data/120K data.pkl",
    )
    print("13")
    OneBandMaterialSimualtorDataUtil.simulate_all_tunnelling_times(
        number_of_states=13,
        stop_times={
            2: 35 * 10 ** -5,
            11: 35 * 10 ** -5,
        },
        number_of_points={
            2: 1000,
            11: 1000,
        },
        average_over=20,
        number_of_repeats=10,
        save_to="conduit data/120K data.pkl",
    )
    print("10")
    OneBandMaterialSimualtorDataUtil.simulate_all_tunnelling_times(
        number_of_states=10,
        stop_times={
            2: 20 * 10 ** -5,
            8: 20 * 10 ** -5,
        },
        number_of_points={
            2: 2000,
            8: 2000,
        },
        average_over=20,
        number_of_repeats=10,
        save_to="conduit data/120K data.pkl",
    )
    print("11")
    OneBandMaterialSimualtorDataUtil.simulate_all_tunnelling_times(
        number_of_states=11,
        stop_times={
            2: 25 * 10 ** -5,
            9: 25 * 10 ** -5,
        },
        number_of_points={
            2: 2000,
            9: 2000,
        },
        average_over=20,
        number_of_repeats=10,
        save_to="conduit data/120K data.pkl",
    )


if __name__ == "__main__":
    # simulate_times()

    # OneBandMaterialSimualtorDataUtil.delete_data("conduit data/150K data.pkl")
    # OneBandMaterialSimualtorDataUtil.plot_decay_times_fitted_ex(
    #     "conduit data/150K data.pkl"
    # )
    # OneBandMaterialSimualtorDataUtil.plot_decay_times_fitted_ex_squared_manually(
    #     "conduit data/150K data.pkl"
    # )
    OneBandMaterialSimualtorDataUtil.plot_decay_times_fitted_ex_squared(
        "conduit data/150K data.pkl"
    )
