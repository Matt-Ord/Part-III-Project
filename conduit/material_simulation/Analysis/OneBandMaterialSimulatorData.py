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
import material_simulation.analysis.one_band_rough_data as rough_data


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

    def _fit_decay_curve(
        self, times, normalised_densities, fit_fn, initial_guess
    ) -> Any:
        # lower_bounds = [0.1 * initial_decay_time_guess]
        # upper_bounds = [4 * initial_decay_time_guess]
        return scipy.optimize.curve_fit(
            fit_fn,
            times,
            normalised_densities,
            p0=initial_guess,
            # bounds=[[-np.inf, 0.3], [np.inf, 0.5]],
        )

    @staticmethod
    def _ex_decay_curve_fit_function(time, decay_time, amplitude):
        return amplitude * np.exp(-((time / decay_time))) + 0.5

    @staticmethod
    def _linear_decay_curve_fit_function(time, decay_time, amplitude):
        return amplitude * np.maximum(1 - (time / decay_time), 0) + 0.5

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

    def plot_each_decay_curve2(self, times, offset, fit_fn):
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
                        "average": fit_fn(
                            np.array(densities_data[0][0]),
                            time[0][0],
                            offset[number_of_states][number_of_electrons][0],
                        ),
                    },
                    times=densities_data[0][0],
                )
                plt.show()

    def generate_decay_time_data_fit_wavepacket(
        self, fit_fn, plot_each: bool = False, fixed_amplitude=None, max_over=10
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
                    upper_filtered_data = np.array(
                        [
                            max(normalised_densities[j : j + max_over])
                            for j in range(len(normalised_densities))
                        ]
                    )
                    # ElectronSimulationPlotter._plot_total_number_against_time(
                    #     number_in_each_state={
                    #         "actual": np.array(upper_filtered_data).tolist(),
                    #     },
                    #     times=times,
                    # )
                    # plt.show()
                    initial_decay_time_guess = (times[-1] - times[0]) / 4

                    initial_guess = (
                        [initial_decay_time_guess, 0.5]
                        if fixed_amplitude is None
                        else [initial_decay_time_guess]
                    )

                    a1, b1 = self._fit_decay_curve(
                        times,
                        upper_filtered_data,
                        fit_fn
                        if fixed_amplitude is None
                        else lambda t, x: fit_fn(t, x, fixed_amplitude),
                        initial_guess,
                    )

                    decay_time_data[number_of_states][number_of_electrons].append(
                        (a1[0], b1[0][0])
                    )
                    amplitude[number_of_states][number_of_electrons].append(
                        fixed_amplitude if fixed_amplitude is not None else a1[1]
                    )
                    if number_of_electrons != 1:
                        lower_filtered_data = 1 - np.array(
                            [
                                min(normalised_densities[j : j + max_over])
                                for j in range(len(normalised_densities))
                            ]
                        )
                        a2, b2 = self._fit_decay_curve(
                            times,
                            lower_filtered_data,
                            fit_fn
                            if fixed_amplitude is None
                            else lambda t, x: fit_fn(t, x, fixed_amplitude),
                            initial_guess,
                        )

                        decay_time_data[number_of_states][number_of_electrons].append(
                            (a2[0], b2[0][0])
                        )
                        amplitude[number_of_states][number_of_electrons].append(
                            fixed_amplitude if fixed_amplitude is not None else a1[1]
                        )

        if plot_each:
            self.plot_each_decay_curve2(
                decay_time_data,
                amplitude,
                fit_fn=fit_fn,
            )
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

    def get_flattened_data(self):
        flattened = defaultdict(dict)
        for key, val in self._data.items():
            for subkey, subval in val.items():
                flattened[(key, subkey)] = subval

        return dict(flattened)

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

    @staticmethod
    def _exponential_curve(y, offset, amplitude):
        return offset + (amplitude / y) ** 2

    @classmethod
    def _fit_rate_curve(cls, x, y, fit_kwargs) -> Any:
        return scipy.optimize.curve_fit(
            f=lambda *args: cls._exponential_curve(*args, **fit_kwargs),
            xdata=x,
            ydata=y,
            bounds=[[0, 0], [np.inf, np.inf]],
            sigma=y,
            p0=[np.min(y), 1],
        )

    def plot_decay_rate_against_number_of_electrons(self, fit_kwargs={}):
        flat_data = self.get_flattened_data()
        number_of_electrons = [k[1] for k in flat_data.keys()]
        decay_rates = [1 / np.average([d[0] for d in v]) for v in flat_data.values()]
        decay_errors = [
            np.average([d[0] for d in v]) ** (-2) * np.var([d[0] for d in v]) ** (0.5)
            for v in flat_data.values()
        ]
        # plot_decay_rates.append(
        #                 1 /
        #             )
        #             plot_decay_rate_errors.append(
        #                 np.average([d[0] for d in data_for_number_of_states]) ** (-2)
        #                 * np.var([d[0] for d in data_for_number_of_states]) ** (0.5)
        #             )
        print(np.min(decay_rates))
        a, b = self._fit_rate_curve(
            x=number_of_electrons, y=decay_rates, fit_kwargs=fit_kwargs
        )
        print(a)

        n_electrons_for_fit = np.linspace(2, 8, 1000)
        fig, ax = plt.subplots(1)
        ax.errorbar(
            number_of_electrons,
            np.array(decay_rates),
            label="data",
            yerr=decay_errors,
            fmt="+",
        )
        ax.plot(
            n_electrons_for_fit,
            self._exponential_curve(n_electrons_for_fit, *a, **fit_kwargs),
            label="fit",
        )
        ax.legend()
        ax.set_title("Plot of decay rate against number of electrons")
        plt.show()
        fig, ax = plt.subplots(1)
        log_errors = [dy / y for (y, dy) in zip(decay_rates, decay_errors)]
        ax.errorbar(
            np.log(number_of_electrons),
            np.log(np.array(decay_rates) - a[0]),
            label="data",
            yerr=log_errors,
            fmt="+",
        )
        ax.plot(
            np.log(n_electrons_for_fit),
            np.log(self._exponential_curve(n_electrons_for_fit, 0, a[1], **fit_kwargs)),
            label="fit",
        )
        ax.legend()
        ax.set_title("Plot of decay rate against number of electrons")
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
        diagonal_terms=True,
    ):

        for number_of_electrons, stop_time in stop_times.items():
            nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
                OneBandMaterialSimulator,
                temperature=150,
                number_of_states_per_band=number_of_states,
                target_frequency=1 * 10 ** (9),
                number_of_electrons=number_of_electrons,
            )
            if not diagonal_terms:
                nickel_sim.remove_diagonal_block_factors_for_simulation()

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
            rough_data.omega_150K, plot_each=False
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
            rough_data.omega_150K, rough_data.times_150K, plot_each=plot_each
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
        # decay_data = data.generate_decay_time_data_fit_ex_squared_manually(
        #     rough_data.omega_150k_corrected,
        #     rough_data.times_150k_corrected,
        #     plot_each=plot_each,
        # )
        # decay_data.plot_decay_rates_against_occupation(
        #     ignore_single=True, display=["linear", "quadratic2"]
        # )
        # decay_data.plot_log_decay_rates_against_occupation(
        #     ignore_single=True, display=["linear", "quadratic2"]
        # )

    @staticmethod
    def plot_decay_times_corrected(load_from):
        data = OneBandMaterialSimualtorData.load_from_file(load_from)
        plot_each = False
        data1 = data.filter_data(
            filter=lambda n_states, n_electrons: n_states
            > 3  # True  # n_states == 8 and n_electrons == 5
        )
        # decay_data = data1.generate_decay_time_data_fit_ex_squared_wavepacket(
        #     plot_each=plot_each, fixed_amplitude=False
        # )
        # decay_data.plot_decay_rates_against_occupation(
        #     display=["linear", "quadratic2"], for_each="electron"
        # )

        half_filled_data = data.filter_data(
            lambda n_states, n_electrons: n_states / n_electrons == 2 and n_states != 2
        )
        half_filled_decay_data = (
            half_filled_data.generate_decay_time_data_fit_wavepacket(
                fit_fn=data1._ex_squared_decay_curve_fit_function,
                plot_each=plot_each,
                fixed_amplitude=False,
            )
        )
        half_filled_decay_data.plot_decay_rate_against_number_of_electrons()

    @staticmethod
    def plot_decay_times_no_diagonal(load_from):
        data = OneBandMaterialSimualtorData.load_from_file(load_from)
        plot_each = False
        data1 = data.filter_data(
            filter=lambda n_states, n_electrons: n_states
            > 3  # True  # n_states == 8 and n_electrons == 5
        )

        decay_data = data1.generate_decay_time_data_fit_wavepacket(
            fit_fn=data1._linear_decay_curve_fit_function,
            plot_each=plot_each,
            # fixed_amplitude=0.5,
            max_over=30,
        )
        decay_data.plot_decay_rates_against_occupation(
            display=["linear", "quadratic2"], for_each="electron"
        )

        half_filled_data = data.filter_data(
            lambda n_states, n_electrons: n_states / n_electrons == 2 and n_states != 2
        )
        half_filled_decay_data = (
            half_filled_data.generate_decay_time_data_fit_wavepacket(
                fit_fn=data1._ex_decay_curve_fit_function,
                plot_each=plot_each,
                fixed_amplitude=False,
            )
        )
        half_filled_decay_data.plot_decay_rate_against_number_of_electrons(
            fit_kwargs={}
        )

    @staticmethod
    def plot_decay_times_fitted_ex(load_from):
        data = OneBandMaterialSimualtorData.load_from_file(load_from)

        plot_each = False
        decay_data = data.generate_decay_time_data_fit_ex(
            rough_data.omega_150K, plot_each=plot_each
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
    for n_states, stop_times in rough_data.rough_150K_100_average_no_diagonal.items():
        print(n_states)
        n_points = {n_electrons: 2000 for n_electrons in stop_times.keys()}
        OneBandMaterialSimualtorDataUtil.simulate_all_tunnelling_times(
            number_of_states=n_states,
            stop_times=stop_times,
            number_of_points=n_points,
            average_over=100,
            number_of_repeats=4,
            save_to="conduit data/150K data no diagonal.pkl",
            diagonal_terms=False,
        )


if __name__ == "__main__":
    # simulate_times()

    # OneBandMaterialSimualtorDataUtil.delete_data("conduit data/150K data.pkl")
    # OneBandMaterialSimualtorDataUtil.plot_decay_times_fitted_ex(
    #     "conduit data/150K data.pkl"
    # )
    OneBandMaterialSimualtorDataUtil.plot_decay_times_no_diagonal(
        "conduit data/150K data no diagonal.pkl"
    )
    OneBandMaterialSimualtorDataUtil.plot_decay_times_corrected(
        "conduit data/150K data with exchange2.pkl"
    )
    # OneBandMaterialSimualtorDataUtil.plot_decay_times_fitted_ex_squared_corrected_manually(
    #     "conduit data/150K data with exchange.pkl"
    # )
    # OneBandMaterialSimualtorDataUtil.plot_decay_times_fitted_ex_squared(
    #     "conduit data/150K data.pkl"
    # )
