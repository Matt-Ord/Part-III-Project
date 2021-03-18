# Utility class used to collect data from the one band material
# simulator
import pickle
from typing import Any, Dict, List

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
    def _occupation_curve_fit_function(time, omega, decay_time):
        return ElectronSimulationPlotter.occupation_curve_fit_function(
            1, time, omega, decay_time
        )

    def _fit_electron_occupation_curve(self, times, normalised_densities, omega) -> Any:
        initial_decay_time_guess = (times[-1] - times[0]) / 4

        lower_bounds = [0.1 * initial_decay_time_guess]
        upper_bounds = [4 * initial_decay_time_guess]
        return scipy.optimize.curve_fit(
            lambda t, x: self._occupation_curve_fit_function(t, omega, x),
            times,
            normalised_densities,
            p0=[initial_decay_time_guess],
            bounds=[lower_bounds, upper_bounds],
        )

    def generate_decay_time_data(self, omega):
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
                    if omega[number_of_states][number_of_electrons] != 0:
                        try:
                            a, b = self._fit_electron_occupation_curve(
                                times,
                                normalised_densities,
                                omega[number_of_states][number_of_electrons],
                            )
                            # ElectronSimulationPlotter._plot_total_number_against_time(
                            #     number_in_each_state={
                            #         "actual": normalised_densities,
                            #         "average": self._occupation_curve_fit_function(
                            #             np.array(times),
                            #             omega[number_of_states][number_of_electrons],
                            #             a[0],
                            #         ),
                            #     },
                            #     times=times,
                            # )
                            # plt.show()
                            decay_time_data[number_of_states][
                                number_of_electrons
                            ].append((a[0], b[0][0]))
                        except RuntimeError:
                            self.plot_data(number_of_states, number_of_electrons, i)
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
    def _decay_N_time_curve(occupation, amplitude):
        return amplitude * occupation * (1 - occupation)

    @classmethod
    def _fit_N_time_curve(cls, all_occupations, all_decay_times) -> Any:
        return scipy.optimize.curve_fit(
            f=cls._decay_N_time_curve,
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
        (fig, ax) = plt.subplots(1)
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

        a, b = self._fit_N_time_curve(all_occupations, all_decay_times)
        ax.plot(
            np.linspace(0.1, 0.9, 1000),
            self._decay_N_time_curve(np.linspace(0.1, 0.9, 1000), a[0]),
            label="polynomial fit",
        )
        a, b = self._fit_N4_time_curve(all_occupations, all_decay_times)
        ax.plot(
            np.linspace(0.1, 0.9, 1000),
            self._decay_N4_time_curve(np.linspace(0.1, 0.9, 1000), 256 * 350000),
            label="polynomial squared fit",
        )
        a, b = self._fit_exponential_time_curve(all_occupations, all_decay_times)
        ax.plot(
            np.linspace(0.1, 0.9, 1000),
            self._decay_exponential_time_curve(np.linspace(0.1, 0.9, 1000), a[0], a[1]),
            label="exponential fit",
        )
        ax.legend()
        plt.show()


class OneBandMaterialSimualtorDataUtil:
    @classmethod
    def simulate_all_tunnelling_times(
        cls,
        number_of_states,
        stop_times: List[float],
        number_of_points: List[int],
        average_over,
        number_of_repeats,
        save_to,
    ):

        for number_of_electrons, stop_time in enumerate(stop_times):
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
    def plot_decay_times(load_from):
        data = OneBandMaterialSimualtorData.load_from_file(load_from)
        omega = {
            7: [0, 0, 2810000, 5500000, 5500000, 2810000, 0, 0],
            8: [0, 0, 2459000, 4900000, 7250000, 4900000, 2459000, 0, 0],
            9: [0, 0, 2189000, 4365000, 0, 0, 4365000, 2189000, 0, 0],
        }
        # omega = {
        #     7: [0, 0, 0, 0, 0, 0, 0, 0],
        #     8: [0, 0, 0, 0, 0, 0, 0, 0, 0],
        #     9: [0, 0, 0, 0, 5800000, 0, 0, 0, 0, 0],
        # }
        # Longer time, wt smaller w smaller
        decay_data = data.generate_decay_time_data(omega)
        decay_data.plot_decay_times_against_occupation()

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
        print(data._data[9][6])
        # data.clear_data(9, 6)
        data.save_to_file(load_from)


def fix_data():
    OneBandMaterialSimualtorDataUtil.fix_data(
        stop_times=[
            0,
            40 * 10 ** -5,
            20 * 10 ** -5,
            5 * 10 ** -5,
            2 * 10 ** -5,
            5 * 10 ** -5,
            20 * 10 ** -5,
            40 * 10 ** -5,
            0,
        ],
        load_from="conduit data/150K data.pkl",
        save_to="conduit data/150K data fixed.pkl",
    )


if __name__ == "__main__":
    # OneBandMaterialSimualtorDataUtil.simulate_all_tunnelling_times(
    #     number_of_states=10,
    #     stop_times=[
    #         0,
    #         0,
    #         20 * 10 ** -5,
    #         6 * 10 ** -5,
    #         2.5 * 10 ** -5,
    #         0,
    #         2.5 * 10 ** -5,
    #         6 * 10 ** -5,
    #         20 * 10 ** -5,
    #         0,
    #         0,
    #     ],
    #     number_of_points=[
    #         0,
    #         0,
    #         2000,  # 2000,
    #         1000,  # 1000,
    #         1000,  # 1000,
    #         0,
    #         1000,
    #         1000,
    #         2000,
    #         0,
    #         0,
    #     ],
    #     average_over=40,
    #     number_of_repeats=30,
    #     save_to="conduit data/150K data.pkl",
    # )

    # OneBandMaterialSimualtorDataUtil.delete_data("conduit data/150K data fixed.pkl")
    OneBandMaterialSimualtorDataUtil.plot_decay_times("conduit data/150K data.pkl")
