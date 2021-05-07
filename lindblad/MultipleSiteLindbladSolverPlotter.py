from typing import Type
from MultipleSiteLindbladSolver import (
    MultipleSiteLindbladSolver,
    MultipleSiteWithSinksLindbladSolver,
    MultipleSiteWithHopsLindbladSolver,
)
import matplotlib.pyplot as plt
import numpy as np
from LindbladSolverPlotter import (
    calculate_fast_direction_lindblad_rate,
    calculate_slow_direction_lindblad_rate,
    rotating_wave_solution1,
    rotating_wave_solution2,
    calculate_combined_lindblad_rate,
)
import experemental_data
import scipy.optimize


class MultipleSiteLindbladSolverPlotter:
    def __init__(self, solver: MultipleSiteLindbladSolver) -> None:
        self.solver = solver

    @staticmethod
    def _plot_probabilities_against_time(times, propbabilities, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(times, np.abs(propbabilities), label=label)
        return ax

    def plot_central_terms_against_time(self, ax=None, labels=["P0", "P1", "P2"]):
        if ax is None:
            fig, ax = plt.subplots()
        self._plot_probabilities_against_time(
            self.solver.times,
            self.solver.p_value_at_index(0),
            label=labels[0],
            ax=ax,
        )
        first_neighbours = set(self.solver._get_neighbours_site_i(0).tolist())
        for j in first_neighbours:
            self._plot_probabilities_against_time(
                self.solver.times,
                self.solver.p_value_at_index(j),
                label=labels[1],
                ax=ax,
            )
        second_neighbours = set(
            np.concatenate(
                [self.solver._get_neighbours_site_i(j) for j in first_neighbours]
            ).tolist()
        )
        for j in (j for j in second_neighbours if j != 0):
            self._plot_probabilities_against_time(
                self.solver.times,
                self.solver.p_value_at_index(j),
                label=labels[2],
                ax=ax,
            )
        ax.set_ylim([0, 1])
        ax.set_xlim([self.solver.times[0], None])
        ax.set_title("Plot of the diagonal terms of the density matrix against time")
        ax.legend()
        return ax.get_figure(), ax

    def plot_summed_occupation_against_time(self, ax=None, labels=["fcc", "hcp"]):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        self._plot_probabilities_against_time(
            self.solver.times,
            self.solver.p_value_at_fcc(),
            label=labels[0],
            ax=ax,
        )
        self._plot_probabilities_against_time(
            self.solver.times,
            self.solver.p_value_at_hcp(),
            label=labels[1],
            ax=ax,
        )
        ax.set_ylim([0, 1])
        ax.set_xlim([self.solver.times[0], None])
        ax.set_title("Plot of the summed probabilities of the density matrix")
        ax.legend()
        return fig, ax

    def plot_mean_square_distance_against_time(self, ax=None, label=""):
        if ax is None:
            fig, ax = plt.subplots(1)

        ax.plot(self.solver.times, self.solver.get_mean_square_distances(), label=label)
        return ax.get_figure(), ax

    def plot_solution(self):
        self.plot_central_terms_against_time()
        self.plot_summed_occupation_against_time()
        plt.show()
        return self

    def print_final_density(self):
        print(
            self.solver.p_value_at_index(0)[-1],
            self.solver.p_value_at_index(1)[-1],
        )
        return self

    @classmethod
    def from_file(cls, solver_type: Type[MultipleSiteLindbladSolver], file):
        return cls(solver_type.load_from_file(file))


def fallen_by(x):
    def fallen_by_event(t, y):
        return y[0] - x

    fallen_by_event.terminal = True
    return fallen_by_event


def calculate_lindblad_rate(
    temperatures, shape=(10, 10), tunnell_condition=lambda t: np.exp(-1)
):
    initial_state_grid = np.zeros(shape=shape)
    initial_state_grid[0][0] = 1
    rates = []
    for t in temperatures:

        tunnell_condition.terminal = True

        solver = MultipleSiteWithSinksLindbladSolver(
            times=np.linspace(0, 5 * 10 ** (-8), 2000),
            temperature=t,
            initial_state_grid=initial_state_grid,
            events=fallen_by(tunnell_condition(t)),
            # sink_coords=[(1, 0), (-1, 0), (-1, -1)],
        )
        rates.append(1 / solver.t_events[0][0])
    return np.array(rates)


def upper_line_of_two_site_occupation(temperature=150):
    solver = MultipleSiteLindbladSolver(
        times=[0],
        temperature=temperature,
        initial_state_grid=np.array([[1], [0]]),
    )
    # Calculates the upper line
    gamma1 = 2 * solver._get_gamma_abcd_omega_ij(1, 0, 1, 0, 1, 0)
    gamma2 = 2 * solver._get_gamma_abcd_omega_ij(0, 1, 0, 1, 0, 1)

    upper_line = gamma2 / (gamma1 + gamma2)
    return upper_line


def e_fall_to_upper_line_of_two_site_occupation(temperature=150):
    upper_line = upper_line_of_two_site_occupation(temperature)
    diff = 1 - upper_line
    return upper_line + np.exp(-1) * diff


def e_fall(t):
    return np.exp(-1)


def half_fall(t):
    return 0.5


# Fall by 1/2: slow rate
# Fall by upper line occupation: fast rate
# Fall by exp(-1) * upper line: faster than combined rate!
# We need to compare several types of data ie
# Deuterium to see if it is possible to make a consistent `good` choice!
def plot_rate_against_temperature():
    temperatures = np.linspace(100, 255, 5)
    fig, ax = plt.subplots()

    ax.plot(
        1 / temperatures,
        np.log10(
            calculate_lindblad_rate(
                temperatures,
                tunnell_condition=e_fall_to_upper_line_of_two_site_occupation,
            )
        ),
        label="lindblad",
    )
    ax.plot(
        1 / temperatures,
        np.log10(3 * calculate_slow_direction_lindblad_rate(temperatures)),
        label="slow direction",
    )
    ax.plot(
        1 / temperatures,
        np.log10(3 * calculate_fast_direction_lindblad_rate(temperatures)),
        label="fast direction",
    )
    ax.plot(
        1 / temperatures,
        np.log10(3 * calculate_combined_lindblad_rate(temperatures)),
        label="combined direction",
    )
    ax.errorbar(
        1 / experemental_data.temperature,
        np.log10(experemental_data.jumprate),
        yerr=[
            np.log10(experemental_data.absuppererrorvalue / experemental_data.jumprate),
            np.log10(experemental_data.jumprate / experemental_data.abslowererrorvalue),
        ],
        label="experemental data",
    )
    ax.set_title("Plot of Experemental and Theoretical Tunnelling Rate")
    ax.set_ylabel("log(jumprate)")
    ax.set_xlabel("1/Temperature")
    ax.legend()
    # print(3 * calculate_lindblad_rate(150))
    plt.show()


def plot_two_site_lindblad():
    solver = MultipleSiteLindbladSolver(
        times=np.linspace(0, 1 * 10 ** (-8), 2000),
        temperature=150,
        initial_state_grid=np.array([[1], [0]]),
    )
    fig, ax = MultipleSiteLindbladSolverPlotter(
        solver
    ).plot_central_terms_against_time()

    gamma1 = 2 * solver._get_gamma_abcd_omega_ij(1, 0, 1, 0, 1, 0)
    gamma2 = 2 * solver._get_gamma_abcd_omega_ij(0, 1, 0, 1, 0, 1)

    decay_rate = gamma1 + gamma2
    print(decay_rate)
    lower_line = gamma1 / (gamma1 + gamma2)
    ax.axhline(
        y=lower_line,
        linestyle="dashed",
        color="black",
        alpha=0.3,
    )

    upper_line = gamma2 / (gamma1 + gamma2)
    ax.axhline(
        y=upper_line,
        linestyle="dashed",
        color="black",
        alpha=0.3,
    )
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Probablity")
    ax.set_title(
        "Plot of the diagonal elements of the matrix against time \n"
        + r"showing a decay with a characteristic time of $1.6x10^{-9}$s"
    )
    plt.show()


def plot_many_site_lindblad(shape):
    initial_state_grid = np.zeros(shape=shape)
    initial_state_grid[0][0] = 1
    solver = MultipleSiteLindbladSolver(
        times=np.linspace(0, 5 * 10 ** (-8), 2000),
        temperature=150,
        initial_state_grid=initial_state_grid,
        events=lambda t, y: y[0] - np.exp(-1),
    )
    fig, ax = MultipleSiteLindbladSolverPlotter(
        solver
    ).plot_central_terms_against_time()

    gamma1 = 2 * solver._get_gamma_abcd_omega_ij(1, 0, 1, 0, 1, 0)
    gamma2 = 2 * solver._get_gamma_abcd_omega_ij(0, 1, 0, 1, 0, 1)

    decay_rate = gamma1 + gamma2
    print(decay_rate)

    lower_line = 2 * gamma1 / ((gamma1 + gamma2) * solver.number_of_sites)
    ax.axhline(
        y=lower_line,
        linestyle="dashed",
        color="black",
        alpha=0.3,
    )

    upper_line = 2 * gamma2 / ((gamma1 + gamma2) * solver.number_of_sites)
    ax.axhline(
        y=upper_line,
        linestyle="dashed",
        color="black",
        alpha=0.3,
    )

    ax.plot(solver.times, np.exp(-3 * gamma2 * solver.times), label="empty target")
    ax.plot(solver.times, np.exp(-3 * decay_rate * solver.times), label="two states")
    # time_to_fall_by_e = solver.t_events[0][0]
    # ax.plot(
    #     solver.times, np.exp(-solver.times / time_to_fall_by_e), label="e fall time"
    # )

    print(solver.t_events[0][0])
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Probablity")
    ax.set_title(
        "Plot of the diagonal elements of the matrix against time \n"
        + r"showing a decay with a characteristic time of $1.6x10^{-9}$s"
    )
    plt.show()


def plot_many_site_distance_against_time(shape):
    times = np.linspace(0, 5 * 10 ** (-8), 2000)

    initial_state_grid = np.zeros(shape=shape)
    initial_state_grid[0][0] = 1
    solver1 = MultipleSiteLindbladSolver(
        times=times,
        temperature=150,
        initial_state_grid=initial_state_grid,
        events=lambda t, y: y[0] - np.exp(-1),
    )
    fig, ax = plt.subplots(1)
    MultipleSiteLindbladSolverPlotter(solver1).plot_mean_square_distance_against_time(
        ax=ax
    )

    initial_state_grid = np.zeros(shape=(16, 16))
    initial_state_grid[0][0] = 1
    solver2 = MultipleSiteLindbladSolver(
        times=times,
        temperature=150,
        initial_state_grid=initial_state_grid,
        events=lambda t, y: y[0] - np.exp(-1),
    )
    MultipleSiteLindbladSolverPlotter(solver2).plot_mean_square_distance_against_time(
        ax=ax
    )

    diff = np.abs(
        solver1.get_mean_square_distances() - solver2.get_mean_square_distances()
    )
    print(diff)
    popt, _ = scipy.optimize.curve_fit(  # type: ignore
        f=lambda x, a: a * x,
        xdata=times[diff < 10 ** (-2)],
        ydata=solver2.get_mean_square_distances()[diff < 10 ** (-2)],
    )
    # ax.plot(times, popt[0] * times)
    # ax.plot(
    #     times[diff < 10 ** (-2)], solver2.get_mean_square_distances()[diff < 10 ** (-2)]
    # )
    ax.set_xlabel("time /s")
    ax.set_ylabel("mean squared distance")
    ax.set_title(
        "Plot of mean squared distance against time" + "\n showing a linear trend"
    )
    plt.show()


def plot_sink_lindblad():
    initial_state_grid = np.zeros(shape=(2, 1))
    print(initial_state_grid)
    initial_state_grid[0][0] = 1
    solver = MultipleSiteWithSinksLindbladSolver(
        times=np.linspace(0, 5 * 10 ** (-8), 2000),
        temperature=150,
        initial_state_grid=initial_state_grid,
        events=lambda t, y: y[0] - np.exp(-1),
        sink_coords=[
            (1, 0),
        ],
    )
    fig, ax = MultipleSiteLindbladSolverPlotter(
        solver
    ).plot_central_terms_against_time()

    gamma1 = 2 * solver._get_gamma_abcd_omega_ij(1, 0, 1, 0, 1, 0)
    gamma2 = 2 * solver._get_gamma_abcd_omega_ij(0, 1, 0, 1, 0, 1)

    decay_rate = gamma1 + gamma2
    print(decay_rate)

    ax.plot(solver.times, np.exp(-3 * gamma1 * solver.times), label="empty target")
    # ax.plot(solver.times, np.exp(-3 * decay_rate * solver.times), label="two states")
    # time_to_fall_by_e = solver.t_events[0][0]
    # ax.plot(
    #     solver.times, np.exp(-solver.times / time_to_fall_by_e), label="e fall time"
    # )

    ax.set_xlabel("Time / s")
    ax.set_ylabel("Probablity")
    ax.set_title(
        "Plot of the diagonal elements of the matrix against time \n"
        + r"showing a decay with a characteristic time of $1.6x10^{-9}$s"
    )
    plt.show()


def summed_rate_fit(time, rate, gamma1, gamma2):
    return (gamma1 / (gamma1 + gamma2)) * ((gamma2 / gamma1) + np.exp(-rate * time))


def plot_total_occupation_lindblad(shape):
    initial_state_grid = np.zeros(shape=shape)
    initial_state_grid[0][0] = 1
    times = np.linspace(0, 0.5 * 10 ** (-8), 2000)
    solver = MultipleSiteLindbladSolver(
        times=times,
        temperature=150,
        initial_state_grid=initial_state_grid,
        events=lambda t, y: y[0] - np.exp(-1),
    )
    fig, ax = MultipleSiteLindbladSolverPlotter(
        solver
    ).plot_summed_occupation_against_time()

    gamma1 = 2 * solver._get_gamma_abcd_omega_ij(1, 0, 1, 0, 1, 0)
    gamma2 = 2 * solver._get_gamma_abcd_omega_ij(0, 1, 0, 1, 0, 1)

    decay_rate = 3 * (gamma1 + gamma2)
    print(decay_rate)

    lower_line = gamma1 / ((gamma1 + gamma2))
    ax.axhline(
        y=lower_line,
        linestyle="dashed",
        color="black",
        alpha=0.3,
    )

    upper_line = gamma2 / ((gamma1 + gamma2))
    ax.axhline(
        y=upper_line,
        linestyle="dashed",
        color="black",
        alpha=0.3,
    )
    ax.plot(times, rotating_wave_solution1(times, 3 * gamma1, 3 * gamma2))
    ax.plot(times, rotating_wave_solution2(times, 3 * gamma1, 3 * gamma2))

    print(solver.t_events[0][0])
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Probablity")
    ax.set_title(
        "Plot of the diagonal elements of the matrix against time \n"
        + r"showing a decay with a characteristic time of $1.6x10^{-9}$s"
    )
    plt.show()


def plot_occupation_with_hops(shape):
    initial_state_grid = np.zeros(shape=shape)
    initial_state_grid[0][0] = 1
    solver = MultipleSiteWithHopsLindbladSolver(
        times=np.linspace(0, 5 * 10 ** (-8), 2000),
        temperature=150,
        initial_state_grid=initial_state_grid,
        hop_times=np.linspace(0.5 * 10 ** (-8), 4.5 * 10 ** (-8), 8).tolist(),
    )
    fig, ax = MultipleSiteLindbladSolverPlotter(
        solver
    ).plot_central_terms_against_time()

    gamma1 = 2 * solver._get_gamma_abcd_omega_ij(1, 0, 1, 0, 1, 0)
    gamma2 = 2 * solver._get_gamma_abcd_omega_ij(0, 1, 0, 1, 0, 1)

    decay_rate = gamma1 + gamma2
    print(decay_rate)

    lower_line = 2 * gamma1 / ((gamma1 + gamma2) * solver.number_of_sites)
    ax.axhline(
        y=lower_line,
        linestyle="dashed",
        color="black",
        alpha=0.3,
    )

    upper_line = 2 * gamma2 / ((gamma1 + gamma2) * solver.number_of_sites)
    ax.axhline(
        y=upper_line,
        linestyle="dashed",
        color="black",
        alpha=0.3,
    )

    ax.plot(solver.times, np.exp(-3 * gamma2 * solver.times), label="empty target")
    ax.plot(solver.times, np.exp(-3 * decay_rate * solver.times), label="two states")
    # time_to_fall_by_e = solver.t_events[0][0]
    # ax.plot(
    #     solver.times, np.exp(-solver.times / time_to_fall_by_e), label="e fall time"
    # )

    ax.set_xlabel("Time / s")
    ax.set_ylabel("Probablity")
    ax.set_title(
        "Plot of the diagonal elements of the matrix against time \n"
        + r"showing a decay with a characteristic time of $1.6x10^{-9}$s"
    )
    plt.show()


if __name__ == "__main__":
    plot_many_site_distance_against_time(shape=(10, 10))
    # plot_occupation_with_hops(shape=(10, 10))
    # plot_total_occupation_lindblad(shape=(10, 10))
    plot_rate_against_temperature()
    plot_sink_lindblad()
    plot_many_site_lindblad(shape=(10, 10))
    plot_two_site_lindblad()