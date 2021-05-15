from typing import Any, Callable, List, Type
from MultipleSiteLindbladSolver import (
    MultipleSiteLindbladSolver,
    MultipleSiteWithSinksLindbladSolver,
    MultipleSiteWithHopsLindbladSolver,
)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

from material_simulation.MultiBandMaterialSimulator import T


class MultipleSiteLindbladSolverPlotter:
    def __init__(self, solver: MultipleSiteLindbladSolver) -> None:
        self.solver = solver

    @staticmethod
    def _plot_probabilities_against_time(
        times, propbabilities, ax=None, color=None, label=None
    ):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(times, propbabilities, label=label, color=color)
        return ax

    def plot_central_terms_against_time(self, ax=None, labels=["P0", "P1", "P2"]):
        if ax is None:
            fig, ax = plt.subplots()
        colors = cm.get_cmap("viridis")(np.linspace(0, 1, len(labels)))

        distances = self.solver.get_all_distance_from_origin()
        sorted_distances = distances[np.argsort(distances)]
        sorted_probabilities = self.solver.all_p_values()[np.argsort(distances)]

        for (distance, p_values) in zip(sorted_distances, sorted_probabilities):
            if distance < len(labels):
                self._plot_probabilities_against_time(
                    self.solver.times,
                    p_values,
                    label=labels[int(distance)],
                    ax=ax,
                    color=colors[int(distance)],
                )
                labels[int(distance)] = None
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


def line_of_two_site_occupation(temperature=150) -> float:
    upper_line = upper_line_of_two_site_occupation(temperature)
    diff = 1 - upper_line
    return upper_line + np.exp(-1) * diff


def fallen_by(x: float, terminal=False):
    def fallen_by_event(t: float, y: List[float]):
        return y[0] - x

    fallen_by_event.terminal = terminal
    return fallen_by_event


def neighbouring_hcp_reaches_maximum(
    solver: MultipleSiteLindbladSolver, terminal=False
):
    def reaches_max(t, y):
        neighbour = solver._get_neighbours_site_i(0)[0]
        return solver.lindbald_derivatives(t, y)[neighbour]

    reaches_max.terminal = terminal
    return reaches_max


def neighbouring_fcc_reaches_maximum(
    solver: MultipleSiteLindbladSolver, terminal=False
):
    def reaches_max(t, y):
        if t == 0:
            return 1
        neighbour = int(np.argwhere(solver.get_all_distance_from_origin() == 2)[0])
        return solver.lindbald_derivatives(t, y)[neighbour]

    reaches_max.terminal = terminal
    return reaches_max


def neighbouring_hcp_fallen_by(
    x: float = (1 - np.exp(-1)) * 0.05417107221145403, number_of_fcc=10, terminal=False
):
    def fallen_by_event(t, y):
        return y[number_of_fcc] - x

    fallen_by_event.terminal = terminal
    return fallen_by_event


def neighbouring_fcc_fallen_by(
    x: float = (1 - np.exp(-1)) * 0.0701394400108004, terminal=False
):
    def fallen_by_event(t, y):
        return y[1] - x

    fallen_by_event.terminal = terminal
    return fallen_by_event


def calculate_lindblad_rate(
    temperatures,
    initial_state_grid,
    tunnell_condition: Callable[[float, List[float]], float] = fallen_by(np.exp(-1)),
) -> np.ndarray:
    rates = []
    for t in temperatures:
        solver = MultipleSiteWithSinksLindbladSolver(
            times=np.linspace(0, 5 * 10 ** (-8), 2000),
            temperature=t,
            initial_state_grid=initial_state_grid,
            events=tunnell_condition,
            # sink_coords=[(1, 0), (-1, 0), (-1, -1)],
        )
        print(solver.t_events[0][0])
        rates.append(1 / solver.t_events[0][0])
    return np.array(rates)


def calculate_lindblad_rate_t_dependant_condition(
    temperatures,
    initial_state_grid,
    condition_fn: Callable[[float], float] = lambda x: np.exp(-1),
    tunnell_condition: Callable[
        [float], Callable[[float, List[float]], float]
    ] = lambda x: fallen_by(x, terminal=True),
    factor=1,
) -> np.ndarray:
    rates = []
    for t in temperatures:

        tunnell_condition.terminal = True

        condition = condition_fn(t)

        solver = MultipleSiteWithSinksLindbladSolver(
            times=np.linspace(0, 5 * 10 ** (-8), 2000),
            temperature=t,
            initial_state_grid=initial_state_grid,
            events=tunnell_condition(condition),
            # sink_coords=[(1, 0), (-1, 0), (-1, -1)],
        )
        print(solver.t_events[0][0])
        rates.append(factor / solver.t_events[0][0])
    return np.array(rates)


def calculate_rms_lindblad_rate(
    temperatures,
    initial_state_grid,
) -> np.ndarray:
    rates = []
    for t in temperatures:

        temp_solver = MultipleSiteWithSinksLindbladSolver(
            times=np.linspace(0, 5 * 10 ** (-8), 2000),
            temperature=t,
            initial_state_grid=initial_state_grid,
        )

        def RMS_event(t, y):
            square_distances = temp_solver.get_all_distance_from_origin() ** 2
            return np.sum((y.T * square_distances).T, axis=0) - 0.25

        solver = MultipleSiteWithSinksLindbladSolver(
            times=np.linspace(0, 5 * 10 ** (-8), 2000),
            temperature=t,
            initial_state_grid=initial_state_grid,
            events=RMS_event,
        )
        print(solver.t_events[0][0])
        rates.append(1 / solver.t_events[0][0])
    return np.array(rates)


# Fall by 1/2: slow rate
# Fall by upper line occupation: fast rate
# Fall by exp(-1) * upper line: faster than combined rate!
# We need to compare several types of data ie
# Deuterium to see if it is possible to make a consistent `good` choice!
def plot_rate_against_temperature(shape=(10, 10), q_dependant_rate_correction=1):
    temperatures = np.linspace(100, 255, 10)
    initial_state_grid = np.zeros(shape=shape)
    initial_state_grid[0][0] = 1
    fig, ax = plt.subplots()
    ax.errorbar(
        1 / experemental_data.temperature,
        np.log10(experemental_data.jumprate),
        yerr=[
            np.log10(experemental_data.absuppererrorvalue / experemental_data.jumprate),
            np.log10(experemental_data.jumprate / experemental_data.abslowererrorvalue),
        ],
        label="experemental data",
    )

    def exited_rate(x, a):
        return 10 ** (a[0] / x + a[1])

    # ax.plot(
    #     1 / experemental_data.temperature,
    #     a[0] / experemental_data.temperature + a[1],
    # )

    print("Initial FCC")
    # ax.plot(
    #     1 / temperatures,
    #     np.log10(
    #         calculate_lindblad_rate(
    #             temperatures,
    #             initial_state_grid,
    #             tunnell_condition=fallen_by(np.exp(-1), terminal=True),
    #         )
    #     ),
    #     label="Initial FCC",
    # )
    print("Initial FCC (boltzmann)")
    a, b = scipy.optimize.curve_fit(  # type: ignore
        lambda x, a, b: a * x + b,
        1 / experemental_data.temperature[:10],
        np.log10(
            experemental_data.jumprate[:10]
            - q_dependant_rate_correction
            * calculate_lindblad_rate(
                experemental_data.temperature[:10],
                initial_state_grid,
                tunnell_condition=fallen_by(
                    line_of_two_site_occupation(), terminal=True
                ),
            )
        ),
    )
    ax.plot(
        1 / temperatures,
        np.log10(
            exited_rate(temperatures, a)
            + q_dependant_rate_correction
            * calculate_lindblad_rate(
                temperatures,
                initial_state_grid,
                tunnell_condition=fallen_by(
                    line_of_two_site_occupation(), terminal=True
                ),
            )
        ),
        label="Initial FCC (boltzmann)",
    )

    def next_HCP_condition_fn(t):
        temp_solver = MultipleSiteWithSinksLindbladSolver(
            times=np.linspace(0, 5 * 10 ** (-8), 2000),
            temperature=t,
            initial_state_grid=initial_state_grid,
        )
        solver = MultipleSiteWithSinksLindbladSolver(
            times=np.linspace(0, 5 * 10 ** (-8), 2000),
            temperature=t,
            initial_state_grid=initial_state_grid,
            events=neighbouring_hcp_reaches_maximum(temp_solver, terminal=True),
        )
        return (1 - np.exp(-1)) * solver.y_events[0][0][shape[0]]

    print("Next HCP")
    a, b = scipy.optimize.curve_fit(  # type: ignore
        lambda x, a, b: a * x + b,
        1 / experemental_data.temperature[:10],
        np.log10(
            experemental_data.jumprate[:10]
            - q_dependant_rate_correction
            * calculate_lindblad_rate_t_dependant_condition(
                experemental_data.temperature[:10],
                initial_state_grid,
                condition_fn=next_HCP_condition_fn,
                tunnell_condition=lambda x: neighbouring_hcp_fallen_by(
                    x, shape[0], terminal=True
                ),
            )
        ),
    )
    ax.plot(
        1 / temperatures,
        +np.log10(
            exited_rate(temperatures, a)
            + q_dependant_rate_correction
            * calculate_lindblad_rate_t_dependant_condition(
                temperatures,
                initial_state_grid,
                condition_fn=next_HCP_condition_fn,
                tunnell_condition=lambda x: neighbouring_hcp_fallen_by(
                    x, shape[0], terminal=True
                ),
            )
        ),
        label="Next HCP",
    )

    def next_FCC_condition_fn(t):
        temp_solver = MultipleSiteWithSinksLindbladSolver(
            times=np.linspace(0, 5 * 10 ** (-8), 2000),
            temperature=t,
            initial_state_grid=initial_state_grid,
        )
        solver = MultipleSiteWithSinksLindbladSolver(
            times=np.linspace(0, 5 * 10 ** (-8), 2000),
            temperature=t,
            initial_state_grid=initial_state_grid,
            events=neighbouring_fcc_reaches_maximum(temp_solver, terminal=True),
        )
        return (1 - np.exp(-1)) * solver.y_events[0][0][1]

    print("Next FCC")
    # ax.plot(
    #     1 / temperatures,
    #     np.log10(
    #         calculate_lindblad_rate_t_dependant_condition(
    #             temperatures,
    #             initial_state_grid,
    #             condition_fn=next_FCC_condition_fn,
    #             tunnell_condition=lambda x: neighbouring_fcc_fallen_by(
    #                 x, terminal=True
    #             ),
    #             factor=4,
    #         )
    #     ),
    #     label="Next FCC",
    # )
    print("Combined Occupation")
    a, b = scipy.optimize.curve_fit(  # type: ignore
        lambda x, a, b: a * x + b,
        1 / experemental_data.temperature[:10],
        np.log10(
            experemental_data.jumprate[:10]
            - 3
            * q_dependant_rate_correction
            * calculate_combined_lindblad_rate(experemental_data.temperature[:10])
        ),
    )
    ax.plot(
        1 / temperatures,
        np.log10(
            exited_rate(temperatures, a)
            + 3
            * q_dependant_rate_correction
            * calculate_combined_lindblad_rate(temperatures)
        ),
        label="Combined Occupation",
    )

    print("RMS")
    a, b = scipy.optimize.curve_fit(  # type: ignore
        lambda x, a, b: a * x + b,
        1 / experemental_data.temperature[:10],
        np.log10(
            experemental_data.jumprate[:10]
            - q_dependant_rate_correction
            * calculate_rms_lindblad_rate(
                experemental_data.temperature[:10], initial_state_grid
            )
        ),
    )
    ax.plot(
        1 / temperatures,
        np.log10(
            exited_rate(temperatures, a)
            + q_dependant_rate_correction
            * calculate_rms_lindblad_rate(temperatures, initial_state_grid)
        ),
        label="RMS Distance",
    )

    ax.set_title(
        "Plot of Experemental and Theoretical Tunnelling Rate\nAgainst Temperature"
    )
    ax.set_ylabel(r"log(Rate /$s^{-1}$)")
    ax.set_xlabel("1/Temperature K")
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
    events = [
        fallen_by(np.exp(-1)),
        fallen_by(line_of_two_site_occupation()),
        neighbouring_hcp_fallen_by(),
        neighbouring_fcc_fallen_by(),
    ]
    solver = MultipleSiteLindbladSolver(
        times=np.linspace(0, 1.5 * 10 ** (-8), 2000),
        temperature=150,
        initial_state_grid=initial_state_grid,
        events=events,
    )
    print()
    fig, ax = MultipleSiteLindbladSolverPlotter(solver).plot_central_terms_against_time(
        labels=["0 hop", "1 hop", "2 hop", "3 hop"]
    )

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

    # ax.plot(solver.times, np.exp(-3 * gamma2 * solver.times), label="empty target")
    # ax.plot(solver.times, np.exp(-3 * decay_rate * solver.times), label="two states")
    # time_to_fall_by_e = solver.t_events[0][0]
    # ax.plot(
    #     solver.times, np.exp(-solver.times / time_to_fall_by_e), label="e fall time"
    # )

    print(solver.t_events)
    ax.set_xlabel("Time / s")
    ax.set_ylabel("Probablity")
    ax.set_title(
        "Plot of the occupation probabilities against time \n"
        + "showing the diffusion of the hydrogen atom"
    )
    plt.show()


def plot_many_site_distance_against_time(shape):
    times = np.linspace(0, 5 * 10 ** (-8), 2000)
    # times = np.concatenate([[0], np.linspace(4e-10, 5.2e-10, 4000)])

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
        ax=ax, label="(10, 10) grid"
    )

    initial_state_grid = np.zeros(shape=(20, 20))
    initial_state_grid[0][0] = 1
    solver2 = MultipleSiteLindbladSolver(
        times=times,
        temperature=150,
        initial_state_grid=initial_state_grid,
        events=lambda t, y: y[0] - np.exp(-1),
    )
    MultipleSiteLindbladSolverPlotter(solver2).plot_mean_square_distance_against_time(
        ax=ax, label="(20, 20) grid"
    )
    has_tunnelled = solver2.times[solver2.get_mean_square_distances() > 0.25]
    print(has_tunnelled[0])

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
    ax.legend()
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
    times = np.linspace(0, 0.3 * 10 ** (-8), 2000)
    solver = MultipleSiteLindbladSolver(
        times=times,
        temperature=150,
        initial_state_grid=initial_state_grid,
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

    ax.set_xlabel("Time / s")
    ax.set_ylabel("Probablity")
    ax.set_title(
        "Plot of the total fcc and HCP occupation against time \n"
        + r"showing a decay with a characteristic time of $5.3x10^{-10}$s"
    )
    ax.legend()
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
    # plot_many_site_distance_against_time(shape=(10, 10))
    # plot_occupation_with_hops(shape=(10, 10))
    # plot_total_occupation_lindblad(shape=(10, 10))
    plot_rate_against_temperature(q_dependant_rate_correction=1)
    # plot_sink_lindblad()
    # plot_many_site_lindblad(shape=(10, 10))
    # plot_two_site_lindblad()
