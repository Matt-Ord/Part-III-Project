from typing import Type
import numpy as np
import matplotlib.pyplot as plt
from RedfieldSolver import RedfieldSolver


class RedFieldSolverPlotter:
    def __init__(self, solver: RedfieldSolver) -> None:
        self.solver = solver

    @staticmethod
    def _plot_probabilities_against_time(times, propbabilities, ax=None, label=None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(times, np.abs(propbabilities), label=label)
        return ax

    def plot_diagonal_terms_against_time(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        self._plot_probabilities_against_time(
            self.solver.times, self.solver.p00_values, label="P00", ax=ax
        )
        # self._plot_probabilities_against_time(
        #     self.solver.times, self.solver.p11_values, label="P11", ax=ax
        # )
        # ax.set_ylim([0, 1])
        ax.set_xlim([self.solver.times[0], None])
        ax.set_title("Plot of the diagonal terms of the density matrix against time")
        ax.legend()
        return fig, ax

    def plot_total_diagonal_terms(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        self._plot_probabilities_against_time(
            self.solver.times,
            self.solver.p00_values + self.solver.p11_values,
            label="P00",
            ax=ax,
        )
        # self._plot_probabilities_against_time(
        #     self.solver.times, self.solver.p11_values, label="P11", ax=ax
        # )
        # ax.set_ylim([0, 1])
        ax.set_xlim([self.solver.times[0], None])
        ax.set_title(
            "Plot of the sum of diagonal terms of the density matrix against time"
        )
        ax.legend()
        return fig, ax

    def plot_final_state_density_against_time(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        self._plot_probabilities_against_time(
            self.solver.times, self.solver.p11_values, label="P11", ax=ax
        )
        ax.set_xlim([self.solver.times[0], None])
        ax.set_title("Plot of the final state density against time")
        ax.legend()
        return fig, ax

    def plot_cross_terms_against_time(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        self._plot_probabilities_against_time(
            self.solver.times, self.solver.p01_values, label="P01", ax=ax
        )
        self._plot_probabilities_against_time(
            self.solver.times, self.solver.p10_values, label="P10", ax=ax
        )
        ax.set_xlim([self.solver.times[0], None])
        ax.set_title("Plot of the cross terms of the density matrix against time")
        ax.legend()
        return fig, ax

    def plot_solution(self):
        self.plot_diagonal_terms_against_time()
        self.plot_total_diagonal_terms()
        self.plot_cross_terms_against_time()
        self.plot_final_state_density_against_time()
        plt.show()
        return self

    def print_final_density(self):
        print(self.solver.p00_values)
        print(
            self.solver.p00_values[-1],
            self.solver.p01_values[-1],
            self.solver.p10_values[-1],
            self.solver.p11_values[-1],
        )
        return self

    @classmethod
    def from_file(cls, solver_type: Type[RedfieldSolver], file):
        return cls(solver_type.load_from_file(file))


if __name__ == "__main__":

    solver = RedfieldSolver(
        times=np.linspace(0, 4 * 10 ** (-10), 1000),
        temperature=150,
        initial_state=[1, 0, 0, 0],
    )
    solver._solve_redfield_equation()
    solver.save_to_file("solvers/redfield1.npz")
    RedFieldSolverPlotter.from_file(
        RedfieldSolver, "solvers/redfield1.npz"
    ).plot_solution()
