import numpy as np
import matplotlib.pyplot as plt
from LindbaldSolver import LindbaldSolver
from NoEnergyGapLindbaldSolver import NoEnergyGapLindbaldSolver


class LindbaldSolverPlotter:
    def __init__(self, solver: LindbaldSolver) -> None:
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
        self._plot_probabilities_against_time(
            self.solver.times, self.solver.p11_values, label="P11", ax=ax
        )
        ax.set_ylim([0, 1])
        ax.set_xlim([self.solver.times[0], None])
        ax.set_title("Plot of the diagonal terms of the density matrix against time")
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
        self.plot_cross_terms_against_time()
        self.plot_final_state_density_against_time()
        plt.show()
        return self

    def print_final_density(self):
        print(
            self.solver.p00_values[-1],
            self.solver.p01_values[-1],
            self.solver.p10_values[-1],
            self.solver.p11_values[-1],
        )
        return self

    @classmethod
    def from_file(cls, solver_type: LindbaldSolver, file):
        return cls(solver_type.load_from_file(file))


if __name__ == "__main__":
    # LindbaldSolverPlotter(
    #     RotatingWaveLindbaldSolver(
    #         times=np.linspace(0, 5 * 10 ** (-5), 100), temperature=150
    #     )
    # ).plot_solution()

    solver = NoEnergyGapLindbaldSolver(
        times=np.linspace(0, 2 * 10 ** (-5), 20000),
        temperature=150,
        initial_state=[1 + 0j, 0, 0, 0],
    )
    solver._solve_lindbald_equation()
    solver.save_to_file("solvers/no_gap_solver.npz")
    LindbaldSolverPlotter.from_file(
        NoEnergyGapLindbaldSolver, "solvers/no_gap_solver.npz"
    ).print_final_density().plot_solution()
