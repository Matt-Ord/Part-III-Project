from typing import List, Tuple
from LindbladSolver import LindbladSolver
import numpy as np
import scipy.integrate


class MultipleSiteLindbladSolver(LindbladSolver):
    def __init__(
        self, times, temperature, initial_state_grid: np.ndarray, events=None
    ) -> None:
        self.shape = initial_state_grid.shape
        if len(initial_state_grid.shape) != 2:
            raise Exception("Initial grid must be 2D")
        if initial_state_grid.shape[0] % 2 != 0:
            raise Exception(
                "Initial grid must have the same number of fcc and hcp sites"
            )

        super().__init__(times, temperature, initial_state_grid.flatten(), events)

    def p_value_at_index(self, i) -> np.ndarray:
        if self._soln is None:
            self._solve_lindbald_equation()
        return self._soln["y"][i]

    def all_p_values(self) -> np.ndarray:
        if self._soln is None:
            self._solve_lindbald_equation()
        return self._soln["y"]

    def p_value_at_coordinate(self, coord) -> np.ndarray:
        if self._soln is None:
            self._solve_lindbald_equation()
        return self._soln["y"][self.get_index_of_coordinate(coord)]

    def p_value_at_fcc(self):
        if self._soln is None:
            self._solve_lindbald_equation()
        # FCC sites are type 0.
        return self._p_value_at_site(site_type=False)

    def p_value_at_hcp(self):
        if self._soln is None:
            self._solve_lindbald_equation()
        # HCP sites are type 1.
        return self._p_value_at_site(site_type=True)

    def _p_value_at_site(self, site_type: bool):
        site_types = np.array(
            [self._get_type_of_site_i(i) for i in range(self.number_of_sites)]
        )
        return np.sum(self._soln["y"][site_types == site_type], axis=0)

    def get_mean_square_distances(self) -> np.ndarray:
        square_distances = self.get_all_distance_from_origin() ** 2
        return np.sum((self.all_p_values().T * square_distances).T, axis=0)

    def get_all_distance_from_origin(self):
        return self._get_all_distance_from_index(0)

    def _get_all_distance_from_index(self, i: int) -> np.ndarray:
        distances = np.inf * np.ones(shape=self.number_of_sites)
        distances[i] = 0
        distance = 1
        while (np.inf in distances) and distance < self.number_of_sites:
            for index in np.argwhere(distances == (distance - 1)):
                for neighbour in self._get_neighbours_site_i(index):
                    distances[neighbour] = min(distances[neighbour], distance)
            distance += 1
        return distances

    def get_coordinate_of_index(self, i) -> np.ndarray:
        return np.unravel_index(i, self.shape)

    def get_index_of_coordinate(self, coord) -> np.ndarray:
        return np.ravel_multi_index(coord, self.shape, mode="wrap")

    def _get_neighbours_site_i(self, i: int) -> np.ndarray:
        coordinate = self.get_coordinate_of_index(i)
        if self._get_type_of_site_i(i):
            return self.get_index_of_coordinate(
                [
                    [coordinate[0] - 1, coordinate[0] + 1, coordinate[0] + 1],
                    [coordinate[1], coordinate[1], coordinate[1] + 1],
                ]
            )
        return self.get_index_of_coordinate(
            [
                [coordinate[0] + 1, coordinate[0] - 1, coordinate[0] - 1],
                [coordinate[1], coordinate[1], coordinate[1] - 1],
            ]
        )

    # Returns the type of site i. Fcc sites are type 0.
    def _get_type_of_site_i(self, i):
        # fcc sites lie on even rows
        return self.get_coordinate_of_index(i)[0] % 2 != 0

    def _get_all_indicies(self):
        return [0, 1]

    def _get_gamma_factor_site_i(self, i):
        type_i = self._get_type_of_site_i(i)
        not_type_i = not type_i
        return self._get_gamma_abcd_omega_ij(
            not_type_i, type_i, not_type_i, type_i, not_type_i, type_i
        )

    def _calculate_derivative_site_i(self, t, probabilities, i):
        neighbours = self._get_neighbours_site_i(i)
        flux_for_each_neighbour = [
            (
                probabilities[j] * self._get_gamma_factor_site_i(j)
                - probabilities[i] * self._get_gamma_factor_site_i(i)
            )
            for j in neighbours
        ]
        # print(flux_for_each_neighbour)
        return 2 * sum(flux_for_each_neighbour)

    @property
    def number_of_sites(self) -> int:
        return self.initial_state.size

    def lindbald_derivatives(self, t, p: np.ndarray):
        return [
            self._calculate_derivative_site_i(t, p, i)
            for i in range(self.number_of_sites)
        ]


class MultipleSiteWithSinksLindbladSolver(MultipleSiteLindbladSolver):
    def __init__(
        self,
        times,
        temperature,
        initial_state_grid: np.ndarray,
        events,
        sink_coords: List[Tuple[int, int]] = [],
    ) -> None:
        super().__init__(times, temperature, initial_state_grid, events=events)
        self.sink_index = [self.get_index_of_coordinate(coord) for coord in sink_coords]

    def _calculate_derivative_site_i(self, t, probabilities, i):
        if i in self.sink_index:
            return 0
        return super()._calculate_derivative_site_i(t, probabilities, i)


class MultipleSiteWithHopsLindbladSolver(MultipleSiteLindbladSolver):
    def __init__(
        self,
        times: np.ndarray,
        temperature,
        initial_state_grid: np.ndarray,
        events=None,
        hop_times: List[float] = [],
    ) -> None:
        self.hop_times = hop_times
        if not events is None:
            raise Exception("events not yet supported")
        super().__init__(times, temperature, initial_state_grid, events=events)

    def _discritise_state(self, state_probabilities):
        site_index = np.random.choice(
            range(self.number_of_sites), p=state_probabilities.tolist()
        )
        discretised = np.zeros_like(state_probabilities)
        discretised[site_index] = 1
        return discretised

    def _solve_lindbald_equation(self):
        start_times = [self.times[0]] + self.hop_times
        end_times = self.hop_times + [self.times[-1]]
        current_soln = {
            "t": [0],
            "y": [[0] for x in range(self.number_of_sites)],
            "t_events": [],
            "y_events": [],
        }
        initial_state = self.initial_state
        for (start_time, end_time) in zip(start_times, end_times):
            times_in_range = [
                t for t in self.times if (t >= start_time and t < end_time)
            ] + [end_time]
            soln = scipy.integrate.solve_ivp(
                fun=self.lindbald_derivatives,
                t_span=(start_time, end_time),
                y0=initial_state,
                t_eval=times_in_range,
                max_step=self._max_step,
            )
            final_state = soln.y[:, -1]
            # print(soln.t + current_soln["t"][:-1])
            # if len(soln.t) == 1:
            #     current_soln["t"] = soln.t[:]
            # else:
            current_soln["t"] = current_soln["t"][:-1] + soln.t.tolist()
            current_soln["y"] = [
                curr[:-1] + new.tolist()
                for (curr, new) in zip(current_soln["y"], soln.y)
            ]
            initial_state = self._discritise_state(final_state)
        self._soln = current_soln
