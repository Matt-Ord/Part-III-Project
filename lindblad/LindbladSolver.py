from abc import ABC, abstractmethod
from typing import Callable, List, Literal, Union
from numpy.linalg import solve
import scipy.integrate
import scipy.constants
from properties.MaterialProperties import NICKEL_MATERIAL_PROPERTIES
import numpy as np
from functools import cached_property
from simulation.Hamiltonian import Hamiltonian

SiteIndex = Union[bool, Literal[0], Literal[1]]


class LindbladSolver(ABC):
    def __init__(
        self,
        times: np.ndarray,
        temperature,
        initial_state,
        events: Union[List[Callable], None] = None,
    ) -> None:
        self.times = times
        self.temperature = temperature
        self.initial_state = initial_state
        self.events = events

    _soln = None
    _max_step = np.inf

    @property
    def t_events(self):
        if self._soln is None:
            self._solve_lindbald_equation()
        return self._soln["t_events"]

    @property
    def y_events(self):
        if self._soln is None:
            self._solve_lindbald_equation()
        return self._soln["y_events"]

    @property
    def fermi_wavevector(self):
        return NICKEL_MATERIAL_PROPERTIES.fermi_wavevector

    @property
    def boltzmaan_energy(self):
        return scipy.constants.Boltzmann * self.temperature

    @staticmethod
    def hydrogen_overlap(a: SiteIndex, b: SiteIndex) -> complex:
        return NICKEL_MATERIAL_PROPERTIES.hydrogen_overlaps[a][b]

    @staticmethod
    def _get_delta_E_ab(a: SiteIndex, b: SiteIndex) -> float:
        return (
            NICKEL_MATERIAL_PROPERTIES.hydrogen_energies[b]
            - NICKEL_MATERIAL_PROPERTIES.hydrogen_energies[a]
        )

    @classmethod
    def _get_omega_ab(cls, a: SiteIndex, b: SiteIndex) -> float:
        return cls._get_delta_E_ab(a, b) / scipy.constants.hbar

    @cached_property
    def gamma_prefactor(self) -> float:
        # Calculation grouped to reduce floating point errors
        a = (scipy.constants.hbar ** 2) / (scipy.constants.elementary_charge ** 2)
        b = (
            self.boltzmaan_energy * scipy.constants.hbar * (self.fermi_wavevector ** 2)
        ) / (scipy.constants.elementary_charge ** 2)
        c = (scipy.constants.epsilon_0 ** 2) / (scipy.constants.electron_mass ** 2)
        d = 32 * np.sqrt(np.pi)
        return a * b * c * d

    def _get_gamma_energy_factor(self, a, b):
        return np.exp(self._get_delta_E_ab(a, b) / (2 * self.boltzmaan_energy))

    def _get_gamma_abcd_omega_ij(
        self,
        a: SiteIndex,
        b: SiteIndex,
        c: SiteIndex,
        d: SiteIndex,
        i: SiteIndex,
        j: SiteIndex,
    ) -> float:
        constant_prefactor = self.gamma_prefactor
        overlap_prefactor = self.hydrogen_overlap(a, b) * self.hydrogen_overlap(d, c)
        energy_factor = self._get_gamma_energy_factor(i, j)
        return constant_prefactor * overlap_prefactor * energy_factor

    @abstractmethod
    def lindbald_derivatives(self, t, p):
        pass

    def _solve_lindbald_equation(self):
        soln = scipy.integrate.solve_ivp(
            fun=self.lindbald_derivatives,
            t_span=(self.times[0], self.times[-1]),
            y0=self.initial_state,
            t_eval=self.times,
            max_step=self._max_step,
            events=self.events,
        )
        self._soln = {
            "t": soln.t,
            "y": soln.y,
            "t_events": soln.t_events,
            "y_events": soln.y_events,
        }

    def save_to_file(self, file):
        if self._soln is None:
            np.savez(
                file,
                times=self.times,
                temperature=self.temperature,
                initial_state=self.initial_state,
                yvals=None,
                allow_pickle=True,
            )
        else:
            np.savez(
                file,
                times=self._soln["t"],
                temperature=self.temperature,
                initial_state=self.initial_state,
                yvals=self._soln["y"],
                allow_pickle=True,
            )

    @classmethod
    def load_from_file(cls, file):
        data = np.load(file, allow_pickle=True)
        temperature = data["temperature"]
        times = data["times"]
        initial_state = data["initial_state"]
        solver = cls(times, temperature, initial_state)
        yvals = data["yvals"]
        if not np.array_equal(yvals, None):  # type: ignore
            solver._soln = {"t": times, "y": yvals}
        return solver


class TwoSiteLindbladSolver(LindbladSolver):
    def __init__(
        self, times, temperature, initial_state=[1, 0, 0, 0], events=None
    ) -> None:
        super().__init__(
            times,
            temperature,
            initial_state=[complex(x) for x in initial_state],
            events=events,
        )

    @property
    def p00_values(self):
        if self._soln is None:
            self._solve_lindbald_equation()
        return self._soln["y"][0]

    @property
    def p01_values(self):
        if self._soln is None:
            self._solve_lindbald_equation()
        return self._soln["y"][1]

    @property
    def p10_values(self):
        if self._soln is None:
            self._solve_lindbald_equation()
        return self._soln["y"][2]

    @property
    def p11_values(self):
        if self._soln is None:
            self._solve_lindbald_equation()
        return self._soln["y"][3]

    @abstractmethod
    def _calculate_derivatie_of_p00(self, t, p00, p01, p10, p11):
        pass

    @abstractmethod
    def _calculate_derivatie_of_p01(self, t, p00, p01, p10, p11):
        pass

    @abstractmethod
    def _calculate_derivatie_of_p10(self, t, p00, p01, p10, p11):
        pass

    @abstractmethod
    def _calculate_derivatie_of_p11(self, t, p00, p01, p10, p11):
        pass

    def lindbald_derivatives(self, t, p):
        p00_derivative = self._calculate_derivatie_of_p00(t, *p)
        p01_derivative = self._calculate_derivatie_of_p01(t, *p)
        p10_derivative = self._calculate_derivatie_of_p10(t, *p)
        p11_derivative = self._calculate_derivatie_of_p11(t, *p)
        return (
            p00_derivative,
            p01_derivative,
            p10_derivative,
            p11_derivative,
        )