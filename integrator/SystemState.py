import numpy as np


class SystemState():

    def __init__(self,
                 states: np.ndarray,
                 wavefunction: np.ndarray,
                 energies: np.ndarray,
                 interaction_hamiltonian: np.ndarray):
        self.wavefunction = np.array(wavefunction)
        self.energies = np.array(energies)
        self.states = np.array(states)
        if not self.wavefunction.shape == \
                self.energies.shape == \
                self.states.shape:
            raise Exception('state data has incorect dimensions')
        self._interaction_hamiltonian = interaction_hamiltonian

    @property
    def interaction_hamiltonian(self):
        return self._interaction_hamiltonian

    @interaction_hamiltonian.setter
    def interaction_hamiltonian(self, value):
        if not value.shape == self._interaction_hamiltonian.shape:
            raise Exception("Hamiltonian has incorrect shape")
        self._interaction_hamiltonian = value

    def get_occupation(self, state: int) -> float:
        return self.wavefunction[self._get_wavefunction_position(state)]

    def progress_time(self,
                      dt: float):

        change_in_wavefunction = np.exp(-1j * self.energies) \
            * np.matmul(self.interaction_hamiltonian,
                        self.wavefunction *
                        np.exp(1j * self.energies))
        self.wavefunction += change_in_wavefunction
        self.wavefunction /= np.linalg.norm(self.wavefunction)
        return self

    def _get_wavefunction_position(self, state: int) -> int:
        return self.states.index_of(state)

    def __str__(self):
        return "Wavefunction " + self.wavefunction.__str__()

    @classmethod
    def create_new_test_state(cls, number_of_electrons):
        states = np.array([3, 5, 6, 9, 10])
        initial_wavefunction = np.zeros_like(states, dtype='complex128')
        initial_wavefunction[0] = 1
        energies = np.random.rand(*states.shape)
        interaction_hamiltonian = np.ones(
            shape=(states.shape[0], states.shape[0]))

        return cls(states,
                   initial_wavefunction,
                   energies,
                   interaction_hamiltonian)

    @classmethod
    def generate_permutations(cls, number_of_states, number_of_electrons):
        if (number_of_states < number_of_electrons or number_of_electrons < 0):
            return []
        if number_of_states == 0:
            return [[]]
        return ([[0] + x for x in
                 cls.generate_permutations(number_of_states - 1,
                                           number_of_electrons)]
                + [[1] + x for x in
                   cls.generate_permutations(number_of_states - 1,
                                             number_of_electrons - 1)]
                )

    @classmethod
    def get_energy_of_permutation(cls, permutation, energies):
        return sum(a*b for a, b in zip(permutation, energies))

    @classmethod
    def create_new_half_filled_state(cls, k_values):
        m_e = 1
        k_energies = [k ** 2 / (2 * m_e) for k in k_values]
        number_of_electrons = int(len(k_values) / 2)
        electron_permutations = cls.generate_permutations(
            number_of_electrons*2,
            number_of_electrons)

        energies = [
            cls.get_energy_of_permutation(
                perm,
                k_energies)
            for perm in electron_permutations]

        states = np.array([int("".join(map(str, perm)), base=2)
                           for perm in electron_permutations])

        initial_wavefunction = np.zeros_like(states, dtype='complex128')
        initial_wavefunction[0] = 1

        interaction_hamiltonian = 0.0001 * np.ones(
            shape=(states.shape[0], states.shape[0]))

        return cls(states,
                   initial_wavefunction,
                   energies,
                   interaction_hamiltonian)


if __name__ == "__main__":
    # a = np.array([1, 2])
    # b = np.array([[1, 2], [3, 4]])
    # print(np.matmul(b, a))

    # print(a * np.matmul(b, a))

    # a = SystemState.create_new_state(0)
    # print(a)
    # a.progress_time(1)
    # print(a)
    a = SystemState.create_new_half_filled_state(
        np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]))

    dela_t = 0.0001
    number_of_steps = 1000
    dt = dela_t / number_of_steps
    print(a)
    for x in range(number_of_steps):
        a.progress_time(dt)
    print('done', a)
