from Hamiltonian import Hamiltonian
import numpy as np
import matplotlib.pyplot as plt


class ElectronSimulator():

    @classmethod
    def get_electron_densities(cls, probabilities, basis_states):
        return np.array(probabilities).dot(basis_states)

    @classmethod
    def simulate_electron_system(cls, k_energies, times):
        number_of_electrons = len(k_energies) / 2
        electron_permutations = cls.generate_permutations(
            len(k_energies), number_of_electrons)

        hamiltonian = cls.create_hamiltonian_for_k_energies(
            k_energies, electron_permutations)

        hamiltonian += Hamiltonian(
            np.ones((len(electron_permutations), len(electron_permutations))))

        initial_state = np.array(
            [0 for x in range(len(electron_permutations))])
        initial_state[-1] = 1
        states = [hamiltonian.evolve_state(initial_state, t) for t in times]

        densities = [cls.get_electron_densities(
            np.abs(state) ** 2, electron_permutations) for state in states]

        fig, ax = plt.subplots()
        ax.set_prop_cycle('color', plt.cm.Spectral(
            np.linspace(0, 1, len(densities))))
        for density in densities:
            ax.plot(np.abs(density))
        ax.set_xlabel('Electron Density')
        ax.set_ylabel('k_vector')
        ax.set_title('Plot of electron density against k')
        plt.show()

        fig, ax = plt.subplots()
        ax.plot(np.average(np.abs(densities), axis=0))
        ax.set_xlabel('Electron Density')
        ax.set_ylabel('k_vector')
        ax.set_title('Plot of Average electron density against k')
        plt.show()

    @classmethod
    def create_hamiltonian_for_k_energies(cls,
                                          k_energies,
                                          electron_permutations):

        energies = [
            cls.get_energy_of_permutation(
                perm,
                k_energies)
            for perm in electron_permutations]
        return Hamiltonian.create_diagonal_hamiltonian(energies)

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


if __name__ == '__main__':

    ElectronSimulator.simulate_electron_system(
        [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 14*14],
        np.linspace(250, 500, 50))
