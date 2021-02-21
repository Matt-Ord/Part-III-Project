from numpy.core.fromnumeric import shape
from Hamiltonian import Hamiltonian
import numpy as np

if __name__ == "__main__":
    # Works for 14C7, takes about 1 minute
    # about 4 mins to decompose into eigenvectors
    # a = Hamiltonian.create_random_hamiltonian(5000)
    # e = a.get_eigenvalues_and_vectors()
    # print("done")

    a = Hamiltonian.create_random_hamiltonian(5000)
    print(a.evolve_state(np.random.rand(5000), 1))
    # print(np.matmul([[2, 3], [4, 5], [6, 7]], [1, 2]))
