import numpy as np

if __name__ == "__main__":
    # Works for 14C7, takes about 1 minute
    # about 4 mins to decompose into eigenvectors
    # a = Hamiltonian.create_random_hamiltonian(5000)
    # e = a.get_eigenvalues_and_vectors()

    # a = Hamiltonian.create_random_hamiltonian(5000)
    # print(np.array([1, 2, 3, 4]) == np.array([[1, 2, 3, 4], [1, 2, 3, 4]]))

    # a = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)
    # a = a + np.conj(a.T)
    # print(np.linalg.eig(a))
    # print(np.linalg.eigh(a))
    print([1, 2, 3, 4, 5][:-1])

    np.save("test", [1, 2, 3])
