import numpy as np
import scipy.constants, scipy.integrate
from properties.MaterialProperties import NICKEL_MATERIAL_PROPERTIES
import matplotlib.pyplot as plt


def get_delta_e(k_1, k_2):
    E1 = (scipy.constants.hbar * k_1) ** 2 / (2 * scipy.constants.m_e)
    E2 = (scipy.constants.hbar * k_2) ** 2 / (2 * scipy.constants.m_e)
    return E1 - E2


def get_fermi_occupation(k, k_f, boltzmann_energy):
    return 1 / (1 + np.exp((get_delta_e(k, k_f)) / boltzmann_energy))


def actual_integrand(k_1, omega, k_f, boltzmann_energy):
    k_3 = np.sqrt(
        k_1 ** 2 - 2 * scipy.constants.m_e * omega / (scipy.constants.hbar ** 2)
    )

    return (
        (k_1 ** 2)
        * (k_3 ** 2)
        * get_fermi_occupation(k_1, k_f, boltzmann_energy)
        * (1 - get_fermi_occupation(k_3, k_f, boltzmann_energy))
        / (
            np.sqrt(
                k_1 ** 2 - 2 * scipy.constants.m_e * omega / (scipy.constants.hbar ** 2)
            )
        )
    )


def approximate_integrand(k_1, omega, k_f, boltzmann_energy):
    return (
        (k_f ** 3)
        * 0.25
        * np.exp(-((get_delta_e(k_1, k_f) / (2 * boltzmann_energy)) ** 2))
        * np.exp(-omega / (2 * boltzmann_energy))
    )


k_f = NICKEL_MATERIAL_PROPERTIES.fermi_wavevector
boltzmann_energy = scipy.constants.Boltzmann * 150
omega = (
    NICKEL_MATERIAL_PROPERTIES.hydrogen_energies[0]
    - NICKEL_MATERIAL_PROPERTIES.hydrogen_energies[1]
)
# omega = 0
d_k = 2 * boltzmann_energy * scipy.constants.m_e / (scipy.constants.hbar ** 2 * k_f)

k_points = np.linspace(k_f - 2 * d_k, k_f + 2 * d_k, 1000)

fig, ax = plt.subplots(1)
ax.plot(
    k_points, actual_integrand(k_points, omega, k_f, boltzmann_energy), label="actual"
)
ax.plot(
    k_points,
    approximate_integrand(k_points, omega, k_f, boltzmann_energy),
    label="approximate",
)
ax.set_title(
    "Plot of actual integrand against k,\n" + r"with the real value of $\omega{}$"
)
ax.legend()
ax.set_xlabel("Wavevector $m^{-1}$")
plt.show()

fig, ax = plt.subplots(1)
ax.plot(k_points, actual_integrand(k_points, 0, k_f, boltzmann_energy), label="actual")
ax.plot(
    k_points,
    approximate_integrand(k_points, 0, k_f, boltzmann_energy),
    label="approximate",
)
ax.set_title("Plot of actual integrand against k\n" + r"with $\omega{}=0$ ")
ax.legend()
ax.set_xlabel("Wavevector $m^{-1}$")
plt.show()

actual = scipy.integrate.quad(
    lambda k: actual_integrand(k, omega, k_f, boltzmann_energy),
    k_f - 20 * d_k,
    k_f + 20 * d_k,
)[0]
print(actual)

approximate = scipy.integrate.quad(
    lambda k: approximate_integrand(k, omega, k_f, boltzmann_energy),
    k_f - 20 * d_k,
    k_f + 20 * d_k,
)[0]
print(approximate)