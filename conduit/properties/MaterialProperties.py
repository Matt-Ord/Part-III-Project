from typing import List, NamedTuple


class MaterialProperties(NamedTuple):
    # Material properties as in SI units
    fermi_wavevector: float
    hydrogen_energies: List[float]
    hydrogen_overlaps: List[List[complex]]


NICKEL_FCC_ENERGY_EV = 0.1668191  # eV
NICKEL_HCP_ENERGY_EV = 0.1794517  # eV

NICKEL_ENERGY_OFFSET_J = 1.522 * 10 ** (-21)

NICKEL_MATERIAL_PROPERTIES = MaterialProperties(
    fermi_wavevector=1.77 * 10 ** (10),  # m^-1
    hydrogen_energies=[-NICKEL_ENERGY_OFFSET_J, NICKEL_ENERGY_OFFSET_J],
    hydrogen_overlaps=[[1, 0.0044], [0.0044, 1]],
)
