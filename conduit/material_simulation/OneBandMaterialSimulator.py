from material_simulation.MultiBandMaterialSimulator import (
    MultiBandMaterialSimulator,
    MultiBandNickelMaterialSimulatorUtil,
)
from properties.MaterialProperties import MaterialProperties
import numpy as np

# Simulates the electron sysytem using a single
# Closely packed band, and uses the theroetical
# electron occupation of the lower band to come to
# an approximate tunnelling rate


class OneBandMaterialSimulator(MultiBandMaterialSimulator):
    def __init__(
        self,
        material_properties: MaterialProperties,
        temperature: float,
        number_of_states_per_band: int,
        bandwidth: float,
        number_of_electrons: int,
    ) -> None:
        self.number_of_electrons = number_of_electrons
        super().__init__(
            material_properties, temperature, number_of_states_per_band, bandwidth
        )

    @property
    def hydrogen_energies_for_simulation(self):
        return [0, 0]

    def _generate_electron_energies(self):
        return self._get_band_energies()

    def _get_fraction_of_occupation(self):
        return self.number_of_electrons / self.number_of_states_per_band

    # Equal to beta (E - Ef)
    def _calculate_average_factor_from_fermi_energy(self):
        return np.log((1 / self._get_fraction_of_occupation()) - 1)

    # Equal to (E - Ef)
    def _calculate_average_energy_from_fermi_energy(self):
        return (
            self.boltzmann_energy * self._calculate_average_factor_from_fermi_energy()
        )

    def _calculate_average_state_energy(self):
        return (
            self._get_fermi_energy()
            + self._calculate_average_energy_from_fermi_energy()
        )

    def calculate_occupation_of_lower_band(self):
        # beta * egap
        energy_gap_exponent_factor = (
            self.hydrogen_energy_difference / self.boltzmann_energy
        )
        exponent_factor = (
            self._calculate_average_factor_from_fermi_energy()
            - energy_gap_exponent_factor
        )
        return 1 / (1 + np.exp(exponent_factor))

    def calculate_effective_tunnelling_rate(self, tuneling_time):

        tunnelling_rate = 1 / tuneling_time
        return self.calculate_occupation_of_lower_band() * tunnelling_rate

    def _get_energy_jitter(self):
        return 0.5 * self._get_energy_spacing()


if __name__ == "__main__":
    nickel_sim = MultiBandNickelMaterialSimulatorUtil.create(
        OneBandMaterialSimulator,
        temperature=150,
        number_of_states_per_band=15,
        target_frequency=10 ** (9),
        number_of_electrons=2,
    )

    nickel_sim.simulate_material(times=np.linspace(0, 0.001 * 10 ** -12, 1000))

    # nickel_sim.simulate_average_material(
    #     times=np.linspace(0, 3 * 10 ** -10, 500), average_over=20
    # )
    nickel_sim.simulate_average_material(
        times=np.linspace(0, 4 * 10 ** -3, 500), average_over=40, jitter_electrons=True
    )
