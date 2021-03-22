from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from material_simulation.analysis.rate_temperature_data import rate_temperature_data


class AdjustedRateAgainstTemperaturePlotter:
    def __init__(self, decay_rates: Dict[str, Dict[str, float]]) -> None:
        self.decay_rates = decay_rates

    @staticmethod
    def _initial_occupation_curve(rate, time):
        return 0.5 + 0.5 * np.exp(-time * rate)

    @staticmethod
    def _final_occupation_curve(rate, time):
        return 0.5 - 0.5 * np.exp(-time * rate)

    def plot_adjusted_rates(self):
        fig, ax = plt.subplots(1)
        for i, (key, rates_data) in enumerate(self.decay_rates.items()):
            ax.plot(
                rates_data["temperatures"],
                rates_data["rates"],
                label=key,
            )

        ax.set_xlabel("Temperature / K")
        ax.set_ylabel("Rate Constant")
        ax.set_title(
            "Plot of the Adjusted Rate constant against temperature\n"
            + "given the calculated decay rates at 150K"
        )
        plt.legend()
        plt.show()


if __name__ == "__main__":
    AdjustedRateAgainstTemperaturePlotter(rate_temperature_data).plot_adjusted_rates()
