from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from material_simulation.analysis.rate_temperature_data import (
    rate_temperature_data_2_electron,
    rate_temperature_data_half_fill,
)
import experemental_data
import scipy.optimize


class AdjustedRateAgainstTemperaturePlotter:
    def __init__(self, decay_rates: Dict[str, Dict[str, np.ndarray]]) -> None:
        self.decay_rates = decay_rates

    @staticmethod
    def _initial_occupation_curve(rate, time):
        return 0.5 + 0.5 * np.exp(-time * rate)

    @staticmethod
    def _final_occupation_curve(rate, time):
        return 0.5 - 0.5 * np.exp(-time * rate)

    def plot_adjusted_rates(
        self,
        title,
    ):
        fig = plt.figure()
        ax = fig.add_subplot()

        for i, (key, rates_data) in enumerate(self.decay_rates.items()):
            ax.plot(
                rates_data["temperatures"],
                rates_data["rates"],
                label=key,
                color=rates_data["color"],
            )

        ax.plot()

        ax.set_ylabel(r"Rate / $s^{-1}$")
        ax.set_xlabel("Temperature / K")

        ax.set_title(title)
        plt.legend()
        plt.show()

    def plot_adjusted_rates_with_theory(self, title):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.errorbar(
            1 / experemental_data.temperature,
            np.log10(experemental_data.jumprate),
            yerr=[
                np.log10(
                    experemental_data.absuppererrorvalue / experemental_data.jumprate
                ),
                np.log10(
                    experemental_data.jumprate / experemental_data.abslowererrorvalue
                ),
            ],
            label="experemental data",
        )

        for i, (key, rates_data) in enumerate(self.decay_rates.items()):
            ax.plot(
                1 / rates_data["temperatures"],
                np.log10(rates_data["rates"]),
                color=rates_data["color"],
                label=key,
            )

        ax.plot()

        ax.set_ylabel(r"log(Rate / $s^{-1}$)")
        ax.set_xlabel("1/Temperature K")

        ax.set_title(title)
        ax.set_xlim(0.003, 0.01)
        ax.set_ylim(9, 10.5)
        plt.legend()
        plt.show()

    def plot_adjusted_rates_with_low_temp_theory(self, title):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.errorbar(
            1 / experemental_data.temperature,
            np.log10(experemental_data.jumprate),
            yerr=[
                np.log10(
                    experemental_data.absuppererrorvalue / experemental_data.jumprate
                ),
                np.log10(
                    experemental_data.jumprate / experemental_data.abslowererrorvalue
                ),
            ],
            label="experemental data",
        )

        for i, (key, rates_data) in enumerate(self.decay_rates.items()):
            low_t_rates_data = rates_data["rates"][
                np.array(
                    [
                        np.abs(rates_data["temperatures"] - t).argmin()
                        for t in experemental_data.temperature[:5]
                    ]
                )
            ]
            print(low_t_rates_data)
            print(experemental_data.jumprate[:5])
            a, b = scipy.optimize.curve_fit(  # type: ignore
                lambda x, a, b: a * x + b,
                1 / experemental_data.temperature[:5],
                np.log10(experemental_data.jumprate[:5] - low_t_rates_data),
            )

            def exited_rate(x, a):
                return 10 ** (a[0] / x + a[1])

            ax.plot(
                1 / rates_data["temperatures"],
                np.log10(
                    exited_rate(rates_data["temperatures"], a) + rates_data["rates"]
                ),
                color=rates_data["color"],
                label=key,
            )

        ax.plot()

        ax.set_ylabel(r"log(Rate / $s^{-1}$)")
        ax.set_xlabel("1/Temperature K")

        ax.set_title(title)
        ax.set_xlim(0.003, 0.01)
        ax.set_ylim(9, 10.5)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    AdjustedRateAgainstTemperaturePlotter(
        rate_temperature_data_half_fill
    ).plot_adjusted_rates_with_low_temp_theory(
        title="Plot of the Rate against temperature\ngiven the half fill decay rate constant"
    )
    AdjustedRateAgainstTemperaturePlotter(
        rate_temperature_data_half_fill
    ).plot_adjusted_rates(
        title="Plot of the Rate against temperature\ngiven the half fill decay rate constant"
    )
    AdjustedRateAgainstTemperaturePlotter(
        rate_temperature_data_half_fill
    ).plot_adjusted_rates_with_theory(
        title="Plot of the Rate against temperature\ngiven the half fill decay rate constant"
    )
    AdjustedRateAgainstTemperaturePlotter(
        rate_temperature_data_2_electron
    ).plot_adjusted_rates(
        title="Plot of the Rate against temperature\ngiven the two electron decay rate constant"
    )
    AdjustedRateAgainstTemperaturePlotter(
        rate_temperature_data_2_electron
    ).plot_adjusted_rates_with_theory(
        title="Plot of the Rate against temperature\ngiven the two electron decay rate constant"
    )
