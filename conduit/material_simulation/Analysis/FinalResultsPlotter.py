from typing import Dict
import matplotlib.pyplot as plt
import numpy as np


class FinalResultsPlotter:
    def __init__(self, decay_rates: Dict[str, Dict[str, float]]) -> None:
        self.decay_rates = decay_rates

    @staticmethod
    def _initial_occupation_curve(rate, time):
        return 0.5 + 0.5 * np.exp(-time * rate)

    @staticmethod
    def _final_occupation_curve(rate, time):
        return 0.5 - 0.5 * np.exp(-time * rate)

    def plot_predicted_tunnelling_curve(self, times):
        fig, ax = plt.subplots(1)

        for i, (key, rates) in enumerate(self.decay_rates.items()):
            (line,) = ax.plot(
                times,
                self._initial_occupation_curve(rates["adjusted"], times),
                label=key,
            )
            ax.plot(
                times,
                self._final_occupation_curve(rates["adjusted"], times),
                color=line.get_color(),
            )
            ax.plot(
                times,
                self._initial_occupation_curve(rates["measured"], times),
                color=line.get_color(),
                alpha=0.3,
            )
            ax.plot(
                times,
                self._final_occupation_curve(rates["measured"], times),
                color=line.get_color(),
                alpha=0.3,
            )

        ax.set_xlabel("time / s")
        ax.set_ylabel("occupation fraction")
        ax.set_title(
            "Plot of the predicted decay rate behaviour\n"
            + "given the calculated decay rates at 150K"
        )
        plt.legend()
        plt.show()


decay_rates_150K = {
    "auto data log fit cosh": {
        "measured": 17712703860.0,
        "adjusted": 20301231441,
    },
    "manual data log fit cosh": {
        "measured": 10569663508,
        "adjusted": 12087970414,
    },
    "auto data cosh": {
        "measured": 19986418992,
        "adjusted": 22909917915,
    },
    "manual data cosh": {
        "measured": 11367063687,
        "adjusted": 13012572061,
    },
}
decay_rates_120K = {
    "auto data cosh": {
        "measured": 12314098562,
        "adjusted": 12036702533,
    },
}
decay_rates_150K_fixed = {
    "manual quadratic data": {
        "measured": 2 * 3 * 310831069.0,
        "adjusted": 2 * 3 * 364875153,
    },
}
if __name__ == "__main__":
    FinalResultsPlotter(
        decay_rates=decay_rates_150K_fixed
    ).plot_predicted_tunnelling_curve(np.linspace(0, 1 * 10 ** (-9), 1000))
