from matplotlib import pyplot as plt
import numpy as np
from fit_aic.scipy import curve_fit
from lmfit import Model as _Model
from fit_aic.lmfit import Model


def simulate_data():
    x = np.linspace(0, 20, 50)
    y = (
        3 * np.exp(-x / 1.0)
        + 5 * np.exp(-x / 10.0)
        + np.random.normal(0, 0.25, size=x.shape)
    )

    return x, y


def main():
    x, y = simulate_data()

    popt1, pcov1, infodict1, mesg1, ier1 = curve_fit(
        lambda x, A1, tau1: A1 * np.exp(-x / tau1),
        x,
        y,
        p0=[3, 1],
        full_output=True,
    )

    popt2, pcov2, infodict2, mesg2, ier2 = curve_fit(
        lambda x, A1, tau1, A2, tau2: A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2),
        x,
        y,
        p0=[3, 1, 5, 3],
        full_output=True,
    )

    popt3, pcov3, infodict3, mesg3, ier3 = curve_fit(
        lambda x, A1, tau1, A2, tau2, A3, tau3: (
            A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2) + A3 * np.exp(-x / tau3)
        ),
        x,
        y,
        p0=[3, 1, 5, 3, 2, 7],
        full_output=True,
    )

    fig, ax = plt.subplots()

    ax.scatter(x, y, s=10, label="Data")
    ax.plot(
        x,
        popt1[0] * np.exp(-x / popt1[1]),
        label=f"Model1 (AIC={infodict1['aic']:.2f}, AICc={infodict1['aicc']:.2f})",
    )
    ax.plot(
        x,
        popt2[0] * np.exp(-x / popt2[1]) + popt2[2] * np.exp(-x / popt2[3]),
        label=f"Model2 (AIC={infodict2['aic']:.2f}, AICc={infodict2['aicc']:.2f})",
    )
    ax.plot(
        x,
        popt3[0] * np.exp(-x / popt3[1])
        + popt3[2] * np.exp(-x / popt3[3])
        + popt3[4] * np.exp(-x / popt3[5]),
        label=f"Model3 (AIC={infodict3['aic']:.2f}, AICc={infodict3['aicc']:.2f})",
    )
    ax.legend()
    ax.grid()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Model 1 AIC:", infodict1["aic"])
    print("Model 2 AIC:", infodict2["aic"])
    print("Model 3 AIC:", infodict3["aic"])


if __name__ == "__main__":
    main()
