from matplotlib import pyplot as plt
import numpy as np
from fit_aic.scipy import curve_fit
from lmfit import Model as _Model
from fit_aic.lmfit import Model


def simulate_data():
    x = np.linspace(0, 20, 30)
    y = (
        3 * np.exp(-x / 1.0)
        + 5 * np.exp(-x / 10.0)
        + np.random.normal(0, 0.25, size=x.shape)
    )

    return x, y


def main():
    x, y = simulate_data()

    popt1, pcov1, infodict1, mesg1, ier1 = curve_fit(
        lambda x, A1, tau1, A2, tau2: A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2),
        x,
        y,
        p0=[3, 1, 5, 3],
        full_output=True,
    )

    popt2, pcov2, infodict2, mesg2, ier2 = curve_fit(
        lambda x, A1, tau1: A1 * np.exp(-x / tau1),
        x,
        y,
        p0=[3, 1],
        full_output=True,
    )

    model1 = Model(
        lambda x, A1, tau1, A2, tau2: A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2)
    )
    result1 = model1.fit(y, x=x, A1=3, tau1=1, A2=5, tau2=3)
    model2 = Model(lambda x, A1, tau1: A1 * np.exp(-x / tau1))
    result2 = model2.fit(y, x=x, A1=3, tau1=1)
    model3 = _Model(lambda x, A1, tau1: A1 * np.exp(-x / tau1))
    result3 = model3.fit(y, x=x, A1=3, tau1=1)

    print(f"Model 1: AIC={result1.aic:.2f})")
    print(f"Model 2: AIC={result2.aic:.2f}")
    print(f"Model 1: AICc={result1.aicc:.2f})")
    print(f"Model 2: AICc={result2.aicc:.2f}")
    print(f"Model 3: AIC={result3.aic:.2f} (lmfit without AICc)")

    fig, ax = plt.subplots()

    ax.scatter(x, y, s=10, label="Data")
    ax.plot(
        x,
        popt1[0] * np.exp(-x / popt1[1]) + popt1[2] * np.exp(-x / popt1[3]),
        label=f"Model 1 (AIC={infodict1['aic']:.2f}, AICc={infodict1['aicc']:.2f})",
    )
    ax.plot(
        x,
        popt2[0] * np.exp(-x / popt2[1]),
        label=f"Model 2 (AIC={infodict2['aic']:.2f}, AICc={infodict2['aicc']:.2f})",
    )
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
