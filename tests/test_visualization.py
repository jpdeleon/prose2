import matplotlib.pyplot as plt
import numpy as np

from prose import visualization


def test_plot_expected_transit_with_depth_draws_model():
    time = np.linspace(0, 2, 20)
    fig, ax = plt.subplots()
    plt.sca(ax)

    visualization.plot_expected_transit(
        time, epoch=1.0, period=1.0, duration=0.2, depth=0.01
    )

    assert len(ax.lines) == 3
    model_y = ax.lines[-1].get_ydata()
    assert np.nanmin(model_y) == 0.99
    assert np.nanmax(model_y) == 1.0
    plt.close(fig)
