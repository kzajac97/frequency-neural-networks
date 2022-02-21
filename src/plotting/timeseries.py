import numpy as np
from matplotlib import pyplot as plt


def plot_states(t: np.array, states: np.array) -> None:
    """Plots states of ND dynamical system"""
    n_states = states.shape[-1]
    figure, axes = plt.subplots(n_states)

    for axis in range(n_states):
        axes[axis].plot(t, states[:, axis])


def plot_predictive_window(
    inputs: np.array,
    labels: np.array,
    predictions: np.array = None,
    scatter: bool = True,
    dt: float = 0.01,
    n_dim: int = 3,
    marker_size: int = 64,
    plot_kws: dict = None,
):
    """
    Plot single prediction window

    :param inputs: array of model inputs
    :param labels: array of true values, (labels)
    :param predictions: array of predicted values
    :param scatter: if True use scatter plots
    :param dt: time step, used for x axis formatting
    :param n_dim: dimensionality of predicted system
    :param marker_size: size of scatter markers
    :param plot_kws: additional scatter-plot parameters
    """
    if not plot_kws:
        plot_kws = {}

    t = np.arange(0, len(labels) * dt, dt)

    for n in range(n_dim):
        plt.subplot(n_dim, 1, n + 1)

        if scatter:
            warmup_t = np.arange(-len(inputs) * dt, 0, dt)  # time array of input steps
            plt.scatter(warmup_t, inputs[:, n], edgecolors="k", label="Inputs", c="#1338be", s=marker_size, **plot_kws)
            plt.scatter(t, labels[:, n], edgecolors="k", label="Targets", c="#2ca02c", s=marker_size, **plot_kws)
        else:
            plt.plot(np.arange(-len(inputs) * dt, 0, dt), inputs[:, n], label="Inputs", c="#1338be", **plot_kws)
            plt.plot(t, labels[:, n], label="Targets", c="#2ca02c", **plot_kws)

        if predictions is not None:
            if scatter:
                plt.scatter(
                    t, predictions[:, n], marker="X", edgecolors="k",
                    label="Predictions", c="#ff7f0e", s=marker_size, **plot_kws
                )
            else:
                plt.plot(t, predictions[:, n], label="Predictions", c="#ff7f0e", **plot_kws)

    plt.legend()


def plot_simulation_window(
    inputs: np.array,
    labels: np.array,
    predictions: np.array = None,
    scatter: bool = True,
    dt: float = 0.01,
    n_dim: int = 3,
    marker_size: int = 64,
    n_last_prediction: bool = False,
    plot_kws: dict = None,
):
    """
    Plot single prediction window

    :param inputs: array of model inputs
    :param labels: array of true values, (labels)
    :param predictions: array of predicted values
    :param scatter: if True use scatter plots
    :param dt: time step, used for x axis formatting
    :param n_dim: dimensionality of predicted system
    :param marker_size: size of scatter markers
    :param n_last_prediction: if True model and targets are only N last steps
    :param plot_kws: additional scatter-plot parameters
    """
    if not plot_kws:
        plot_kws = {}

    for values in (inputs, labels, predictions):
        if values is not None and not n_last_prediction:
            t = np.arange(0, len(values) * dt, dt)
        else:
            t = np.arange(0, len(inputs) * dt, dt)

    def _draw_array(array: np.array, dim: int, label: str, c: str, kws: dict) -> None:
        """Internal function plotting with constant settings"""
        if not kws:
            kws = {}

        if scatter:
            plt.scatter(t[-len(array):], array[:, dim], edgecolors="k", label=label, c=c, s=marker_size, **kws)
        else:
            plt.plot(t[-len(array):], array[:, dim], label=label, c=c, **kws)

    for n in range(n_dim):
        plt.subplot(n_dim, 1, n + 1)

        if inputs is not None:
            _draw_array(inputs, dim=n, label="Inputs", c="#1338be", kws=plot_kws)
        if labels is not None:
            _draw_array(labels, dim=n, label="Targets", c="#2ca02c", kws=plot_kws)

        if predictions is not None:
            if scatter:
                kwargs = {"marker": "X"}
                kwargs.update(plot_kws)
            else:
                kwargs = plot_kws
            _draw_array(predictions, dim=n, label="Predictions", c="#ff7f0e", kws=kwargs)

    plt.legend()
