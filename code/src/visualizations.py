import os
import pickle
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


# Configure color palletes
plt.style.use("default")


def load_network_snapshots(
        years: Iterable,
        months: Iterable,
        dir: str = "../data/02_preprocessed",
) -> dict:
    """Load preprocessed user-user interaction network snapshots."""
    snapshots = {}
    for year in years:
        for month in months:
            period = f"{year}#{month}"
            try:
                filename = f"user_network_month_{period}.pkl"
                filepath = os.path.join(dir, filename)
                if os.path.exists(filepath):
                    with open(filepath, "rb") as fp:
                        snapshots[period] = pickle.load(fp)
                    print(f"Successfully loaded snapshot for period {period}")
                else:
                    print(f"Snapshot file not found for period {period} ({filepath})")
            except Exception as e:
                print(f"Error loading snapshot for period {period}: {e}")
                continue
    return snapshots


def load_detected_communities(
        dir: str = "../data/02_preprocessed",
        filename: str = "community_memberships.pkl",
) -> dict:
    """Load data about communities detected in user-user interaction network snapshots."""
    try:
        filepath = os.path.join(dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "rb") as fp:
                communities = pickle.load(fp)
            print(f"Successfully loaded communities from {filepath}")
            return communities
        else:
            print(f"Communities file not found at {filepath}")
            return None
    except Exception as e:
        print(f"Error loading communities file: {e}")
        return None


def load_network_metrics(
        dir: str = "../data/02_preprocessed",
        filename: str = "metrics.pkl",
) -> tuple:
    """Load metrics and distribution data about user-user interaction network snapshots."""
    try:
        filepath = os.path.join(dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "rb") as fp:
                metrics = pickle.load(fp)
            print(f"Successfully loaded metrics from {filepath}")
            return pd.DataFrame(metrics[0]), metrics[1]
        else:
            print(f"Metrics file not found at {filepath}")
            return None
    except Exception as e:
        print(f"Error loading metrics file: {e}")
        return None


def create_y_series_metrics(
        sorted_periods: Iterable,
        df_metrics: pd.DataFrame,
        metric_name: str,
        include_network: bool = True,
        include_gcc: bool = True,
        include_largest_community: bool = True,
):
    y_data = []

    if include_network:
        y_data_network = [
            df_metrics.loc[
                (
                    (df_metrics["network"] == x)
                    & (df_metrics["metric_name"] == metric_name)
                ),
                "metric_value"
            ]
            for x in sorted_periods
        ]
        y_data.append(y_data_network)

    if include_gcc:
        y_data_gcc = [
            df_metrics.loc[
                (
                    (df_metrics["network"] == f"{x}-GCC")
                    & (df_metrics["metric_name"] == metric_name)
                ),
                "metric_value"
            ].iloc[0]
            for x in sorted_periods
        ]
        y_data.append(y_data_gcc)

    if include_largest_community:
        y_data_largest_community = [
            df_metrics.loc[
                (
                    (df_metrics["network"] == f"{x}-COMM")
                    & (df_metrics["metric_name"] == metric_name)
                ),
                "metric_value"
            ].iloc[0]
            for x in sorted_periods
        ]
        y_data.append(y_data_largest_community)

    return y_data


def create_plot(
        x_data: Iterable,
        y_data: Iterable,
        y_data_labels: Iterable,
        x_axis_label: str = "",
        y_axis_label: str = "",
        title: str = "",
        is_histogram: bool = False,
        log_scaled: bool = False,
        log_axes: str = "xy",
        save_dir: str = "../data/04_reports/plots",
        save_file: str = "plot.png",
) -> None:
    """
    Create a temporal distribution plot, optionally with log-scaled axes.
    Save the plot as `save_file` at `save_dir` and displays the figure.

    Args:
        x_data (Iterable): Data points to display along the X-axis.
        y_data (Iterable): Series of data points to display along the Y-axis.
        y_data_labels (Iterable): Names of the labels for each y_data series.
        x_axis_label (str, optional): Label to display on the X-axis (default: '').
        y_axis_label (str, optional): Label to display on the Y-axis (default: '').
        title (str, optional): Plot title (default: '').
        is_histogram (bool, optional): If true, plots a scatter histogram of the
            distribution of `x_data` (default: False).
        log_scaled (bool, optional): Whether to log-scale the plot axes (default: False).
        log_axes (str, optional): Name of the axes to log-scale (default: 'xy').
        save_dir (str, optional): Folder in which to save the output (default: '../data/04_reports').
        save_file (str, optional): File name to save the output (default: 'plot.png').
    """

    def _set_log10(ax, x_axis_label, y_axis_label, axis="xy"):
        """Compatibility wrapper for different matplotlib versions."""
        try:
            if "y" in axis:
                ax.set_yscale("log", base=10)
                y_axis_label = f"{y_axis_label} (log10)"
            if "x" in axis:
                ax.set_xscale("log", base=10)
                x_axis_label = f"{x_axis_label} (log10)"
        except TypeError:
            if "y" in axis:
                ax.set_yscale("log", basey=10)
                y_axis_label = f"{y_axis_label} (log10)"
            if "x" in axis:
                ax.set_xscale("log", basex=10)
                x_axis_label = f"{x_axis_label} (log10)"
        finally:
            return x_axis_label, y_axis_label

    _, ax = plt.subplots(figsize=(15, 8))

    if is_histogram:
        frequencies = pd.Series(x_data).value_counts()
        ax.scatter(frequencies.index, frequencies.values)

    else:
        for y, label in zip(y_data, y_data_labels):
            ax.plot(x_data, y, label=label)
        ax.legend(bbox_to_anchor=(1.0, 1.0))

    if log_scaled:
        x_axis_label, y_axis_label = _set_log10(ax, x_axis_label, y_axis_label, log_axes)

    ax.set_title(title)
    ax.set_xlabel(x_axis_label.strip())
    ax.set_ylabel(y_axis_label.strip())
    ax.tick_params(axis="x", rotation=90)
    ax.grid(which="major", color="grey", linestyle="--", linewidth=0.5)
    ax.grid(which="minor", color="lightgrey", linestyle="--", linewidth=0.2)
    plt.tight_layout()

    save_filename = os.path.join(save_dir, save_file)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_filename, dpi=300, bbox_inches="tight")

    plt.show()
