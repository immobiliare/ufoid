import glob
import os.path
import re

import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from benchmarks.dataset_creation.imagededup import (
    download_and_prepare_imagededup_dataset,
)
from benchmarks.dataset_creation.quality import prepare_quality_benchmark_dataset
from ufoid import core, utils

logger = utils.get_logger(__name__)


def generate_duplicates(base_path):
    """Generate duplicate pairs between reference and new datasets and classify them as correct or
    errors.

    Args:
        base_path (str): Base path for dataset files.

    Returns:
        list: List of correct duplicate pairs.
        list: List of error duplicate pairs.
    """
    duplicates = core.extract_duplicates_between_two_datasets(
        paths_reference=[os.path.join(base_path, "quality_benchmark/manipulated")],
        paths_new=[os.path.join(base_path, "quality_benchmark/virgin")],
        distance_threshold=256,
    )

    correct_pairs = []
    errors = []

    for dup in tqdm(duplicates, colour="green"):
        n1 = os.path.basename(dup[0])
        n2 = os.path.basename(dup[1])

        if n1 == n2:
            correct_pairs.append(dup)
        else:
            errors.append(dup)

    return correct_pairs, errors


def generate_benchmark_dataframe(base_path, logger):
    """Generate a benchmark dataframe and save it as a CSV file.

    Args:
        base_path (str): Base path for dataset files.
        logger (Logger): Logger object for logging information about the CSV generation.
    """
    out_path = os.path.join("benchmarks/results/csv", "benchmark_quality.csv")
    if not os.path.isfile(out_path):
        logger.info("Finding duplicates: ")
        correct_pairs, errors = generate_duplicates(base_path=base_path)
        logger.info("Generating the CSV: ")
        manipulations = [
            os.path.basename(man)
            for man in glob.glob(os.path.join(base_path, "quality_benchmark/manipulated/*"))
        ]
        manipulations = sorted(manipulations)
        manip_correct_distances = {}

        for manip in manipulations:
            manip_correct_distances[manip] = np.array([pair[2] for pair in correct_pairs if manip in pair[0]])

        error_distances = np.array([pair[2] for pair in errors])

        n_virgin = len(glob.glob(os.path.join(base_path, "quality_benchmark/virgin", "*")))
        n_manip = len(glob.glob(os.path.join(base_path, "quality_benchmark/manipulated", "*", "*")))
        possible_errors = (n_virgin * n_manip) - (n_virgin * 18)

        benchmark_data = []

        for th in tqdm(range(0, 180), colour="green"):
            th_data = [th]

            for manip in manipulations:
                th_data.append((sum(manip_correct_distances[manip] < (th + 1))) / n_virgin)

            collisions = sum(error_distances < (th + 1))
            th_data.append(collisions)
            th_data.append(collisions / possible_errors)
            benchmark_data.append(th_data)

        df = pd.DataFrame(
            benchmark_data,
            columns=["threshold"]
            + [man + "_precision" for man in manipulations]
            + ["collisions", "collisions_percentage"],
        )
        os.makedirs("benchmarks/results/csv/", exist_ok=True)
        df.to_csv(os.path.join("benchmarks/results/csv", "benchmark_quality.csv"), index=False)
        logger.info("Benchmark data saved as CSV.")
    else:
        logger.info("Benchmark data CSV already existing.")


def plot_csv(path, logger):
    """Generate and save precision and collision plots based on the benchmark data.

    Args:
        path (str): Path to the benchmark data CSV file.
        logger (Logger): Logger object for logging information about the plot generation.
    """
    df = pd.read_csv(path)[0:100]
    plt.style.use(style="fivethirtyeight")
    plt.figure(figsize=(18, 10))
    thresholds = df["threshold"]

    # Create a dictionary to map colors based on the first word in column names
    color_map = {}

    for column in df.columns:
        if column not in ("threshold", "collisions", "collisions_percentage"):
            first_word = re.split(r"_", column)[0]

            if first_word not in color_map:
                color_map[first_word] = mp.colormaps.get_cmap("tab20")(len(color_map))

            color = color_map[first_word]

            # Check if "pos" or "neg" is in the column name and set the marker accordingly
            marker = ">" if "pos" in column else "<" if "neg" in column else "*"

            # Check if the third word is "soft" or "heavy" in the column name and set the markersize accordingly
            markersize = 6 if "soft" in column else 8 if "heavy" in column else 8

            plt.plot(
                thresholds,
                df[column],
                label=column,
                linewidth=3,
                marker=marker,
                markersize=markersize,
                color=color,
            )

    plt.legend()
    plt.xlabel("threshold")
    plt.ylabel("TP_percentage")
    os.makedirs("benchmarks/results/figurs", exist_ok=True)
    plt.savefig(os.path.join("benchmarks/results/figures", "benchmark_quality_precision.png"))
    plt.xlim(0, 16)
    plt.savefig(os.path.join("benchmarks/results/figures", "benchmark_quality_precision_zoom.png"))

    df = pd.read_csv(path)
    plt.figure(figsize=(18, 10))
    plt.plot(range(0, len(df)), df["collisions_percentage"])
    plt.xlabel("threshold")
    plt.ylabel("collision_percentage")
    plt.savefig(os.path.join("benchmarks/results/figurs", "benchmark_quality_collisions.png"))
    logger.info("Precision and collision plots saved.")


if __name__ == "__main__":
    download_and_prepare_imagededup_dataset(base_path="data", logger=logger)
    prepare_quality_benchmark_dataset(base_path="data", logger=logger)
    generate_benchmark_dataframe(base_path="data", logger=logger)
    plot_csv(path="benchmarks/results/csv/benchmark_quality.csv", logger=logger)
