import logging
import os
import shutil
import time
import yaml
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.dataset_creation.speed import generate_n_random_images
from ufoid.core import extract_duplicates_within_a_dataset
from ufoid.utils import get_logger

log = get_logger(__name__)


def run_ufoid(
        dataset_path: str,
        chunk_length: int,
        num_processes: int,
        logger_level: str = logging.INFO,
):
    """Single experiment for benchmarking.

    Args:
        dataset_path (str): Path to the dataset.
        chunk_length (int): The length of chunks for splitting the image group.
        num_processes (int): The number of processes to use for parallel execution.
        logger_level (int): Logging level.

    Returns:
        float: The elapsed time in seconds to find duplicates within the group.
    """

    start_time = time.time()
    _, _ = extract_duplicates_within_a_dataset(
        paths=[str(dataset_path)],
        chunk_length=chunk_length,
        num_processes=num_processes,
        distance_threshold=30,
        logger_level=logger_level,
    )
    elapsed_time = time.time() - start_time
    return elapsed_time


def main():
    cfg_filepath = Path(__file__).resolve().parent.parent / 'config' / 'optimization.yaml'
    with open(cfg_filepath) as f:
        cfg = yaml.safe_load(f)

    early_stop = cfg['GENERAL']['early_stop']
    num_processes = sorted(cfg['UFOID_PARAMS']['num_processes'])
    chunk_lengths = cfg['UFOID_PARAMS']['chunk_lengths']

    dataset_path = Path(__file__).resolve().parent.parent.parent / 'data' / "benchmark_speed_temp" / "images"
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)

    results = []
    for dataset in cfg['DATASETS']:
        dataset_name, num_images, image_size, duplicates_percentage = (
            dataset["name"],
            dataset['image_numbers'],
            dataset['image_size'],
            dataset['duplicates_percentages'],
        )

        log.info(f'Generating "{dataset_name}"')
        generate_n_random_images(
            n=num_images,
            size=image_size,
            path=dataset_path,
            duplicates_percentage=duplicates_percentage,
        )

        best_elapsed_time = np.inf
        current_elapsed_time = np.inf
        for num_proc in num_processes:
            for chunk_len in chunk_lengths:
                log.info(f'Running ufoid on "{dataset_name}" with {num_proc=}, {chunk_len=}')
                for _ in range(2):
                    # first iteration for warmup
                    current_elapsed_time = run_ufoid(
                        dataset_path=dataset_path,
                        chunk_length=chunk_len,
                        num_processes=num_proc,
                    )

                results.append({
                    "dataset_name": dataset_name,
                    "n_of_images": num_images,
                    "n_of_processes": num_proc,
                    "chunk_length": chunk_len,
                    "image_size": image_size,
                    "duplicates_percentage": duplicates_percentage,
                    "time (s)": current_elapsed_time,
                })

                if current_elapsed_time < best_elapsed_time:
                    best_elapsed_time = current_elapsed_time
                    log.info(f"New best elapsed time: {best_elapsed_time} with {num_proc=} and {chunk_len=}")
                elif early_stop:
                    log.info(f"Got {current_elapsed_time=} gte than {best_elapsed_time=}")
                    log.info("--------- Early Stop ---------")
                    break

        shutil.rmtree(dataset_path)

        csv_dir_path = Path(__file__).resolve().parent.parent / 'results' / 'csv'
        os.makedirs(csv_dir_path, exist_ok=True)
        csv_filepath = csv_dir_path / 'optimization.csv'
        df = pd.DataFrame(results).round(decimals=3).sort_values(by='time (s)')
        df.to_csv(csv_filepath, index=False)
        log.info(f"results saved to {csv_filepath}")


if __name__ == '__main__':
    main()
