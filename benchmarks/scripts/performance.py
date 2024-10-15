import glob
import logging
import os
import shutil
import multiprocessing as mp
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable, Union, Literal, Optional

import yaml

import pandas as pd
from imagededup.evaluation import evaluate
from imagededup.methods import PHash

from benchmarks.dataset_creation.imagededup import download_and_prepare_imagededup_dataset
from benchmarks.dataset_creation.speed import generate_n_random_images
from ufoid import core, utils
from ufoid.core import get_images_from_paths

log = utils.get_logger(__name__)

ProcessStatus = Literal["ERROR", "SUCCESSFUL", "TIMEOUT"]
WarmupProcessFailedStatus = Literal["WARMUP ERROR", "WARMUP TIMEOUT"]
ProcessWithWarmupStatus = Union[ProcessStatus, WarmupProcessFailedStatus]


def ufoid2dedup(duplicates_list: list):
    duplicates_dict = {}
    # Loop through the list of duplicates and add them to the dictionary
    for original, duplicate, _ in duplicates_list:
        # Extract the file names
        original_filename = os.path.basename(original)
        duplicate_filename = os.path.basename(duplicate)

        # If the original file is not already in the dictionary, add it with an empty list
        if original_filename not in duplicates_dict:
            duplicates_dict[original_filename] = []
        if duplicate_filename not in duplicates_dict:
            duplicates_dict[duplicate_filename] = []

        # Append the duplicate to the list for the original file
        duplicates_dict[original_filename].append(duplicate_filename)
        duplicates_dict[duplicate_filename].append(original_filename)

    return duplicates_dict


def find_duplicates_ufoid(shared_dict: dict, dataset_path: str, threshold: Union[int, float], chunk_size: int, num_processes: int):
    """Find duplicate images using the UFOID library.

    Args:
        shared_dict (dict): Shared dictionary used for multiprocessing output communication.
        dataset_path (str): The path to the directory containing images for deduplication.
        threshold (int): The threshold for considering images as duplicates.
        num_processes (int): The number of processes for ufoid algorithm.
        chunk_size (int): Chunk size for ufoid algorithm.

    Returns:
        Tuple: A tuple containing the retrieved map and a list of times
    """
    duplicates_list, _ = core.extract_duplicates_within_a_dataset(
        [dataset_path],
        distance_threshold=threshold + 1,
        num_processes=num_processes,
        chunk_length=chunk_size,
        logger_level=logging.ERROR,
    )
    nodup_images = set(get_images_from_paths([dataset_path])) - {img for pair in duplicates_list for img in pair[:2]}
    retrieved_map = ufoid2dedup(duplicates_list)
    retrieved_map.update({os.path.basename(x): [] for x in nodup_images})
    shared_dict.update(retrieved_map)


def find_duplicates_dedup(shared_dict: dict, dataset_path: str, threshold: Union[int, float]):
    """Find duplicate images using the ImageDedup library.

    Args:
        shared_dict (dict): The dictionary used for multiprocessing output communication.
        dataset_path (str): The path to the directory containing images for deduplication.
        threshold (int): The threshold for considering images as duplicates.

    Returns:
        Tuple: A tuple containing the retrieved map and a list of times
    """
    phasher = PHash()
    retrieved_map = phasher.find_duplicates(
        image_dir=dataset_path,
        max_distance_threshold=threshold,
        scores=False,
        num_enc_workers=6,
        num_dist_workers=6,
    )
    shared_dict.update(retrieved_map)


def get_ground_truth_imagededup(path: str):
    filenames = sorted(
        [os.path.basename(i) for i in glob.glob(os.path.join(path, "*")) if "_copy" not in i]
    )
    ground_truth_map = {
        filename: [filename.replace(".jpg", "_copy.jpg")] for filename in filenames
    }
    ground_truth_map_rev = {
        filename.replace(".jpg", "_copy.jpg"): [filename] for filename in filenames
    }
    ground_truth_map = ground_truth_map | ground_truth_map_rev
    return ground_truth_map


def get_ground_truth_local(dataset_path: str):
    image_list = [os.path.basename(x) for x in get_images_from_paths([dataset_path])]
    grouped = defaultdict(list)

    for image in image_list:
        base_name = image.split('-')[0]
        grouped[base_name].append(image)

    final_dict = {}

    for base_name, images in grouped.items():
        for image in images:
            final_dict[image] = [img for img in images if img != image]

    return final_dict


def get_metrics(ground_truth_map: dict, retrieved_map: dict):
    metrics = evaluate(ground_truth_map, retrieved_map, metric="classification")
    nodup_precision, dup_precision = metrics["precision"]
    nodup_recall, dup_recall = metrics["recall"]
    return nodup_precision, dup_precision, nodup_recall, dup_recall


def setup_dataset(dataset_args: dict) -> str:
    if dataset_args["type"] == "imagededup":
        download_and_prepare_imagededup_dataset(base_path="data", logger=log)
        return Path(__file__).resolve().parent.parent.parent / "data/image_dedup_benchmark"

    elif dataset_args["type"] == "local":
        return dataset_args["path"]

    elif dataset_args["type"] == "synth":
        dataset_dir_path = Path(__file__).resolve().parent.parent.parent / "data" / dataset_args["name"]
        if os.path.isdir(dataset_dir_path):
            shutil.rmtree(dataset_dir_path)
        os.makedirs(dataset_dir_path)
        log.info(
            f"Generating {dataset_args['image_count']} {dataset_args['image_size']}x{dataset_args['image_size']} images "
            f"and saving them to {dataset_dir_path}."
            f" {dataset_args['duplicates_percentage'] * 100}% of images will be duplicates between themselves. \n"
            f" Those will be generated after the other images, their indexes are pre-shuffled to mimic a real scenario."
        )
        generate_n_random_images(
            n=dataset_args['image_count'],
            size=dataset_args['image_size'],
            path=dataset_dir_path,
            duplicates_percentage=dataset_args['duplicates_percentage'],
            num_processes=3,
        )
        return dataset_dir_path


def run_with_timeout(func, timeout, *args, **kwargs) -> (ProcessStatus, Optional[float], tuple):
    manager = mp.Manager()
    shared_dict = manager.dict()
    p = mp.Process(target=func, args=(shared_dict, *args), kwargs=kwargs)

    start_time = time.time()
    p.start()
    try:
        p.join(timeout=timeout)
    except Exception as e:
        log.error(f"Process exited with error {e}")
        return "ERROR", None, None
    elapsed_time = time.time() - start_time

    if p.is_alive():
        p.terminate()
        p.join()
        manager.shutdown()
        return "TIMEOUT", elapsed_time, None

    return "SUCCESSFUL", elapsed_time, shared_dict


def run_with_warmup(func: Callable, timeout: Union[int, float],
                    *args, **kwargs) -> (ProcessWithWarmupStatus, Optional[float], Optional[tuple]):
    log.info("Starting warmup")
    status, elapsed_time, out = run_with_timeout(func, timeout, *args, **kwargs)
    log.info(f"Warmup finished with status {status}")

    if status == "SUCCESSFUL":
        status, elapsed_time, out = run_with_timeout(func, timeout, *args, **kwargs)
    else:
        status = f"WARMUP {status}"

    return status, elapsed_time, out


def main():
    config_filepath = Path(__file__).resolve().parent.parent / 'config' / 'performance.yaml'
    csv_dir_path = Path(__file__).resolve().parent.parent / 'results' / 'csv'
    csv_filepath = csv_dir_path / 'performance.csv'
    with open(config_filepath) as f:
        cfg = yaml.safe_load(f)

    results = []
    timeout = cfg["GENERAL"]["timeout"]
    find_duplicates_func = {
        "imagededup": find_duplicates_dedup,
        "ufoid": partial(
            find_duplicates_ufoid,
            num_processes=cfg['MODELS']['ufoid']['num_processes'],
            chunk_size=cfg['MODELS']['ufoid']['chunk_size'],
        ),
    }
    get_ground_truth_func = {
        "imagededup": get_ground_truth_imagededup,
        "local": get_ground_truth_local,
    }

    for dataset in cfg["DATASETS"]:
        dataset_path = str(setup_dataset(dataset))
        dataset_name = os.path.basename(dataset_path)
        num_images = len(get_images_from_paths([dataset_path]))

        for lib_name, model_args in cfg["MODELS"].items():
            if not model_args["active"]:
                log.info(f"Skipping model {lib_name}")
                continue

            evaluate_precision = model_args['precision_eval']['active'] and dataset["type"] != "synth"
            gt_map = get_ground_truth_func[dataset["type"]](dataset_path) if evaluate_precision else None
            for threshold in model_args['precision_eval']["thresholds"]:
                log.info(f'Starting images deduplication for "{dataset_name}" using {lib_name} with {threshold=}')
                status, elapsed_time, retrieved_map = run_with_warmup(
                    find_duplicates_func[lib_name],
                    timeout,
                    dataset_path=dataset_path,
                    threshold=threshold,
                )
                log.info(f"Image deduplication finished with status {status}")

                nodup_precision, dup_precision, nodup_recall, dup_recall = None, None, None, None
                if gt_map is not None and retrieved_map is not None:
                    nodup_precision, dup_precision, nodup_recall, dup_recall = get_metrics(gt_map, retrieved_map)

                results.append({
                    "dataset": dataset_name,
                    "num_images": num_images,
                    "lib": lib_name,
                    "threshold": threshold,
                    "non_duplicate_precision": nodup_precision,
                    "duplicate_precision": dup_precision,
                    "non_duplicate_recall": nodup_recall,
                    "duplicate_recall": dup_recall,
                    "status": status,
                    "time (s)": elapsed_time,
                })

                if not evaluate_precision:
                    break

    df = pd.DataFrame(results).round(decimals=3)
    os.makedirs(csv_dir_path, exist_ok=True)
    df.to_csv(csv_filepath, index=False)
    log.info(f"Results saved in {csv_filepath}")


if __name__ == "__main__":
    main()
