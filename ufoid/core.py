import glob
import logging
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from imagehash import phash
from numba import njit, prange
from numba_progress import ProgressBar
from omegaconf.listconfig import ListConfig
from PIL import Image
from tqdm.auto import tqdm

from . import utils

logger = utils.get_logger(__name__)


def compute_hash(path, hash_size=16):
    """Computes the perceptual hash of an image.

    Args:
        path (str): The path to the image file.
        hash_size (int): The length in bit of the hash_size (16 is optimal for our use case, since it allows to get all
         the exact duplicate (also with some resilience to minor manipulations on images), while avoiding collisions.
         See https://docs.google.com/document/d/16DS-Z-SHKtmTzQikxCO4SwRJU0cHS_-TVkf9UAKYRZA/edit#heading=h.uybo5cpys4ee
          for an extensive study on this).

    Returns:
        ImageHash: The computed average hash.
    """
    return phash(Image.open(path), hash_size=hash_size).hash


def parallel_paths_to_hashes(images: list[str], num_processes=cpu_count()-1):
    """Computes the hashes of images in parallel.

    Args:
        images (list): List of image paths.
        num_processes (int): Number of parallel processes to create.

    Returns:
        numpy.ndarray: Array of image hashes of size (num_hashes, num_bits).
    """
    logger.info("Hashing images")
    # Create a ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Compute hashes in parallel
        hashes_np = np.array(
            list(tqdm(executor.map(compute_hash, images), total=len(images), colour="green"))
        )

    return hashes_np


@njit(parallel=True)
def get_distance_matrix(images, targets, progress_proxy, symmetric=False):
    """Compares a target image with a list of images by using the hamming distance algorithm. This
    version is optimized by pre-compilation and parallelization using numba. It becomes faster when
    images list dimension is high, but the "breaking point" depends on system hardware, so
    experimentation on your given system is suggested.

    Args:
        images (np.ndarray[np.ndarray]): A NumPy array of NumPy arrays to compare with the target. Size (num_hashes, num_bits).
        targets (np.ndarray[np.ndarray]):: an `np.ndarray` target obj to compare with the given list of size (num_bits).
        progress_proxy (ProgressBar): An object of type ProgressBar from the numba_progress module that allows progress updates.
        symmetric (bool, optional): If True, optimize the computation by calculating the triangular difference matrix.

    Returns:
        np.ndarray: A matrix of hamming distances representing how similar the two sets of images are.
        The positions in the matrix correspond to each comparison.
    """
    num_hashes_1 = images.shape[0]
    num_hashes_2 = targets.shape[0]
    distance_matrix = np.zeros((num_hashes_1, num_hashes_2), dtype=np.uint64)

    # Iterate over the indices using prange for parallel execution
    for i in prange(num_hashes_1):
        progress_proxy.update(1)
        if symmetric:
            for j in range(i + 1, num_hashes_2):
                diff = np.bitwise_xor(images[i], targets[j]).sum()
                distance_matrix[i, j] = diff
                distance_matrix[j, i] = diff
        else:
            for j in range(num_hashes_2):
                distance_matrix[i, j] = np.bitwise_xor(images[i], targets[j]).sum()

    return distance_matrix


@njit()
def get_duplicates_idxs(distance_matrix, progress_proxy, distance_threshold=11, symmetric=False):
    """Find duplicate indices based on a given distance_threshold in a distance matrix. This
    optimized version of the function uses Numba for faster execution.

    Args:
        distance_matrix (numpy.ndarray): A 2D numpy array representing the differences between elements.
        progress_proxy (ProgressBar): An object of type ProgressBar from the numba_progress module that allows progress updates.
        distance_threshold (int, optional): The distance_threshold used to determine duplicates. Elements with a distance
            less than this threshold are considered duplicates. Default is 11.
        symmetric (bool, optional): If True, considers only the upper triangular part of the matrix to avoid
            duplicate comparisons for symmetric matrices. Default is False.

    Returns:
        List[Tuple[int, int, int]]: A list of tuples where each tuple contains the indices (i, j) and the distance
        value for elements that are considered duplicates based on the given threshold.

    Note:
        - The function iterates through the elements of the distance_matrix and checks if the difference between
          elements is less than the specified distance_threshold.
        - If 'symmetric' is True, the function only checks the upper triangular part of the matrix, assuming that
          the matrix is symmetric.
        - Numba is used for optimization, which can significantly improve performance.
    """
    duplicates = []

    for i in prange(distance_matrix.shape[0]):
        progress_proxy.update(1)
        if symmetric:
            for j in range(i + 1, distance_matrix.shape[1]):
                diff = distance_matrix[i, j]
                if diff < distance_threshold:
                    duplicates.append((i, j, diff))
        else:
            for j in range(distance_matrix.shape[1]):
                diff = distance_matrix[i, j]
                if diff < distance_threshold:
                    duplicates.append((i, j, diff))

    return duplicates


def get_images_from_paths(paths, extensions=("png", "jpg", "jpeg")):
    """Retrieves image paths from the given list of paths.

    Args:
        paths (list): List of directory paths.
        extensions (tuple): Tuple of image file extensions to search for.

    Returns:
        list: List of image paths.
    """
    if not isinstance(paths, (list, ListConfig)):
        raise ValueError("The 'paths' argument must be a list of directory paths or ListConfig.")

    image_paths = []
    for path in paths:
        for ext in extensions:
            image_paths.extend(glob.glob(path + f"/**/*.{ext}", recursive=True))

    if not image_paths:
        raise ValueError(f"No images found in the specified paths {paths}.")

    return image_paths


def extract_duplicates_within_a_list_of_paths(images, hashes, distance_threshold=11):
    """Extracts duplicates within a list of image paths.

    Args:
        images (list): List of image paths.
        hashes (numpy.ndarray): Array of image hashes.
        distance_threshold (int): The distance_threshold for considering images as duplicates. (10 is optimal for our use
         case, since it allows to get all the exact duplicate (also with some resilience to minor manipulations on
         images), while avoiding collisions.
         See https://docs.google.com/document/d/16DS-Z-SHKtmTzQikxCO4SwRJU0cHS_-TVkf9UAKYRZA/edit#heading=h.uybo5cpys4ee
          for an extensive study on this).

    Returns:
        list: List of tuples containing duplicate image pairs (path1, path2, hamming distance).
    """

    logger.info("Getting distance matrix")
    with ProgressBar(total=len(hashes), colour="green") as progress:
        distance_matrix = get_distance_matrix(
            images=hashes, targets=hashes, symmetric=True, progress_proxy=progress
        )

    logger.info("Searching for duplicates idxs")
    with ProgressBar(total=len(hashes), colour="green") as progress:
        duplicates_idxs = get_duplicates_idxs(
            distance_matrix,
            symmetric=True,
            progress_proxy=progress,
            distance_threshold=distance_threshold,
        )
    logger.info("Finding duplicates couples by matching indexes with images lists")
    duplicates = [
        (images[i], images[j], diff) for i, j, diff in tqdm(duplicates_idxs, colour="GREEN")
    ]

    return duplicates


def extract_duplicates_between_two_lists_of_paths(
    images_1, images_2, hashes_1, hashes_2, distance_threshold=11
):
    """Extracts duplicates between two lists of image paths.

    Args:
        images_1 (list): List of image paths from the first set
        images_2 (list): List of image paths from the second set.
        hashes_1 (numpy.ndarray): Array of image hashes from the first set.
        hashes_2 (numpy.ndarray): Array of image hashes from the second set.
        distance_threshold (int): The distance_threshold for considering images as duplicates. (10 is optimal for our use
         case, since it allows to get all the exact duplicate (also with some resilience to minor manipulations on
         images), while avoiding collisions.
         See https://docs.google.com/document/d/16DS-Z-SHKtmTzQikxCO4SwRJU0cHS_-TVkf9UAKYRZA/edit#heading=h.uybo5cpys4ee
          for an extensive study on this).

    Returns:
        list: List of tuples containing duplicate image pairs (path1, path2, hamming distance).
    """

    logger.info("Getting distance matrix")
    with ProgressBar(total=len(hashes_1), colour="green") as progress:
        distance_matrix = get_distance_matrix(
            images=hashes_1, targets=hashes_2, symmetric=False, progress_proxy=progress
        )

    logger.info("Searching for duplicates idxs")
    with ProgressBar(total=len(hashes_1), colour="green") as progress:
        duplicates_idxs = get_duplicates_idxs(
            distance_matrix,
            symmetric=False,
            progress_proxy=progress,
            distance_threshold=distance_threshold,
        )
    logger.info("Finding duplicates couples by matching indexes with images lists")
    duplicates = [
        (images_1[i], images_2[j], diff) for i, j, diff in tqdm(duplicates_idxs, colour="GREEN")
    ]

    return duplicates


def divide_in_chunks(input_list, chunk_length):
    """Divides a list into chunks of the specified length.

    Args:
        input_list (list): The input list.
        chunk_length (int): The length of each chunk.

    Returns:
        generator: A generator yielding chunks of the list.
    """
    for i in range(0, len(input_list), chunk_length):
        yield input_list[i: i + chunk_length]


def extract_duplicates_within_a_dataset_no_chunk(images, num_processes=cpu_count()-1, distance_threshold=11):
    """Extracts duplicates within a dataset without using chunks.

    Args:
        images (list): List of images.
        num_processes (int): Number of processes for parallel execution.
        distance_threshold (int): The distance_threshold for considering images as duplicates.

    Returns:
        list: List of tuples containing duplicate image pairs.
        np.ndarray: hashes of the images, useful to avoid recalculating them when comparing to a different set.
    """
    logger.info("Starting duplicate detection within the dataset")

    # Calculate hashes for the entire dataset
    hashes_np = parallel_paths_to_hashes(images, num_processes=num_processes)

    duplicates = extract_duplicates_within_a_list_of_paths(
        images, hashes=hashes_np, distance_threshold=distance_threshold
    )

    logger.info("Duplicate detection within the dataset completed.")

    return sorted(duplicates), hashes_np


def extract_duplicates_within_a_dataset_with_chunks(
    images, chunk_length=50000, num_processes=cpu_count()-1, distance_threshold=11
):
    """Extracts duplicates within a dataset using chunks.

    Args:
        images (list): List of images.
        chunk_length (int): The length of each chunk.
        num_processes (int): Number of processes for parallel execution.
        distance_threshold (int): The distance_threshold for considering images as duplicates.

    Returns:
        list: List of tuples containing duplicate image pairs.
        np.ndarray: hashes of the images, useful to avoid recalculating them when comparing to a different set.
    """
    hashes_np = parallel_paths_to_hashes(images, num_processes=num_processes)
    chunks = list(divide_in_chunks(images, chunk_length))
    hashes_chunks = list(divide_in_chunks(hashes_np, chunk_length))

    duplicates = []

    logger.info(
        "Starting duplicate detection within the dataset using chunks, to avoid memory overload due to high "
        "number of images"
    )

    for chunk_index, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {chunk_index + 1}/{len(chunks)}")
        chunk_hashes = hashes_chunks[chunk_index]
        chunk_duplicates = extract_duplicates_within_a_list_of_paths(
            chunk, hashes=chunk_hashes, distance_threshold=distance_threshold
        )
        logger.info("extending duplicates list")
        duplicates.extend(chunk_duplicates)

    logger.info("Duplicate detection within the dataset using chunks completed.")
    logger.info("Starting duplicate detection between chunks")

    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            logger.info(f"Checking for duplicates between chunks {i} and {j}")
            chunk_i_hashes = hashes_chunks[i]
            chunk_j_hashes = hashes_chunks[j]
            inter_chunk_duplicates = extract_duplicates_between_two_lists_of_paths(
                chunks[i],
                chunks[j],
                hashes_1=chunk_i_hashes,
                hashes_2=chunk_j_hashes,
                distance_threshold=distance_threshold,
            )
            logger.info("extending duplicates list")
            duplicates.extend(inter_chunk_duplicates)

    logger.info("Duplicate detection between chunks completed.")
    return sorted(duplicates), hashes_np


def extract_duplicates_within_a_dataset(
    paths, chunk_length=50000, num_processes=cpu_count()-1, distance_threshold=11, logger_level=logging.INFO
):
    """Extracts duplicates within a dataset and automatically chooses whether to chunk based on
    dataset size.

    Args:
        paths (list): List of directory paths.
        chunk_length (int): The length of each chunk.
        num_processes (int): Number of processes for parallel execution.
        distance_threshold (int): The distance_threshold for considering images as duplicates.
        logger_level (int): Logging level.

    Returns:
        list: List of tuples containing duplicate image pairs.
        np.ndarray: hashes of the images, useful to avoid recalculating them when comparing to a different set.
    """
    if logger_level != logging.INFO:
        logger.setLevel(logger_level)

    images = sorted(get_images_from_paths(paths))

    if len(images) > chunk_length:
        # If the dataset is large, use chunking
        return extract_duplicates_within_a_dataset_with_chunks(
            images, chunk_length, num_processes, distance_threshold
        )
    else:
        # If the dataset can fit in memory, use the non-chunked version
        return extract_duplicates_within_a_dataset_no_chunk(
            images, num_processes, distance_threshold
        )


def extract_duplicates_between_two_datasets_no_chunk(
    images_ref, images_new, num_processes=cpu_count()-1, distance_threshold=11, hashes_new=None
):
    """Extract duplicates between two datasets without using chunks.

    Args:
        images_ref (list): List of reference images.
        images_new (list): List of images to be checked.
        num_processes (int): Number of processes for parallel execution.
        distance_threshold (int): The distance_threshold for considering images as duplicates.
        hashes_new (numpy.ndarray, optional): Array of image hashes for the 'paths_new' dataset.
            If provided, the function will use these pre-calculated hashes instead of calculating them.

    Returns:
        list: List of tuples containing duplicate image pairs.
    """

    logger.info("Starting duplicate detection between two datasets")

    # Calculate hashes for both datasets
    hashes_ref = parallel_paths_to_hashes(images_ref, num_processes=num_processes)

    if hashes_new is None:
        hashes_new = parallel_paths_to_hashes(images_new, num_processes=num_processes)
    else:
        logger.info("Using preloaded hashes for new data")

    # Detect duplicates between the reference and new datasets
    duplicates = extract_duplicates_between_two_lists_of_paths(
        images_ref,
        images_new,
        hashes_1=hashes_ref,
        hashes_2=hashes_new,
        distance_threshold=distance_threshold,
    )

    logger.info("Duplicate detection between two datasets completed.")
    return sorted(duplicates)


def extract_duplicates_between_two_datasets_with_chunks(
    images_reference, images_new, chunk_length=50000, num_processes=cpu_count()-1, distance_threshold=11
):
    """Extracts duplicates between two datasets using chunks.

    Args:
        images_reference (list): List of reference images.
        images_new (list): List of images to be checked.
        chunk_length (int): The length of each chunk.
        num_processes (int): Number of processes for parallel execution.
        distance_threshold (int): The distance_threshold for considering images as duplicates.

    Returns:
        list: List of tuples containing duplicate image pairs.
    """
    hashes_ref = parallel_paths_to_hashes(images_reference, num_processes=num_processes)
    hashes_new = parallel_paths_to_hashes(images_new, num_processes=num_processes)
    chunks_ref = list(divide_in_chunks(images_reference, chunk_length))
    chunks_new = list(divide_in_chunks(images_new, chunk_length))
    hashes_chunks_ref = list(divide_in_chunks(hashes_ref, chunk_length))
    hashes_chunks_new = list(divide_in_chunks(hashes_new, chunk_length))

    duplicates = []

    logger.info("Starting duplicate detection between two datasets using chunks")

    # Cross-chunk duplicates
    for i, chunk_ref in enumerate(chunks_ref):
        for j, chunk_new in enumerate(chunks_new):
            logger.info(
                f"Processing reference chunk {i + 1}/{len(chunks_ref)} and new chunk {j + 1}/{len(chunks_new)}"
            )
            # Calculate hashes for the current chunks being compared
            hashes_i = hashes_chunks_ref[i]
            hashes_j = hashes_chunks_new[j]
            logger.info("extending duplicates list")
            duplicates.extend(
                extract_duplicates_between_two_lists_of_paths(
                    chunk_ref,
                    chunk_new,
                    hashes_1=hashes_i,
                    hashes_2=hashes_j,
                    distance_threshold=distance_threshold,
                )
            )

    logger.info("Duplicate detection between two datasets using chunks completed.")
    return sorted(duplicates)


def extract_duplicates_between_two_datasets(
    paths_reference,
    paths_new,
    chunk_length=50000,
    num_processes=cpu_count()-1,
    distance_threshold=11,
    logger_level=logging.INFO,
    hashes_new=None,
):
    """Extract duplicates between two datasets and automatically chooses whether to chunk based on
    dataset sizes.

    Args:
        paths_reference (list): List of reference directory paths of images.
        paths_new (list): List of directory paths of images to be checked.
        chunk_length (int): The length of each chunk.
        num_processes (int): Number of processes for parallel execution.
        distance_threshold (int): The distance_threshold for considering images as duplicates.
        logger_level (int): Logging level.
        hashes_new (numpy.ndarray, optional): Array of image hashes for the 'paths_new' dataset.
            If provided, the function will use these pre-calculated hashes instead of calculating them.

    Returns:
        list: List of tuples containing duplicate image pairs.
    """
    if logger_level != logging.INFO:
        logger.setLevel(logger_level)

    images_ref = sorted(get_images_from_paths(paths_reference))
    images_new = sorted(get_images_from_paths(paths_new))

    if len(images_ref) > chunk_length or len(images_new) > chunk_length:
        # If either of the datasets is large, use chunking
        return extract_duplicates_between_two_datasets_with_chunks(
            images_ref, images_new, chunk_length, num_processes, distance_threshold
        )
    else:
        # If both datasets can fit in memory, use the non-chunked version
        return extract_duplicates_between_two_datasets_no_chunk(
            images_ref, images_new, num_processes, distance_threshold, hashes_new
        )
