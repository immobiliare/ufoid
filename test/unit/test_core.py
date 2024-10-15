import glob
import logging
import math
import os
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pytest
from imagehash import phash
from numba_progress import ProgressBar
from PIL import Image

from benchmarks.dataset_creation.speed import (
    generate_duplicate_images,
    generate_n_random_images,
    generate_random_image,
)
from ufoid.core import (
    compute_hash,
    divide_in_chunks,
    extract_duplicates_between_two_datasets,
    extract_duplicates_between_two_datasets_no_chunk,
    extract_duplicates_between_two_datasets_with_chunks,
    extract_duplicates_between_two_lists_of_paths,
    extract_duplicates_within_a_dataset,
    extract_duplicates_within_a_dataset_no_chunk,
    extract_duplicates_within_a_dataset_with_chunks,
    extract_duplicates_within_a_list_of_paths,
    get_distance_matrix,
    get_duplicates_idxs,
    get_images_from_paths,
    parallel_paths_to_hashes,
)

test_img = "data/toy_data/imgs_onlyorig/1131647350.jpg"

test_images = glob.glob("data/toy_data/*/*.jpg")[0:10]


def test_compute_hash():
    hashed_image = compute_hash(test_img, hash_size=16)
    assert np.array_equal(hashed_image, phash(Image.open(test_img), hash_size=16).hash)
    assert len(hashed_image.flatten()) == 256

    hashed_image = compute_hash(test_img, hash_size=8)
    assert np.array_equal(hashed_image, phash(Image.open(test_img), hash_size=8).hash)
    assert len(hashed_image.flatten()) == 64


def test_parallel_paths_to_hashes():
    result_hashes = parallel_paths_to_hashes(test_images, num_processes=4)

    assert np.array_equal(result_hashes[0], compute_hash(test_images[0]))


def test_get_difference_matrix_within_a_set():
    test_hashes = np.array([[0] * 16, [1] * 16])

    with ProgressBar(total=len(test_hashes)) as progress:
        difference_matrix = get_distance_matrix(
            test_hashes, test_hashes, symmetric=True, progress_proxy=progress
        )

        assert difference_matrix.shape == (len(test_hashes), len(test_hashes))
        assert difference_matrix[0, 1] == 16
        assert difference_matrix[1, 1] == 0


def test_get_difference_matrix_between_two_sets():
    # Example test hashes for illustration purposes
    test_hashes_1 = np.array([[0] * 16, [1] * 16])

    test_hashes_2 = np.array(
        [
            [0] * 16,
            [1] * 16,
        ]
    )

    with ProgressBar(total=len(test_hashes_1)) as progress:
        difference_matrix = get_distance_matrix(
            test_hashes_1, test_hashes_2, progress_proxy=progress
        )

        assert difference_matrix.shape == (len(test_hashes_1), len(test_hashes_2))
        assert difference_matrix[0, 0] == 0  # Replace with an actual expected value
        assert difference_matrix[0, 1] == 16  # Replace with an actual expected value
        assert difference_matrix[1, 1] == 0  # Replace with an actual expected value


def test_search_for_duplicates_within_a_set():
    M = np.array([[0, 20, 0], [20, 0, 20], [0, 20, 0]])

    with ProgressBar(total=M.shape[0]) as progress:
        duplicate_pairs = get_duplicates_idxs(M, progress, symmetric=True, distance_threshold=11)

        assert len(duplicate_pairs) == 1

    with ProgressBar(total=M.shape[0]) as progress:
        duplicate_pairs = get_duplicates_idxs(
            M, progress, symmetric=True, distance_threshold=30
        )  # higher threshold

        assert len(duplicate_pairs) == 3


def test_search_for_duplicates_between_two_sets():
    M = np.array([[100, 0, 10], [100, 100, 100]])

    with ProgressBar(total=M.shape[0]) as progress:
        duplicate_pairs = get_duplicates_idxs(M, progress, symmetric=False, distance_threshold=10)

        assert len(duplicate_pairs) == 1

    with ProgressBar(total=M.shape[0]) as progress:
        duplicate_pairs = get_duplicates_idxs(M, progress, symmetric=False, distance_threshold=30)

        assert len(duplicate_pairs) == 2


def test_extract_duplicates_within_a_list_of_paths():
    test_images = ["image0.jpg", "image1.jpg", "image2.jpg"]
    test_hashes = np.array([[0] * 16, [0] * 16, [0] * 16])

    # Mock the difference matrix function
    with patch(
        "ufoid.core.get_distance_matrix",
        return_value=np.zeros((len(test_images), len(test_images))),
    ):
        # Mock the search_for_duplicates_within_a_set function
        with patch(
            "ufoid.core.get_duplicates_idxs", return_value=[(1, 2, 0), (0, 2, 0), (0, 1, 0)]
        ):
            duplicate_pairs = extract_duplicates_within_a_list_of_paths(
                test_images, test_hashes, distance_threshold=11
            )

            assert len(duplicate_pairs) == 3
            assert sorted(duplicate_pairs) == [
                ("image0.jpg", "image1.jpg", 0),
                ("image0.jpg", "image2.jpg", 0),
                ("image1.jpg", "image2.jpg", 0),
            ]  # Replace with expected values


def test_extract_duplicates_between_two_lists_of_paths():
    test_images = ["image0.jpg", "image1.jpg"]
    test_images2 = ["copy0", "copy0_bis"]
    test_hashes = np.array([[0] * 16, [1] * 16])
    test_hashes2 = np.array([[0] * 16, [0] * 16])

    # Mock the difference matrix function
    with patch(
        "ufoid.core.get_distance_matrix",
        return_value=np.zeros((len(test_images), len(test_images2))),
    ):
        # Mock the search_for_duplicates_within_a_set function
        with patch("ufoid.core.get_duplicates_idxs", return_value=[(0, 0, 0), (0, 1, 0)]):
            duplicate_pairs = extract_duplicates_between_two_lists_of_paths(
                test_images, test_images2, test_hashes, test_hashes2, distance_threshold=11
            )

            assert len(duplicate_pairs) == 2
            assert duplicate_pairs == [
                ("image0.jpg", "copy0", 0),
                ("image0.jpg", "copy0_bis", 0),
            ]  # Replace with expected values


def test_divide_in_chunks():
    test_list = [1, 2, 3, 4, 5, 6, 7, 8]
    chunk_length = 3

    chunk_generator = divide_in_chunks(test_list, chunk_length)

    chunks = list(chunk_generator)

    assert len(chunks) == 3
    assert chunks[0] == [1, 2, 3]
    assert chunks[1] == [4, 5, 6]
    assert chunks[2] == [7, 8]


def test_extract_duplicates_within_a_dataset():
    test_paths = ["path/to/directory1", "path/to/directory2"]
    chunk_length = 1
    num_processes = 8

    with patch("ufoid.core.get_images_from_paths", return_value=["image1.jpg", "image2.jpg"]):
        with patch("ufoid.core.divide_in_chunks", return_value=[["image1.jpg"], ["image2.jpg"]]):
            with patch(
                "ufoid.core.parallel_paths_to_hashes",
                return_value=np.zeros((2, 256), dtype=np.uint64),
            ):
                with patch(
                    "ufoid.core.extract_duplicates_within_a_list_of_paths", return_value=[]
                ):
                    with patch(
                        "ufoid.core.extract_duplicates_between_two_lists_of_paths",
                        return_value=[("image1.jpg", "image2.jpg", 5)],
                    ):
                        duplicate_pairs, _ = extract_duplicates_within_a_dataset(
                            test_paths,
                            chunk_length,
                            num_processes,
                            distance_threshold=11,
                            logger_level=logging.ERROR,
                        )

                        assert len(duplicate_pairs) == 1
                        assert duplicate_pairs[0] == ("image1.jpg", "image2.jpg", 5)


def test_extract_duplicates_between_two_datasets():
    test_paths = ["path/to/directory1", "path/to/directory2"]
    test_paths2 = ["path/to/directory3", "path/to/directory4"]
    chunk_length = 1
    num_processes = 8

    with patch("ufoid.core.get_images_from_paths", return_value=["image1.jpg", "image2.jpg"]):
        with patch("ufoid.core.divide_in_chunks", return_value=[["image1.jpg"], ["image2.jpg"]]):
            with patch(
                "ufoid.core.parallel_paths_to_hashes",
                return_value=np.zeros((2, 256), dtype=np.uint64),
            ):
                with patch(
                    "ufoid.core.extract_duplicates_within_a_list_of_paths", return_value=[]
                ):
                    with patch(
                        "ufoid.core.extract_duplicates_between_two_lists_of_paths",
                        return_value=[("image1.jpg", "image2.jpg", 5)],
                    ):
                        duplicate_pairs = extract_duplicates_between_two_datasets(
                            test_paths,
                            test_paths2,
                            chunk_length,
                            num_processes,
                            distance_threshold=11,
                            logger_level=logging.ERROR,
                        )

                        assert len(duplicate_pairs) == 4
                        assert duplicate_pairs[0] == ("image1.jpg", "image2.jpg", 5)


test_data_paths = ["data/toy_data/imgs_someunique_somedupli"]


@pytest.fixture
def temp_dir():
    with TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_generate_n_random_images(temp_dir):
    n = 10
    size = 100
    path = temp_dir
    num_processes = 1
    duplicates_percentage = 0.3  # 30% duplicates

    generate_n_random_images(n, size, path, num_processes, duplicates_percentage)

    # Verify that the correct number of images is generated
    generated_images = os.listdir(path)
    assert len(generated_images) == n

    # Verify that the expected number of duplicate images is generated
    duplicates, _ = extract_duplicates_within_a_dataset(
        [path], logger_level=logging.ERROR, distance_threshold=1
    )
    assert len(duplicates) == math.comb(int(n * duplicates_percentage), 2)


def test_generate_random_image(temp_dir):
    generate_random_image(1, 100, temp_dir)

    # Verify that the generated image file exists
    image_path = os.path.join(temp_dir, "1_random_img.png")
    assert os.path.exists(image_path)


def test_generate_duplicate_images(temp_dir):
    generate_duplicate_images(1, 100, temp_dir)

    # Verify that the generated duplicate image file exists
    image_path = os.path.join(temp_dir, "1_random_img.png")
    assert os.path.exists(image_path)


def test_duplicate_detection():
    # Get images from the data paths
    images = sorted(get_images_from_paths(test_data_paths))

    # Calculate duplicates using each function
    result_no_chunk, _ = extract_duplicates_within_a_dataset_no_chunk(images)
    result_with_chunk, _ = extract_duplicates_within_a_dataset_with_chunks(images, chunk_length=6)
    result_auto_chunk, _ = extract_duplicates_within_a_dataset(test_data_paths, chunk_length=6)
    result_auto_chunk2, _ = extract_duplicates_within_a_dataset(test_data_paths, chunk_length=1000)

    # Compare the results
    assert result_no_chunk == result_with_chunk
    assert result_no_chunk == result_auto_chunk
    assert result_with_chunk == result_auto_chunk
    assert result_with_chunk == result_auto_chunk2


def test_duplicate_detection_between_datasets():
    # Prepare data paths for the reference and new datasets
    images = sorted(get_images_from_paths(test_data_paths))

    # Calculate duplicates using each function
    result_no_chunk = extract_duplicates_between_two_datasets_no_chunk(images, images)
    result_with_chunk = extract_duplicates_between_two_datasets_with_chunks(
        images, images, chunk_length=6
    )
    result_auto_chunk = extract_duplicates_between_two_datasets(
        test_data_paths, test_data_paths, chunk_length=6
    )
    result_auto_chunk2 = extract_duplicates_between_two_datasets(
        test_data_paths, test_data_paths, chunk_length=1000
    )

    # Compare the results
    assert result_no_chunk == result_with_chunk
    assert result_no_chunk == result_auto_chunk
    assert result_with_chunk == result_auto_chunk
    assert result_with_chunk == result_auto_chunk2
