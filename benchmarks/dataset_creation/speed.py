import multiprocessing
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from tqdm.auto import tqdm


def generate_random_image(i, size, path):
    """Generate a random image with random pixel values and save it to the specified path.

    Args:
        i (int): The index for the image.
        size (int): The image size (width and height).
        path (str): The directory to save the generated image.
    """
    imarray = np.random.rand(size, size, 3) * 255
    im = Image.fromarray(imarray.astype("uint8")).convert("RGB")
    name = f"{i}_random_img.png"
    im.save(os.path.join(path, name))


def generate_duplicate_images(i, size, path):
    """Generate a duplicate image with constant pixel values and save it to the specified path.

    Args:
        i (int): The index for the image.
        size (int): The image size (width and height).
        path (str): The directory to save the generated image.
    """
    im_array = np.ones((size, size, 3), dtype="uint8") * 255
    im = Image.fromarray(im_array)
    name = f"{i}_random_img.png"
    im.save(os.path.join(path, name))


def generate_n_random_images(
    n=100,
    size=100,
    path="",
    num_processes=None,
    duplicates_percentage=0.0,
):
    """Generate random and duplicate images and save them to the specified path.

    Args:
        n (int, optional): The total number of images to generate. Default is 100.
        size (int, optional): The size of the images (width and height). Default is 100.
        path (str, optional): The directory to save the generated images. Default is an empty string.
        num_processes (int, optional): The number of parallel processes to use for generation. Default is (CPU cores - 1).
        duplicates_percentage (float, optional): The percentage of images to be duplicates. Default is 0.0.
    """
    os.makedirs(path, exist_ok=True)
    duplicate_count = int(n * duplicates_percentage)
    indexes = list(range(1, n + 1))
    random.shuffle(indexes)

    if num_processes is None:
        num_processes = multiprocessing.cpu_count() - 1
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        random_args = [(i, size, path) for i in indexes[: n - duplicate_count]]
        _ = list(
            tqdm(
                executor.map(generate_random_image, *zip(*random_args)),
                total=n - duplicate_count,
                colour="green",
            )
        )

    # Generate duplicate images in another ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        duplicate_args = [(i, size, path) for i in indexes[n - duplicate_count:]]
        _ = list(
            tqdm(
                executor.map(generate_duplicate_images, *zip(*duplicate_args)),
                total=duplicate_count,
                colour="green",
            )
        )
