import glob
import multiprocessing
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image, ImageEnhance
from tqdm.auto import tqdm


def generate_virgin_images(logger, base_path="data"):
    """Generate or check the existence of virgin images for the quality benchmark dataset.

    Args:
        logger: A logger object for logging messages.
        base_path (str, optional): The base directory for storing the dataset. Default is "data".
    """
    out_data_path = os.path.join(base_path, "quality_benchmark")
    virgin_path = os.path.join(out_data_path, "virgin")
    if not os.path.isdir(virgin_path) or len(glob.glob(os.path.join(virgin_path, "*"))) < 1000:
        os.makedirs(virgin_path, exist_ok=True)
        idx = 0
        for image in glob.glob(os.path.join(base_path, "image_dedup_benchmark", "*")):
            if "_copy" not in image:
                shutil.copy(image, os.path.join(virgin_path, f"{str(idx)}.png"))
                idx += 1
                if idx == 1000:
                    break

    logger.info("Virgin images generated or already existing")


def manipulate_and_save_image(image, manipulation, intensity, path):
    """Apply image manipulation and save the manipulated image to the specified path.

    Args:
        image (str): The path to the input image.
        manipulation (function): The image manipulation function to apply.
        intensity (float): The intensity or strength of the manipulation.
        path (str): The path to save the manipulated image.
    """
    virgin = Image.open(image)
    manipulator = manipulation(virgin)
    manipulated_image = manipulator.enhance(intensity)
    manipulated_image.save(path)


def generate_manipulated_images(
    logger, base_path="data", num_processes=multiprocessing.cpu_count() - 1
):
    """Generate or check the existence of manipulated images for the quality benchmark dataset.

    Args:
        logger: A logger object for logging messages.
        base_path (str, optional): The base directory for storing the dataset. Default is "data".
        num_processes (int, optional): The number of parallel processes to use. Default is (CPU cores - 1).
    """
    out_data_path = os.path.join(base_path, "quality_benchmark")
    virgin_images = glob.glob(os.path.join(out_data_path, "virgin", "*"))
    if (
        len(glob.glob(os.path.join(out_data_path, "manipulated", "*", "*")))
        < len(virgin_images) * 16
    ):
        manipulations = {
            "contrast": ImageEnhance.Contrast,
            "sharpness": ImageEnhance.Sharpness,
            "color": ImageEnhance.Color,
            "brightness": ImageEnhance.Brightness,
        }

        manip_intensity = {"neg_heavy": 0.5, "neg_soft": 0.8, "pos_soft": 1.2, "pos_heavy": 1.5}

        for man_name, _ in manipulations.items():
            for int_name, _ in manip_intensity.items():
                man_path = os.path.join(out_data_path, "manipulated", f"{man_name}_{int_name}")
                os.makedirs(man_path, exist_ok=True)

        args_list = []
        for image in virgin_images:
            idx = os.path.basename(image)
            for man_name, manipulation in manipulations.items():
                for int_name, intensity in manip_intensity.items():
                    path = os.path.join(
                        out_data_path, "manipulated", f"{man_name}_{int_name}", idx
                    )
                    args_list.append((image, manipulation, intensity, path))

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            _ = list(
                tqdm(
                    executor.map(manipulate_and_save_image, *zip(*args_list)),
                    total=len(args_list),
                    colour="green",
                )
            )

    logger.info("Manipulated images generated or already existing")


def resize_and_save_image(image, factor, path):
    """Resize an image and save it to the specified path.

    Args:
        image (str): The path to the input image.
        factor (float): The resizing factor by which to scale the image.
        path (str): The path to save the resized image.
    """
    virgin = Image.open(image)
    resized_image = virgin.resize(
        (int(np.ceil(virgin.size[0] * factor)), int(np.ceil(virgin.size[1] * factor)))
    )
    resized_image.save(path)


def stretch_and_save_image(image, x_factor, y_factor, path):
    """Stretch an image by different factors along the X and Y axes and save it to the specified
    path.

    Args:
        image (str): The path to the input image.
        x_factor (float): The stretching factor for the X-axis.
        y_factor (float): The stretching factor for the Y-axis.
        path (str): The path to save the stretched image.
    """
    virgin = Image.open(image)
    resized_image = virgin.resize(
        (int(np.ceil(virgin.size[0] * x_factor)), int(np.ceil(virgin.size[1] * y_factor)))
    )
    resized_image.save(path)


def generate_resized_and_stretched_images(
    logger, base_path="data", num_processes=multiprocessing.cpu_count() - 1
):
    """Generate or check the existence of resized and stretched images for the quality benchmark
    dataset.

    Args:
        logger: A logger object for logging messages.
        base_path (str, optional): The base directory for storing the dataset. Default is "data".
        num_processes (int, optional): The number of parallel processes to use. Default is (CPU cores - 1).
    """
    out_data_path = os.path.join(base_path, "quality_benchmark")
    virgin_images = glob.glob(os.path.join(out_data_path, "virgin", "*"))
    if (
        len(glob.glob(os.path.join(out_data_path, "manipulated", "*", "*")))
        < len(virgin_images) * 18
    ):
        os.makedirs(os.path.join(out_data_path, "manipulated", "resized"))
        args_list = []
        for image in virgin_images:
            idx = os.path.basename(image)
            factor = random.random() * 2
            path = os.path.join(out_data_path, "manipulated", "resized", idx)
            args_list.append((image, factor, path))

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            _ = list(
                tqdm(
                    executor.map(resize_and_save_image, *zip(*args_list)),
                    total=len(args_list),
                    colour="green",
                )
            )

        os.makedirs(os.path.join(out_data_path, "manipulated", "stretched"))
        args_list = []
        for image in virgin_images:
            idx = os.path.basename(image)
            factor = random.random() * 2
            factor2 = random.random() * 2
            path = os.path.join(out_data_path, "manipulated", "stretched", idx)
            args_list.append((image, factor, factor2, path))

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            _ = list(
                tqdm(
                    executor.map(stretch_and_save_image, *zip(*args_list)),
                    total=len(args_list),
                    colour="green",
                )
            )
    logger.info("Resized and stretched images generated or already existing")


def prepare_quality_benchmark_dataset(logger, base_path="data"):
    """Prepare the quality benchmark dataset by generating or checking the existence of virgin,
    manipulated, resized, and stretched images.

    Args:
        logger: A logger object for logging messages.
        base_path (str, optional): The base directory for storing the dataset. Default is "data".
    """
    generate_virgin_images(logger, base_path=base_path)
    generate_manipulated_images(logger, base_path=base_path)
    generate_resized_and_stretched_images(logger, base_path=base_path)
