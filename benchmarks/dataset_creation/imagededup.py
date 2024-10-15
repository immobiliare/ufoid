import glob
import os
import shutil
import sys
import zipfile

import wget


def bar_progress(current, total, width=80):
    """Display a progress bar for a download operation.

    Args:
        current (int): The current progress value.
        total (int): The total value representing completion.
        width (int, optional): The width of the progress bar. Default is 80.
    """
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (
        current / total * 100,
        current,
        total,
    )
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def download_ukbench_dataset(
    url="https://archive.org/download/ukbench/ukbench.zip", base_path="data"
):
    """Download and extract the UKBench dataset from the specified URL.

    Args:
        url (str, optional): The URL of the dataset. Default is the UKBench dataset URL.
        base_path (str, optional): The base directory for storing the dataset. Default is "data".
    """
    if not (
        os.path.isdir(os.path.join(base_path, "ukbench"))
        or os.path.isdir(os.path.join(base_path, "image_dedup_benchmark"))
    ):
        if not os.path.isfile(os.path.join(base_path, "ukbench.zip")):
            wget.download(url, bar=bar_progress, out=os.path.join(base_path, "ukbench.zip"))
        with zipfile.ZipFile(os.path.join(base_path, "ukbench.zip"), "r") as zip_ref:
            zip_ref.extractall(os.path.join(base_path, "ukbench"))
        os.remove(os.path.join(base_path, "ukbench.zip"))


def prepare_imagededup_dataset(base_path="data"):
    """Prepare the Image Deduplication benchmark dataset.

    Args:
        base_path (str, optional): The base directory for storing the dataset. Default is "data".
    """
    out_data_path = os.path.join(base_path, "image_dedup_benchmark")
    # remove unnecessary files
    for fileordir in glob.glob(os.path.join(base_path, "ukbench", "*")):
        if fileordir not in [os.path.join(base_path, "ukbench", "full"), out_data_path]:
            if os.path.isdir(fileordir):
                shutil.rmtree(fileordir)
            if os.path.isfile(fileordir):
                os.remove(fileordir)

    if not os.path.isdir(out_data_path):
        # take only one image from group of different views as explained in imagededup_docs
        os.makedirs(out_data_path)
        for image in glob.glob(os.path.join(base_path, "ukbench", "full", "*")):
            image_idx = int(os.path.basename(image).replace(".jpg", "").replace("ukbench", ""))
            if image_idx % 4 == 0:
                shutil.copy(image, os.path.join(out_data_path, os.path.basename(image)))
        shutil.rmtree(os.path.join(base_path, "ukbench"))
        # clone each image as explained in imagededup_docs
        for image in glob.glob(os.path.join(out_data_path, "*")):
            shutil.copy(
                image,
                os.path.join(out_data_path, os.path.basename(image).replace(".jpg", "_copy.jpg")),
            )


def download_and_prepare_imagededup_dataset(logger, base_path="data"):
    """Download and prepare the Image Deduplication benchmark dataset.

    Args:
        logger: A logger object for logging messages.
        base_path (str, optional): The base directory for storing the dataset. Default is "data".
    """
    download_ukbench_dataset(base_path=base_path)
    prepare_imagededup_dataset(base_path=base_path)
    logger.info("Image dedup dataset generated or already existing")
