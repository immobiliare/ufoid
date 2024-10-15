import multiprocessing
import os
import shutil

import pandas as pd
from omegaconf import OmegaConf

from ufoid import core, utils

logger = utils.get_logger(__name__)


def main(cfg_path="ufoid/config/config.yaml"):
    cfg = OmegaConf.load(cfg_path)

    logger.info("Starting the duplicate detection process")
    logger.info(
        f"Number of new images: {len(core.get_images_from_paths(cfg.new_paths))} found in paths: {(', '.join(p for p in cfg.new_paths))}"
    )
    duplicates_itself, duplicates_with_old = [], []
    hashes_new_data = None

    if cfg.num_processes == "max":
        cfg.num_processes = multiprocessing.cpu_count()
        logger.info(
            f"Number of processes was not provided, defaulting to logical cpu count: {cfg.num_processes}"
        )

    if cfg.check_with_itself:
        logger.info("Checking for duplicates within the new dataset")
        duplicates_itself, hashes_new_data = core.extract_duplicates_within_a_dataset(
            cfg.new_paths,
            chunk_length=cfg.chunk_length,
            num_processes=cfg.num_processes,
            distance_threshold=cfg.distance_threshold,
        )
        logger.info(f"Number of  duplicates within itself found: {len(duplicates_itself)}")

    if cfg.check_with_old_data:
        logger.info("Checking for duplicates between the new and old datasets")
        logger.info(
            f"Number of old images that will be compared with new images: {len(core.get_images_from_paths(cfg.old_paths))} found in paths: {(', '.join(p for p in cfg.old_paths))}"
        )
        duplicates_with_old = core.extract_duplicates_between_two_datasets(
            cfg.old_paths,
            cfg.new_paths,
            chunk_length=cfg.chunk_length,
            num_processes=cfg.num_processes,
            distance_threshold=cfg.distance_threshold,
            hashes_new=hashes_new_data,
        )
        logger.info(f"Number of duplicates with old images found: {len(duplicates_with_old)}")

    duplicates = duplicates_itself + duplicates_with_old
    logger.info(f"Number of duplicates found: {len(duplicates)}")

    if cfg.csv_output:
        logger.info(f"Saving duplicate information to the CSV file {cfg.csv_output_file}")

        df = pd.DataFrame(duplicates, columns=["path1", "path2", "diff"])

        # Save the DataFrame to a CSV file
        df.to_csv(cfg.csv_output_file, index=False)

    to_remove = {couple[1] for couple in duplicates}
    logger.info(f"Number of unique duplicate images found: {len(to_remove)}")

    if cfg.create_folder_with_no_duplicates:
        logger.info(f"Creating a folder with non-duplicate images in {cfg.new_folder}")
        os.makedirs(cfg.new_folder, exist_ok=True)
        all_imgs = core.get_images_from_paths(cfg.new_paths)
        to_keep = set(all_imgs) - to_remove
        logger.info(f"Copyng {len(to_keep)} in {cfg.new_folder}")
        for img in to_keep:
            shutil.copy(img, os.path.join(cfg.new_folder, os.path.basename(img)))

    if cfg.delete_duplicates:
        logger.info(f"Deleting {len(to_remove)} duplicate images")
        for img in to_remove:
            os.remove(img)

    logger.info("Duplicate detection process completed.")


if __name__ == "__main__":
    main()
