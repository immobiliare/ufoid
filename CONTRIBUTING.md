# Contributing

This document describes how to work and contribute on this project.

- [Contributing](#contributing)
  - [1 How to clone the project](#1-how-to-clone-the-project)
  - [2 Requirements](#2-requirements)
  - [3 Configuration](#3-configuration)
  - [4 Installation](#4-installation)
    - [4.1 `pyenv` usage (skip if the installed python version is the same required by the project)](#41-pyenv-usage-skip-if-the-installed-python-version-is-the-same-required-by-the-project)
    - [4.2 Create virtualenv and install requirements](#42-create-virtualenv-and-install-requirements)
  - [5 Run procedure](#5-run-procedure)
  - [6 Test procedure](#6-test-procedure)
  - [7 CI Structure](#7-ci-structure)
    - [7.1 Runners](#71-runners)
    - [7.2 Variables](#72-variables)
    - [7.3 Test](#73-test)
    - [7.4 Badges](#74-badges)
  - [8 Code style and conventions](#8-code-style-and-conventions)

## 1 How to clone the project

In order to get this project, execute the following command

```console
git clone https://github.com/immobiliare/ufoid.git
```

## 2 Requirements

This project requires the following technologies:

- Python 3.9.2 (as shown in the `.python-version` file)
- PIP + packages in `requirements.txt`

You can try to install all requirements using the following command

```console
make env
```

or if you have pyenv installed

```console
make env_pyenv
```

## 3 Configuration

`ufoid/config/config.yaml` allows you to customize various aspects of the duplicate detection process. Here are some key parameters you can modify:

- `num_processes`: Number of processes for parallel execution.
- `chunk_length`: The length of each chunk for chunk-based processing. See below for more information.
- `new_paths`: List of directory paths containing the new dataset for duplicate detection (automatically recursive).
- `old_paths`: List of directory paths containing the old dataset for comparison with the new dataset (automatically recursive).
- `check_with_itself`: Boolean flag to indicate whether to check for duplicates within the new dataset.
- `check_with_old_data`: Boolean flag to indicate whether to check for duplicates between the new and old datasets.
- `txt_output`: Boolean flag to indicate whether to save duplicate information to the output file.
- `txt_output_file`: Path to the output file where duplicate information will be saved.
- `delete_duplicates`: Boolean flag to indicate whether to delete duplicate images from the dataset.
- `create_folder_with_no_duplicates`: Boolean flag to indicate whether to create a folder with non-duplicate images.
- `new_folder`: Path to the folder where non-duplicate images will be stored.
- `distance_threshold`: The distance threshold for considering images as duplicates. 10 is optimal for our use case, since it allows to get all the exact duplicate (also with some resilience to minor manipulations on images, while avoiding collisions. See https://docs.google.com/document/d/16DS-Z-SHKtmTzQikxCO4SwRJU0cHS_-TVkf9UAKYRZA/edit#heading=h.uybo5cpys4eefor an extensive study on this.


## 4 Installation

In order to guarantee the proper operation of the application, the recommended python version is 3.9.2.

### 4.1 `pyenv` usage (skip if the installed python version is the same required by the project)

If the default python version on the system is different from the recommended one, you can install the required version using the `pyenv` tool available [here](https://github.com/pyenv/pyenv#installation).
Once `pyenv` is installed, you can install the required python version executing

```console
pyenv install 3.9.2
```

and you can activate the recommended version for the current shell by

```console
pyenv shell 3.9.2
```

### 4.2 Create virtualenv and install requirements

In order to create a clean environment for the execution of the application, a new virtualenv should be created inside the current folder, using the command

```console
python3 -m venv venv
```

A new folder named `venv` will be created in `.`

In order to activate the virtualenv, execute

```console
source venv/bin/activate
```

and install python requirements executing

```console
pip install -r requirements.txt
```
A different approach consists in using the Makefile by running from the project root the command

```console
make
```

This operation will:

- create the venv;
- update pip to the latest version;
- install the requirements;
- install the git hook.

## 5 Run procedure

Start the script using the following command:

```console
python -m ufoid
```

## 6 Test procedure

Inside the project you can find functional tests created to verify the correctness of implemented features.
It is possible to start the execution of all `pytest` tests (unit + integration) running:

```console
pytest
```

In order to see the coverage for unit tests, execute

```console
pytest test/unit --cov
```

## 7 Code style and conventions

This project makes use of the standard Python coding style: PEP-8
