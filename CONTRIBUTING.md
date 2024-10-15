# Contributing to UFOID

First off, thank you for considering contributing to UFOID! Your contributions help make this project better for everyone.

## Table of Contents

- [Getting Started](#getting-started)
- [Submitting Issues](#submitting-issues)
- [Pull Requests](#pull-requests)
- [Coding Guidelines](#coding-guidelines)
- [Running Tests](#running-tests)

## Getting Started

1. **Fork the repository**: Click the "Fork" button at the top right of the repository page.
2. **Clone your fork**: Clone the forked repository to your local machine.
   ```bash
   git clone https://github.com/<your-username>/ufoid.git
   cd ufoid
   ```
3. **Create a virtual environment and install dependencies**: Set up a virtual environment to install dependencies.
   ```bash
   make
   source venv/bin/activate
   ```

## Submitting Issues

If you encounter any bugs or have feature requests, please create an issue in the GitHub repository. When creating an issue, please include:

- A clear and descriptive title.
- A detailed description of the issue or request.
- Steps to reproduce the issue (if applicable).
- Any relevant logs, screenshots, or other information.

## Pull Requests

Before submitting a pull request, please ensure you have followed these steps:

1. **Fork the repository** and **clone your fork** to your local machine.
2. **Create a new branch** from the `main` branch.
   ```bash
   git checkout -b my-feature-branch
   ```
3. **Make your changes**: Implement your changes in the new branch.
4. **Run tests**: Ensure all tests pass by running the test suite.
5. **Commit your changes**: Write clear and concise commit messages.
   ```bash
   git add .
   git commit -m "Add my new feature"
   ```
6. **Push to your fork**: Push your changes to your forked repository.
   ```bash
   git push origin my-feature-branch
   ```
7. **Open a pull request**: Go to the original repository and open a pull request. Please provide a detailed description of your changes and link any related issues.

## Coding Guidelines

To maintain code quality, please adhere to the following guidelines:

- Follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code.
- Write clear and concise code with comments where necessary.
- Keep the codebase DRY (Don't Repeat Yourself) and modular.
- Ensure all new features are covered by tests.
- Update documentation as needed when making changes.

## Running Tests and lint

Before submitting your pull request, ensure that all tests pass and that the code is linted:

1. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate
   ```
2. **Run the tests** using pytest:
   ```bash
   pytest --cov
   ```
3. **Run the tests** using pytest:
   ```bash
   pre-commit run --all-files
   ```

Thank you for your contributions!

## Contact

If you have any questions or need further assistance, feel free to reach out to the maintainers.

## Contributors

- @EgonFerri
- @lorenzomassimiani
