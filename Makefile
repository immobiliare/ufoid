PROJECT = ufoid
VENV_NAME = venv

env-pyenv:
	uv python pin 3.9
	uv venv
	uv sync
