PROJECT = ufoid
VENV_NAME = venv

env-pyenv:
	pyenv install 3.9.2 --skip-existing
	pyenv local 3.9.2
	$(HOME)/.pyenv/shims/python -m venv venv
	venv/bin/pip install -U pip flake8 pre-commit
	venv/bin/pip install -r requirements.txt
	venv/bin/pip install -r requirements-benchmark.txt
	venv/bin/pip install -r requirements-test.txt
