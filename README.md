# littlebooktimeseries

Timeseries in 3 hours

## Environment setup

Choose one of the options below to get the workshop notebooks ready.

### Poetry
- Install Poetry: `curl -sSL https://install.python-poetry.org | python3 -`
- Install dependencies: `poetry install`
- Start the shell: `poetry shell`
- Launch Jupyter: `jupyter lab`

### pip
- Create a virtual environment: `python3 -m venv .venv`
- Activate it: `source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Launch Jupyter: `jupyter lab`

### uv
- Install uv: `pip install uv`
- Sync dependencies: `uv sync`
- Activate the environment: `source .venv/bin/activate`
- Launch Jupyter: `jupyter lab`

> All options require Python 3.11 or 3.12. The notebooks live under `notebooks/`; open them in Jupyter Lab during the session.
