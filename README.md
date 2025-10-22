# Python Brasil 2025 Workshop

Repositório com notebooks para Python Brasil 2025

## Setup

Escolha seu método preferido para configurar o ambiente de desenvolvimento:

### Poetry
- Instale o Poetry: `curl -sSL https://install.python-poetry.org | python3 -`
- Instale as dependências: `poetry install`
- Inicie o shell: `poetry shell`
- Inicie o Jupyter: `jupyter lab`

### pip
- Crie um ambiente virtual: `python3 -m venv .venv`
- Ative-o: `source .venv/bin/activate`
- Instale as dependências: `pip install -r requirements.txt`
- Inicie o Jupyter: `jupyter lab`

### uv
- Instale o uv: `pip install uv`
- Sincronize as dependências: `uv sync`
- Ative o ambiente: `source .venv/bin/activate`
- Inicie o Jupyter: `jupyter lab`

> Todas as opções requerem Python 3.11 ou 3.12. Os notebooks estão na pasta `notebooks/`; abra-os no Jupyter Lab durante a sessão.
