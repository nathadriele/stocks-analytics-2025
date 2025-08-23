# Contribuindo

Valeu por contribuir com o **Stock Market Analytics — Zoomcamp 2025**!  
Este guia resume como configurar o ambiente, padrões de código e fluxo de trabalho.

## 1) Ambiente

- Python **3.11**
- Crie a venv e instale o projeto:
  ```bash
  python -m venv .venv
  source .venv/bin/activate          # Windows: .venv\Scripts\activate
  pip install -U pip
  pip install -e ".[dev]"
  pre-commit install
