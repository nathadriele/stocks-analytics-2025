# Stocks Analytics 2025

Projeto desenvolvido como parte do Stock Market Analytics Zoomcamp (Cohort 2025). O objetivo é aplicar técnicas de engenharia de dados, análise financeira e machine learning para criar um pipeline reprodutível que vai desde a ingestão de dados de mercado até a simulação de estratégias de trading com automação. Este projeto é estritamente educacional e não constitui recomendação de investimento.

## Objetivos do Projeto

- Coletar e armazenar dados de ações e índices de mercado.
- Realizar análise exploratória e engenharia de features.
- Treinar modelos de previsão (regressão e classificação).
- Desenvolver e simular estratégias de trading.
- Automatizar o pipeline com agendamento, CI/CD e Docker.

## Estrutura Básica

```
├── README.md
├── pyproject.toml
├── .gitignore
├── .pre-commit-config.yaml
├── Makefile
├── .env.example
├── data/                 # Dados brutos, processados e resultados
├── models/               # Modelos treinados
├── reports/              # Relatórios e imagens
├── storage/              # Banco SQLite
├── notebooks/            # EDA e análises exploratórias
└── src/                  # Código-fonte organizado por módulos
    ├── data/             # Ingestão
    ├── features/         # Engenharia de features
    ├── models/           # Treinamento
    ├── strategy/         # Estratégias de trading
    ├── backtest/         # Simulação e métricas
    ├── app/              # Automação e execução diária
    └── notify/           # Notificações (opcional)
tests/                    # Testes unitários
```

## Configuração do Ambiente

1. Clone o repositório e crie um ambiente virtual:

git clone <repo-url>
cd stocks-analytics-zoomcamp-2025
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

























