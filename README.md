# Stocks Analytics 2025

Projeto desenvolvido como parte do Stock Market Analytics Zoomcamp ( 2025). O objetivo é aplicar técnicas de engenharia de dados, análise financeira e machine learning para criar um pipeline reprodutível que vai desde a ingestão de dados de mercado até a simulação de estratégias de trading com automação. Este projeto é estritamente educacional e não constitui recomendação de investimento.

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

Clone o repositório e crie um ambiente virtual:

```
git clone <repo-url>
cd stocks-analytics-zoomcamp-2025
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Instale as dependências (serão definidas em pyproject.toml):

```
pip install -e .
pre-commit install
```

Configure o arquivo .env baseado em .env.example:

```
DATA_START=2015-01-01
TICKERS=AAPL,MSFT,SPY
DB_PATH=storage/app.db
```

## Como Executar

Os principais comandos são centralizados no Makefile:

```
make ingest       # coleta dados do mercado
make features     # gera features e salva em parquet
make train        # treina modelos e salva artefatos
make signals      # gera sinais de trading
make backtest     # simula estratégias
make run_all      # executa o pipeline completo
make test         # roda testes unitários
```

## Resultados Esperados

- EDA: gráficos de preços, retornos, correlações e indicadores técnicos.
- Modelos: métricas de previsão (MAE, MAPE, AUC, F1).
- Trading Strategies: relatórios de backtest com:
   - CAGR, Sharpe, Sortino
   - Drawdown máximo
   - Taxa de acerto e nº de trades
   - Comparação com benchmark (buy & hold)

## Automação e Deploy

- **Banco de dados**: SQLite (storage/app.db)
- **Agendamento**: cron jobs ou DAG simples no Airflow
- **Docker**: container para rodar pipeline end-to-end
- **CI/CD**: lint, testes e build automatizados com GitHub Actions























