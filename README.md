# Stocks Analytics 2025

<img width="1672" height="571" alt="6" src="https://github.com/user-attachments/assets/76b76843-6f1f-4122-9384-28de1e88b30f" />

Projeto desenvolvido como parte do **Stock Market Analytics Zoomcamp (2025)**. O objetivo é aplicar técnicas de engenharia de dados, análise financeira e machine learning para criar um pipeline reprodutível que vai desde a ingestão de dados de mercado até a simulação de estratégias de trading com automação. Este projeto é estritamente educacional e não constitui recomendação de investimento.

## Sumário

### 🔹 Contextualização do Projeto
- [Visão Geral](#visão-geral)  
- [Objetivos](#objetivos)  
- [Informações do Curso](#informações-do-curso)  
- [Materiais de Apoio](#materiais-de-apoio)  

### 🔹 Desenvolvimento
- [Entregáveis do Projeto](#entregáveis-do-projeto)  
  - [Ingestão de Dados](#1-ingestão-de-dados)  
  - [Feature Engineering](#2-feature-engineering)  
  - [Modelagem](#3-modelagem)  
  - [Estratégias & Sinais](#4-estratégias--sinais)  
  - [Backtest](#5-backtest)  
  - [Automação & Deployment](#6-automação--deployment)  
- [Estrutura do Repositório](#estrutura-do-repositório)  

### 🔹 Configuração e Execução
- [Configuração & Instalação](#configuração--instalação)  
  - [Docker](#docker)  
  - [Cron](#cron)  
- [Execução (Makefile)](#execução-makefile)  
- [Configuração (.env)](#configuração-env)  

### 🔹 Avaliação e Resultados
- [Métricas & Relatórios](#métricas--relatórios)  
- [Resultados](#resultados)  

### 🔹 Boas Práticas e Automação
- [Qualidade & Boas Práticas](#qualidade--boas-práticas)  
- [Automação e Deploy](#automação-e-deploy)  

### 🔹 Colaboração
- [Contribuição](#contribuição)  

### 🔹 Encerramento
- [Conclusão](#conclusão)  

## Visão Geral

- Este projeto implementa um pipeline de ponta a ponta para análise de ações, com foco em:
- Ingestão de dados financeiros (via Yahoo Finance API – yfinance)
- Tratamento e geração de features (retornos, volatilidade, indicadores técnicos, calendário)
- Modelagem analítica (regressão para retornos futuros, classificação para direção de mercado)
- Geração de sinais de trading (long/short)
- Backtest e simulação (estratégias equally-weighted, custos de transação, métricas)
- Automação e deployment (pipeline diário, Docker, cron, notificações por e-mail)
- Reprodutibilidade e colaboração (CI/CD, pre-commit, testes unitários, notebooks e relatórios)

## Objetivos

- Coletar e armazenar dados de ações/índices (OHLCV, dividendos, metadados).
- Realizar EDA, limpeza, *feature engineering* (incl. indicadores técnicos).
- Treinar modelos de previsão (regressão e/ou classificação de direção).
- Definir estratégias de trading e **backtestar** com métricas financeiras.
- Automatizar a rotina diária (cron/Airflow), empacotar (Docker) e validar (CI + testes).
- Demonstrar aplicação prática de ciência de dados no mercado financeiro.
- Criar um projeto reprodutível, organizado e extensível.
- Atender os critérios de avaliação do curso, incluindo ingestão, features, modelagem, backtest, automação e documentação.

## Informações do Curso

- **Cohort 2025**: início em **19 maio 2025 (segunda)**, **18:30 GMT+1 (Dublin)**  
- **Transmissão**: YouTube do **PythonInvest** (playlist 2025)  
- **Pré-lançamento (overview + Q&A)**: **14 abril 2025**  
- **Projeto**:  
  - Semanas 1–2 → desenvolvimento  
  - Semana 3 → *peer review*

> Modo **self-paced**: todo material é aberto; siga o syllabus semanal e use o Slack para suporte.

## Materiais de Apoio

**Workshops**
- *Economics and Automation Workshop: Building a Data Pipeline for Economic Insights*  
- *Predicting Financial Time-Series*

**Pré-leitura / Notícias e Análises**
- PythonInvest — Financial News Feed  
- PythonInvest — Blog (artigos analíticos)  
- Simply Wall St — Market Insights  
- CNBC — Investing  
- FT — *Unhedged* (podcast/artigos)  
- Yahoo Finance

**Livros**
- *The Trading Game: A Confession*  
- *Unknown Market Wizards* (latest edition)  
- *The Man Who Solved the Market*  
- *The Tao of Trading*  
- *The Unlucky Investor’s Guide to Options Trading*

## Entregáveis do Projeto

1️⃣ Ingestão de Dados

- Fonte: Yahoo Finance via yfinance.
- Incremental, persistência em Parquet e SQLite.
- Configuração via .env.
- Script: src/data/ingest.py.

2️⃣ Feature Engineering

- Retornos (1d, 5d, 21d).
- Volatilidade rolling.
- Features de calendário (dia da semana, mês).
- Indicadores técnicos (SMA, EMA, RSI, MACD, Bollinger).
- Targets (regressão: retorno 5d, classificação: direção 5d).
- Script: src/features/build_features.py.

3️⃣ Modelagem

- Regressão Linear (previsão de retornos).
- Logistic Regression (direção positiva/negativa).
- Split temporal (TimeSeriesSplit).
- Métricas: MAE, MAPE, Accuracy, F1, AUC.
- Artefatos salvos em models/.
- Script: src/models/train.py.

4️⃣ Estratégias & Sinais

- Conversão de previsões em sinais (+1 long, -1 short, 0 flat).
- Geração em data/signals/.
- Script: src/strategy/generate_signals.py.

5️⃣ Backtest

- Simulação next-day.
- Custos e slippage configuráveis.
- Métricas: CAGR, Sharpe, Volatilidade anualizada, Max Drawdown.
- Saídas: positions.parquet, equity.parquet, summary.json.
- Script: src/backtest/simulator.py.

6️⃣ Automação & Deployment

- Pipeline diário end-to-end: src/app/run_daily_pipeline.py.
- Previsões diárias: src/app/predict_daily.py.
- Notificação por e-mail (opcional): src/notify/email_report.py.
- Deployment via Docker e Compose.
- Automação via cron (ops/cron.example)

<img width="285" height="581" alt="8" src="https://github.com/user-attachments/assets/a74f0d57-377c-4699-a1f0-8e3788fa5318" />

## Estrutura do Repositório

```
stocks-analytics-zoomcamp-2025/
│
├── data/                  # Dados (não versionados no Git)
│   ├── raw/               # Dados brutos (download da API)
│   ├── processed/         # Dados tratados (parquet/csv/sqlite)
│   ├── analytics/         # Features e previsões
│   ├── signals/           # Sinais de trading
│   └── backtests/         # Resultados de simulação
│
├── models/                # Modelos treinados (.pkl)
│
├── notebooks/             # Notebooks exploratórios
│   ├── 01_eda_features.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_trading_strategy.ipynb
│
├── reports/               # Relatórios
│   ├── eda_summary.md
│   ├── modeling_report.md
│   ├── backtest_results.md
│   └── img/               # Gráficos
│
├── src/                   # Código principal
│   ├── data/              # Ingestão
│   ├── features/          # Features
│   ├── models/            # Treinamento
│   ├── strategy/          # Sinais
│   ├── backtest/          # Simulador & métricas
│   ├── app/               # Pipeline & previsões
│   ├── notify/            # Notificações (e-mail)
│   ├── utils/             # Funções utilitárias
│   └── config.py          # Configurações globais (.env)
│
├── tests/                 # Testes unitários (pytest)
│
├── docker/                # Dockerfile
├── ops/                   # Automação
├── scripts/               # Scripts auxiliares (ex.: run_local.sh)
│
├── .github/workflows/     # CI/CD (lint, format, tests)
│
├── .env.example           # Exemplo de variáveis
├── requirements.txt       # Dependências
├── pyproject.toml         # Configuração do projeto
├── docker-compose.yml     # Orquestração via Compose
├── Makefile               # Atalhos (lint, test, run, etc.)
├── CONTRIBUTING.md        # Guia de contribuição
└── README.md              # Este documento
```

## Configuração & Instalação

> As dependências ficam no `pyproject.toml`.

Clone o repositório e crie um ambiente virtual:

```
git clone https://github.com/nathadriele/stocks-analytics-2025
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

#### Docker

```
docker compose up --build
```

#### Cron

Agende via ops/cron:

```
30 19 * * * /path/to/scripts/run_local.sh >> /path/to/logs/cron.log 2>&1
```

## Execução (Makefile)

Os principais comandos são centralizados no Makefile:

```
make ingest       # coleta/atualiza dados de mercado
make features     # gera e salva features
make train        # treina modelos e salva artefatos
make signals      # converte previsões em sinais
make backtest     # simula estratégias e exporta relatórios
make run_all      # pipeline fim-a-fim
make test         # testes unitários (pytest)
```

## Configuração (.env)

Crie um .env baseado em .env.example:

```
DATA_START=2015-01-01
TICKERS=AAPL,MSFT,SPY
DB_PATH=storage/app.db
# API_KEY=... (se usar provedores pagos)
```

## Métricas & Relatórios

- Modelos: MAE/MAPE (regressão); AUC/F1/Accuracy (classificação)
- Backtest: CAGR, Sharpe, Sortino, Max Drawdown, Volatilidade, WinRate, #Trades, Turnover;
gráficos de equity curve e drawdown em reports/img/

## Qualidade & Boas Práticas

- Reprodutibilidade: make run_all executa o fluxo completo
- Time-series split; sem look-ahead; custos e slippage parametrizados
- Testes (pytest) e lint/format (ruff, black) no CI
- Segredos fora do repositório (.env, nunca versionar)
- Documentação clara (README + reports/*)

## Resultados

- EDA: gráficos de preços, retornos, correlações e indicadores técnicos.
- Modelos: métricas de previsão (MAE, MAPE, AUC, F1).
- Trading Strategies: relatórios de backtest com:
   - CAGR, Sharpe, Sortino
   - Drawdown máximo
   - Taxa de acerto e nº de trades
   - Comparação com benchmark (buy & hold)
 
<img width="844" height="562" alt="7" src="https://github.com/user-attachments/assets/883e565f-bd20-4001-89ee-1b80dea851d5" />

## Automação e Deploy

- **Banco de dados**: SQLite (storage/app.db)
- **Agendamento**: cron jobs ou DAG simples no Airflow
- **Docker**: container para rodar pipeline end-to-end
- **CI/CD**: lint, testes e build automatizados com GitHub Actions

## Conclusão

Este projeto cobre todos os critérios de avaliação do curso:

- Ingestão → ✅
- Features → ✅
- Modelagem → ✅
- Estratégias & Backtest → ✅
- Automação & Deployment → ✅
- Reprodutibilidade & Documentação → ✅

Demonstra aplicação prática de ciência de dados no mercado financeiro, com código aberto, modular, testável e pronto para extensão futura.

## Contribuição

Siga estas orientações para manter o projeto limpo, reprodutível e fácil de evoluir:  

1. **Fork & Branch**  
   - Faça um fork do repositório.  
   - Crie uma branch a partir de `main`:  
     ```bash
     git checkout -b feature/nome-da-feature
     ```

2. **Configuração do ambiente**  
   - Configure o ambiente virtual e instale dependências com:  
     ```bash
     python -m venv .venv
     source .venv/bin/activate   # Windows: .venv\Scripts\activate
     pip install -e ".[dev]"
     pre-commit install
     ```

3. **Padrões de código**  
   - Utilize **Black** (formatação), **Ruff** (lint) e **Pre-commit**.  
   - Antes de commitar:  
     ```bash
     make format   # aplica Black
     make lint     # roda Ruff
     make test     # pytest
     ```

4. **Mensagens de commit**  
   - Escreva no **imperativo** e de forma clara:  
     - `Add RSI feature to build_features`  
     - `adicionando RSI`  

5. **Pull Request**  
   - Abra um PR descrevendo **o que** foi alterado, **por que** e **como testar**.  
   - Garanta que os testes passam no CI.  

6. **Testes**  
   - Todo novo código deve incluir ou atualizar testes em `tests/`.  
   - Rodar testes localmente:  
     ```bash
     make test
     ```

7. **Boas práticas**  
   - Não versionar segredos (`.env`).  
   - Não incluir dados brutos em Git (apenas `.gitkeep`).  
   - Documentar novas funções/módulos com docstrings.  

---

Para detalhes adicionais, veja [`CONTRIBUTING.md`](CONTRIBUTING.md).

