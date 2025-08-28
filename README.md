# Stocks Analytics 2025

<img width="1853" height="543" alt="capa" src="https://github.com/user-attachments/assets/44d66af9-b8ec-44d8-8ee5-bade8f72415a" />

Projeto desenvolvido como parte do **Stock Market Analytics Zoomcamp (2025)**. O objetivo é aplicar técnicas de engenharia de dados, análise financeira e machine learning para criar um pipeline reprodutível que vai desde a ingestão de dados de mercado até a simulação de estratégias de trading com automação. Este projeto é estritamente educacional e não constitui recomendação de investimento.

## Sumário

1. [Contextualização do Projeto](#contextualização-do-projeto)  
2. [Objetivos](#objetivos)  
3. [Arquitetura do Pipeline](#arquitetura-do-pipeline)  
4. [Desenvolvimento e Entregáveis](#desenvolvimento-e-entregáveis)  
   - [Ingestão de Dados](#1-ingestão-de-dados)  
   - [Feature Engineering](#2-feature-engineering)  
   - [Modelagem](#3-modelagem)  
   - [Estratégias & Sinais](#4-estratégias--sinais)  
   - [Backtest](#5-backtest)  
   - [Automação & Deployment](#6-automação--deployment)  
5. [Estrutura do Repositório](estrutura-do-repositório)  
6. [Configuração e Instalação](#configuração-e-instalação)  
   - [Ambiente Virtual](#ambiente-virtual)  
   - [Docker](#docker)  
   - [Cron Jobs](#cron-jobs)  
   - [Cloud Deployment](#cloud-deployment)  
7. [Execução (Makefile & CLI)](#execução)  
8. [Métricas e Resultados](#métricas-e-resultados)  
9. [Projeto Contempla](#projeto-contempla)  
10. [Boas Práticas](#boas-práticas)  
11. [Contribuição](#contribuição)  
12. [Bibliografia e Referências](#bibliografia-e-referências)  
13. [Conclusão](#conclusão)  

## Contextualização do Projeto

O mercado financeiro é repleto de dados complexos e dinâmicos: preços de ações, indicadores macroeconômicos, balanços corporativos e notícias em tempo real. Transformar esses dados em insights requer um pipeline robusto que combine:
   - Engenharia de Dados para ingestão e tratamento.
   - Machine Learning para previsão de tendências e classificação de sinais.
   - Simulação de Estratégias para avaliar performance sob custos e riscos reais.
   - Automação para rodar processos diariamente em escala.
Este projeto implementa exatamente esse ciclo, usando Python, bibliotecas open-source, boas práticas de MLOps e ferramentas modernas de automação.

## Objetivos

   - Construir um pipeline reprodutível, modular e automatizado para análise de ações.
   - Coletar dados de múltiplas fontes (Yahoo Finance, Stooq, Tiingo).
   - Extrair e gerar features relevantes (retornos, volatilidade, indicadores técnicos, calendário).
   - Treinar múltiplos modelos de ML para prever retornos/direções.
   - Simular estratégias (long-only, long-short, stop-loss/take-profit, top-k).
   - Comparar resultados com benchmarks (SPY, ACWI, equal-weight).
   - Automatizar execução com Docker, Cron, CI/CD e integrações externas (broker API, Telegram).

## Arquitetura do Pipeline

O fluxo segue 7 camadas principais:

- Ingestão de Dados
     - Coleta incremental via yfinance, Stooq e Tiingo.
     - Armazenamento em Parquet (análises locais) e SQLite (persistência).

- Feature Engineering
     - Retornos (1d, 5d, 21d), volatilidade, indicadores técnicos (RSI, MACD, Bollinger).
     - 36+ features documentadas em reports/features_catalog.md.

- Modelagem
     - DecisionTree, RandomForest (com tuning).
     - Regras custom por probabilidade.
     - Novos modelos: XGBoost com calibração.

- Sinais de Trading
     - Conversão de previsões em sinais (+1 long, -1 short, 0 flat).
     - Estratégias: long-only, long-short, top-k, probabilidade.

- Backtesting
     - Vetorial: simulações rápidas.
     - Exata (iterativa): reinvestimento, SL/TP, gestão de risco.
     - Métricas: CAGR, Sharpe, Sortino, Drawdown, Rolling returns.

- Relatórios & Métricas
     - reports/backtest_results.md + gráficos de equity curve e drawdown.
      
- Automação & Deploy
     - src/app/run_all.py: pipeline fim-a-fim.
     - Docker/Compose para empacotamento.
     - GitHub Actions com agendamento diário.
     - Notificações via e-mail e Telegram.
     - Broker API (Alpaca, paper trading).

## Desenvolvimento e Entregáveis

1️⃣ Ingestão de Dados

- Scripts: src/data/ingest.py, src/data/alt_provider.py.
- Fontes: Yahoo Finance, Stooq, Tiingo.
- Incremental, com persistência em Parquet + SQLite.

2️⃣ Feature Engineering

- Scripts: src/features/build_features.py.
- Features: retornos, volatilidade, indicadores técnicos, calendário, interações.
- Catálogo: reports/features_catalog.md.

3️⃣ Modelagem

- Modelos básicos: DecisionTree, RandomForest.
- Tuning: GridSearchCV, RandomizedSearchCV.
- Novo modelo: XGBoost.
- Scripts: scripts/train_trees.py, scripts/train_xgb.py.

4️⃣ Estratégias & Sinais

- Script: src/strategy/generate_signals.py.
- Estratégias suportadas:
- Long-only (prob threshold).
- Long-short (top-bottom).
- Probabilidade calibrada.
- Top-k seleções.

5️⃣ Backtest

Scripts:

- src/backtest/vector_backtester.py
- src/backtest/exact_simulator.py
- Métricas: CAGR, Sharpe, Sortino, Drawdown, Volatilidade, WinRate, Turnover.
- Relatórios: reports/backtest_results.md.

6️⃣ Automação & Deployment

- Pipeline: src/app/run_all.py.
- Dockerfile e docker-compose.yml.
- Cron jobs (ops/cron.example).
- CI/CD com .github/workflows/run_pipeline.yml.
- Integrações: broker API (Alpaca), Telegram bot.

## Estrutura do Repositório

```
stocks-analytics-2025/
├── data/                  # Dados (não versionados no Git)
├── models/                # Modelos salvos
├── notebooks/             # Prototipagem
├── reports/               # Relatórios (EDA, modelagem, backtests)
├── src/                   # Código principal
│   ├── data/              # Ingestão
│   ├── features/          # Feature engineering
│   ├── models/            # Treinamento
│   ├── strategy/          # Geração de sinais
│   ├── backtest/          # Simulação
│   ├── app/               # Pipelines
│   ├── notify/            # Notificações
│   ├── brokers/           # Integrações externas
│   └── utils/             # Funções auxiliares
├── tests/                 # Testes unitários (pytest)
├── scripts/               # Executáveis de treino/modelos
├── .github/workflows/     # CI/CD
├── docker/                # Dockerfile
├── ops/                   # Automação
├── requirements.txt
├── pyproject.toml
├── docker-compose.yml
├── Makefile
└── README.md
```

## Configuração e Instalação

### Ambiente Virtual

<img width="922" height="151" alt="image" src="https://github.com/user-attachments/assets/7bc44a1d-e69e-4b3b-a9ce-46e187d84d8d" />

### Docker

<img width="607" height="63" alt="image" src="https://github.com/user-attachments/assets/0b101a80-a26a-4dbf-b5fa-685539023199" />

### Cron Jobs

<img width="632" height="70" alt="image" src="https://github.com/user-attachments/assets/c6397245-35be-44ae-87e3-aceaaf3c6993" />

## Cloud Deployment

- GitHub Actions já configurado (.github/workflows/run_pipeline.yml).
- Pode ser estendido para AWS (ECS + S3) ou GCP (Cloud Run + BigQuery).

## Execução

<img width="626" height="322" alt="image" src="https://github.com/user-attachments/assets/0c5ef5a9-2bb1-46c4-a15c-4946717382cc" />

## Métricas e Resultados

<img width="2378" height="1980" alt="output" src="https://github.com/user-attachments/assets/f085948e-1a3c-4b52-9868-25ee5c79bf19" />

- Modelos:
    - MAE, MAPE (regressão).
    - Accuracy, F1, AUC (classificação).

- Backtest:
    - CAGR, Sharpe, Sortino.
    - Max Drawdown, Volatilidade, WinRate.
    - Comparação com benchmark (SPY, ACWI).
 
<img width="1400" height="612" alt="image" src="https://github.com/user-attachments/assets/59b6f166-bcbc-443f-a4c1-4fd045759ca5" />

 ## Projeto Contempla

- Modelagem: DT, RF, XGBoost; tuning; regras custom.
- EDA & Features: dataset unificado, 36+ features, correlações.
- Fontes: YF + Stooq/Tiingo; dataset >1M linhas.
- Problema: README claro, novas definições (long-short, SL/TP, crypto).
- Simulação: vetorial + exata, reinvestimento, comparativos vs benchmark.
- Automação: Docker, cron, CI/CD, incremental storage.
- Extras: broker API, Telegram bot, modularidade.

## Boas Práticas

- Black + Ruff (lint).
- Testes unitários (pytest).
- Pre-commit hooks.
- CI/CD com GitHub Actions.
- Segredos em .env.
- Documentação modular (README, reports, notebooks).

## Contribuição

Consulte CONTRIBUTING.md.

## Bibliografia e Referências

- The Man Who Solved the Market – Gregory Zuckerman
- Unknown Market Wizards – Jack Schwager
- The Tao of Trading – Simon Ree
- PythonInvest Blog & Workshops
- Simply Wall St Insights
- CNBC Investing, FT Unhedged
- Yahoo Finance, Tiingo, Stooq

## Conclusão

Este projeto cobre todos os critérios da avaliação do Stock Market Analytics Zoomcamp (2025):

- Ingestão → ✅

- Features → ✅

- Modelagem → ✅

- Estratégias & Backtest → ✅

- Automação & Deployment → ✅

- Reprodutibilidade & Documentação → ✅

Demonstra como aplicar Data Engineering + Machine Learning + Finance de forma reprodutível, modular e extensível para análise de mercados financeiros.
