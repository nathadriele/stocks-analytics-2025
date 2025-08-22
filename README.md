# Stocks Analytics 2025

Projeto desenvolvido como parte do Stock Market Analytics Zoomcamp ( 2025). O objetivo é aplicar técnicas de engenharia de dados, análise financeira e machine learning para criar um pipeline reprodutível que vai desde a ingestão de dados de mercado até a simulação de estratégias de trading com automação. Este projeto é estritamente educacional e não constitui recomendação de investimento.

Sumário
- [Objetivos](#-objetivos)
- [Informações do Curso](#-informações-do-curso)
- [Checklist Pré-Curso](#-checklist-précurso)
- [Materiais de Apoio](#-materiais-de-apoio)
- [Syllabus → Entregáveis do Projeto](#-syllabus--entregáveis-do-projeto)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Configuração & Instalação](#-configuração--instalação)
- [Execução (Makefile)](#-execução-makefile)
- [Configuração (.env)](#-configuração-env)
- [Métricas & Relatórios](#-métricas--relatórios)
- [Qualidade & Boas Práticas](#-qualidade--boas-práticas)
- [Etiquetas para Dúvidas (Slack)](#-etiquetas-para-dúvidas-slack)
- [Licença](#-licença)

## Objetivos

- Coletar e armazenar dados de ações/índices (OHLCV, dividendos, metadados).
- Realizar EDA, limpeza, *feature engineering* (incl. indicadores técnicos).
- Treinar modelos de previsão (regressão e/ou classificação de direção).
- Definir estratégias de trading e **backtestar** com métricas financeiras.
- Automatizar a rotina diária (cron/Airflow), empacotar (Docker) e validar (CI + testes).

## Informações do Curso

- **Cohort 2025**: início em **19 maio 2025 (segunda)**, **18:30 GMT+1 (Dublin)**  
- **Transmissão**: YouTube do **PythonInvest** (playlist 2025)  
- **Pré-lançamento (overview + Q&A)**: **14 abril 2025**  
- **Projeto**:  
  - Semanas 1–2 → desenvolvimento  
  - Semana 3 → *peer review*

> Modo **self-paced**: todo material é aberto; siga o syllabus semanal e use o Slack para suporte.

## Checklist Pré-Curso

- [ ] Inscrever-se no **PythonInvest** (newsletter/comunicados)  
- [ ] Entrar no **Slack do DataTalks.Club**  
- [ ] Participar do canal `#course-stocks-analytics-zoomcamp`  
- [ ] Entrar no **Telegram** de anúncios do curso  
- [ ] Acompanhar vídeos no **YouTube/PythonInvest**  
- [ ] Ler **FAQ** e **Syllabus curto** no site oficial  

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

### Módulo 1 — Introduction & Data Sources
- Decisões orientadas por dados; panorama de investimentos; risco × retorno  
- Setup (Colab/local) e download inicial via APIs financeiras  
- Escolha de API (critérios, limites, custo)  
**Entregáveis:**  
- `src/data/ingest.py` (CLI de ingestão)  
- Persistência em `data/processed/` (parquet) e/ou `storage/app.db` (SQLite)  
- README seção “Data Sources & Setup”

### Módulo 2 — Working with the Data (Pandas)
- Núcleo: **NumPy, Pandas, Matplotlib, Seaborn, Plotly Express**  
- Tipos (numérico, string, data), *dummies*, *feature generation* (dow/dom/mês, retornos, janelas móveis)  
- Indicadores técnicos (**TA-Lib** ou `ta`: SMA/EMA, RSI, MACD, BB, ATR)  
- *Data cleaning*, *joins*, EDA e correlações  
**Entregáveis:**  
- `src/features/build_features.py`  
- `notebooks/01_eda_features.ipynb`  
- `reports/eda_summary.md` (+ figuras em `reports/img/`)

### Módulo 3 — Analytical Modeling
- Hipóteses e previsão de séries (tendência/sazonalidade)  
- Regressão (p.ex. Linear/Ridge/Lasso) e Classificação (Logistic/RandomForest)  
- (Opcional) exemplo *neural*  
- *TimeSeriesSplit* / *walk-forward*, métricas (MAE/MAPE, AUC/F1)  
**Entregáveis:**  
- `src/models/train.py` (pipelines + artefatos em `models/`)  
- `reports/modeling_report.md`

### Módulo 4 — Trading Strategy & Simulation
- Da previsão ao sinal: *thresholds*, *position sizing*, risco e taxas  
- Exemplos: **buy & hold**, portfólio diversificado, **market-neutral**, *mean reversion*, dividendos, *penny stocks*  
- (Talvez) opções básicas  
- Simulação: execução “próxima abertura”, custos, *slippage*  
**Entregáveis:**  
- `src/strategy/generate_signals.py`  
- `src/backtest/{simulator.py,metrics.py}`  
- `reports/backtest_results.md` (equity curve, drawdown, tabela de métricas)

### Módulo 5 — Deployment & Automation
- Notebooks → scripts `.py` (CLIs)  
- Armazenamento persistente (arquivos/SQLite + intro a SQL)  
- Automação: **cron** e/ou **Apache Airflow** (DAG diária)  
- (Talvez) e-mail automático com previsões/trades/PnL  
**Entregáveis:**  
- `src/app/{predict_daily.py,run_daily_pipeline.py}`  
- `docker/Dockerfile`, `.github/workflows/ci.yml`  
- README com instruções de execução/automação

## Estrutura do Repositório

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
    └── notify/           # Notificações
tests/                    # Testes unitários
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























