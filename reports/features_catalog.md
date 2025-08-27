# Features Catalog (Stocks Analytics 2025)

Este catálogo descreve as **features** usadas no pipeline, com **nome canônico**, **definição**, **janela**, **tipo**, **motivação** e **nota de segurança (leakage)**.

> Convenções:
> - `close_t`: preço de fechamento no dia *t* (por *ticker*).
> - `ret_k`: retorno simples k-dias = `close_t / close_{t-k} - 1`.
> - Todas as janelas são **exclusivas do presente** (rolling passado) e **shiftadas** apropriadamente na engenharia de features (sem olhar o futuro).

---

## 1) Conjuntos de colunas (grupos)

- **TO_PREDICT**  
  - `target_up_5d` (binária: 1 se `ret_5d_fwd > 0`, 0 caso contrário)  
  - `ret_5d` (contínua: para referência/diagnóstico)

- **NUMERIC** (principais numéricas contínuas; ver tabela abaixo)  
- **DUMMIES** (se aplicável)  
  - `dow_0`…`dow_4` (segunda…sexta)  
  - `month_1`…`month_12`

> Observação: Para **DecisionTree/RandomForest**, `dow`/`month` podem ser usados como **inteiros**; para regressões lineares/logísticas, prefira **dummies**.

---

## 2) Tabela de Features (preço/indicadores/calendário)

| Nome | Definição / Fórmula | Janela | Tipo | Motivação | Leakage |
|---|---|---:|---|---|---|
| `ret_1d` | `(close_t / close_{t-1}) - 1` | 1 | Retorno | Momentum curto prazo | Não |
| `ret_5d` | `(close_t / close_{t-5}) - 1` | 5 | Retorno | Tendência semanal | Não |
| `ret_21d` | `(close_t / close_{t-21}) - 1` | 21 | Retorno | Mensal (≈ 1 mês) | Não |
| `log_ret_1d` | `ln(close_t) - ln(close_{t-1})` | 1 | Retorno (log) | Normaliza amplitude | Não |
| `ret_lag_1` | `ret_1d` shiftado | — | Lag | Autocorrelação curtíssima | Não |
| `ret_lag_3` | `ret_1d` shiftado 3 | — | Lag | Persistência curta | Não |
| `ret_lag_5` | `ret_1d` shiftado 5 | — | Lag | Semana anterior | Não |
| `ret_lag_10` | `ret_1d` shiftado 10 | — | Lag | Quinzena | Não |
| `vol_21d` | `std(ret_1d, 21)` | 21 | Volatilidade | Risco mensal | Não |
| `vol_63d` | `std(ret_1d, 63)` | 63 | Volatilidade | Risco trimestral | Não |
| `sma_5` | `mean(close, 5)` | 5 | Tendência | Média curta | Não |
| `sma_10` | `mean(close, 10)` | 10 | Tendência | Média curta | Não |
| `sma_20` | `mean(close, 20)` | 20 | Tendência | Média curta/média | Não |
| `sma_50` | `mean(close, 50)` | 50 | Tendência | Suporte/resistência | Não |
| `sma_200` | `mean(close, 200)` | 200 | Tendência | Tendência de longo prazo | Não |
| `ema_5` | `EMA(close, 5)` | 5 | Tendência | Peso recente | Não |
| `ema_10` | `EMA(close, 10)` | 10 | Tendência | Peso recente | Não |
| `ema_20` | `EMA(close, 20)` | 20 | Tendência | Peso recente | Não |
| `ema_50` | `EMA(close, 50)` | 50 | Tendência | Peso recente | Não |
| `rsi_14` | `RSI(close, 14)` | 14 | Oscilador | Sobrecompra/venda | Não |
| `macd` | `EMA12(close) - EMA26(close)` | 12/26 | Oscilador | Momentum cruzamentos | Não |
| `macd_signal` | `EMA9(macd)` | 9 | Oscilador | Sinal do MACD | Não |
| `macd_hist` | `macd - macd_signal` | — | Oscilador | Força do sinal | Não |
| `bb_upper_20` | `SMA20 + 2*STD20` | 20 | Bandas | Volatilidade | Não |
| `bb_lower_20` | `SMA20 - 2*STD20` | 20 | Bandas | Volatilidade | Não |
| `bb_width_20` | `(bb_upper_20 - bb_lower_20) / sma_20` | 20 | Bandas | Largura (vol) | Não |
| `zscore_20` | `(close - sma_20) / std_20` | 20 | Normalização | Desvios da média | Não |
| `mom_10` | `close_t - close_{t-10}` | 10 | Momentum | Força direcional | Não |
| `roc_10` | `(close_t / close_{t-10}) - 1` | 10 | Momentum | Variação relativa | Não |
| `roll_max_252` | `max(close, 252)` | 252 | Extremos | Alta de 52s | Não |
| `roll_min_252` | `min(close, 252)` | 252 | Extremos | Baixa de 52s | Não |
| `pct_from_52w_high` | `(close - roll_max_252) / roll_max_252` | 252 | Extremos | Distância do topo | Não |
| `pct_from_52w_low` | `(close - roll_min_252) / roll_min_252` | 252 | Extremos | Distância do fundo | Não |
| `sma20_slope` | `sma_20(t) - sma_20(t-5)` | 5/20 | Tendência | Inclinação média | Não |
| `ema_ratio_10_50` | `ema_10 / ema_50 - 1` | 10/50 | Tendência | Cruzamento normalizado | Não |
| `price_sma20_ratio` | `close / sma_20 - 1` | 20 | Tendência | *Overextension* | Não |
| `dow` | Dia da semana (`0`=seg … `4`=sex) | — | Calendário | Padrões semanais | Não |
| `month` | Mês do ano (`1`..`12`) | — | Calendário | Sazonalidade | Não |

> **Total listado:** 36 features.

---

## 3) Segurança contra *leakage*
- Todas as janelas/funções são calculadas **apenas com dados passados** e **shiftadas** se necessário.  
- Targets (`target_up_5d` / `ret_5d`) são derivados **com `shift(-5)`** somente no alvo, **nunca** nas features.  
- No treino/teste, usamos **split temporal** (últimos 20% para teste) e, em CV, **TimeSeriesSplit**.

---

## 4) Mapeamento por grupo

- **NUMERIC (sugestão para RF/DT):**  
  `ret_1d, ret_5d, ret_21d, log_ret_1d, ret_lag_1, ret_lag_3, ret_lag_5, ret_lag_10, vol_21d, vol_63d, sma_5, sma_10, sma_20, sma_50, sma_200, ema_5, ema_10, ema_20, ema_50, rsi_14, macd, macd_signal, macd_hist, bb_upper_20, bb_lower_20, bb_width_20, zscore_20, mom_10, roc_10, roll_max_252, roll_min_252, pct_from_52w_high, pct_from_52w_low, sma20_slope, ema_ratio_10_50, price_sma20_ratio, dow, month`

- **DUMMIES (opcional, para modelos lineares):**  
  `dow_0..dow_4`, `month_1..month_12`

---

## 5) Uso no treino (via CLI)

Você pode passar a lista de features com:

```bash
python -m scripts.train_trees \
  --data-path data/analytics/features.parquet \
  --features-file src/features/feature_list.txt \
  --model both
