# Exploratory Data Analysis (EDA) Summary

Este relatório documenta a análise exploratória realizada nos dados de mercado coletados.

---

## 1. Descrição Geral do Dataset
- Período analisado: *(a definir com base em DATA_START no .env)*
- Ativos incluídos: *(tickers definidos em .env)*
- Número de observações: *(será preenchido após ingestão)*
- Campos principais: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`

---

## 2. Estatísticas Descritivas
*(inserir tabelas com `describe()` e resumos de retornos, volatilidade, etc.)*

---

## 3. Distribuições
- Histogramas de retornos diários
- Boxplots por ativo
- Kernel Density Estimates (KDE)

---

## 4. Correlações
- Heatmap de correlação entre ativos
- Relação entre volume e variação de preço

---

## 5. Indicadores Técnicos
- SMA/EMA
- RSI
- MACD
- Bollinger Bands

---

## 6. Visualizações Salvas
As imagens geradas durante a EDA foram exportadas para `reports/img/`.

---

## 7. Observações Iniciais
- *(pontos relevantes encontrados na análise serão descritos aqui)*
