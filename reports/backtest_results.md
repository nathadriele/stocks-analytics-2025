# Backtest Results

Este relatório documenta os resultados das simulações de estratégias de trading.

---

## 1. Objetivo
Avaliar a performance de diferentes estratégias aplicadas aos sinais gerados pelos modelos.

---

## 2. Estratégias Testadas
- Buy & Hold (benchmark)
- Momentum Long-Only
- Mean Reversion
- Market-Neutral (pairs trading)
- Dividend strategy *(opcional)*

---

## 3. Metodologia
- Execução: sinais convertidos em posições (`long`, `short`, `flat`)
- Custos de transação: *(ex.: 0.05% por trade)*
- Slippage: *(parâmetro configurável)*
- Rebalanceamento: diário / semanal

---

## 4. Métricas Calculadas
- CAGR (Compound Annual Growth Rate)
- Sharpe Ratio
- Sortino Ratio
- Volatilidade Anualizada
- Máx. Drawdown
- Win Rate (% de trades vencedores)
- Nº total de trades
- Turnover

---

## 5. Resultados
*(tabela consolidada com métricas por estratégia + benchmark)*

---

## 6. Visualizações
- Equity curve (curva de crescimento do portfólio)
- Gráfico de drawdown
- Distribuição de retornos por trade

*(figuras exportadas para `reports/img/`)*

---

## 7. Conclusões
- Estratégia mais robusta: *(a preencher)*
- Riscos observados: *(a preencher)*
- Possíveis melhorias: otimização de parâmetros, inclusão de stop-loss/take-profit
