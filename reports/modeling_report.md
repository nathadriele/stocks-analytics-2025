# Modeling Report

Este relatório documenta os experimentos de modelagem realizados sobre os dados financeiros.

---

## 1. Objetivo
Avaliar modelos de regressão e classificação para prever retornos futuros ou direção do mercado.

---

## 2. Conjunto de Dados
- Features utilizadas: *(geradas em `src/features/build_features.py`)*
- Target:
  - **Regressão:** retorno futuro em janelas de 5, 21 dias
  - **Classificação:** 1 (retorno positivo), 0 (retorno negativo)
- Split temporal: *(ex.: treino 2015–2020, teste 2021–2024)*

---

## 3. Modelos Testados
- Regressão Linear, Ridge, Lasso
- Regressão Logística
- Random Forest
- (Opcional) Redes Neurais simples

---

## 4. Métricas
### Regressão
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² (opcional)

### Classificação
- Accuracy
- F1-score
- AUC-ROC
- Matriz de Confusão

---

## 5. Resultados
*(tabelas comparando modelos e hiperparâmetros)*

---

## 6. Análise de Erros
- Períodos em que os modelos performaram mal
- Possíveis causas (volatilidade, choques de mercado, falta de features explicativas)

---

## 7. Conclusões
- Melhor modelo de regressão: *(a preencher)*
- Melhor modelo de classificação: *(a preencher)*
- Próximos passos: tuning, novas features, ensembles
