#!/usr/bin/env bash
# ===============================================
# Runner local do Stock Market Analytics Pipeline
# ===============================================

set -euo pipefail

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_DIR"

if [ -d ".venv" ]; then
  echo "[run_local] Ativando ambiente virtual..."
  source .venv/bin/activate
fi

RETRAIN=${RETRAIN:-false}
SIGNAL_COL=${SIGNAL_COL:-signal_cls}
FEE_BPS=${FEE_BPS:-5}
SLIPPAGE_BPS=${SLIPPAGE_BPS:-2}
SKIP_BACKTEST=${SKIP_BACKTEST:-false}

echo "[run_local] Iniciando pipeline em $(date)"
python -m src.app.run_daily_pipeline \
  --retrain $RETRAIN \
  --signal-col $SIGNAL_COL \
  --fee-bps $FEE_BPS \
  --slippage-bps $SLIPPAGE_BPS \
  --skip-backtest $SKIP_BACKTEST

echo "[run_local] Finalizado com sucesso em $(date)"
