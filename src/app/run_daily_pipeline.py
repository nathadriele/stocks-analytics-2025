"""
Orquestra o pipeline diário de ponta a ponta.

Etapas:
1) Ingestão de dados (incremental por padrão)
2) Geração de features
3) (Opcional) Treinamento de modelos
4) Previsões diárias
5) Geração de sinais
6) Backtest (rolling) com métricas

Uso:
    python -m src.app.run_daily_pipeline

Com opções:
    python -m src.app.run_daily_pipeline \
        --retrain false \
        --signal-col signal_cls \
        --fee-bps 5 --slippage-bps 2 \
        --skip-backtest false
"""

from __future__ import annotations

import traceback
import typer

from src import config

# Importa as funções "run" dos módulos
from src.data.ingest import run as ingest_run
from src.features.build_features import run as features_run
from src.models.train import run as train_run
from src.app.predict_daily import run as predict_run
from src.strategy.generate_signals import run as signals_run
from src.backtest.simulator import run as backtest_run

app = typer.Typer(add_completion=False, help="Pipeline diário (end-to-end)")


def _safe_step(step_name: str, func, **kwargs) -> None:
    """
    Executa uma etapa do pipeline com tratamento de erro e logs legíveis.
    """
    typer.secho(f"[pipeline] >>> {step_name}", fg=typer.colors.CYAN, bold=True)
    try:
        func(**kwargs)
        typer.secho(f"[pipeline] OK: {step_name}", fg=typer.colors.GREEN)
    except SystemExit as se:  # módulos com Typer podem levantar SystemExit
        if int(getattr(se, "code", 0)) != 0:
            typer.secho(f"[pipeline] FAIL: {step_name} (SystemExit code={se.code})", fg=typer.colors.RED)
            raise
        else:
            # SystemExit(0) → sucesso
            typer.secho(f"[pipeline] OK: {step_name}", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"[pipeline] ERRO em {step_name}: {e}", fg=typer.colors.RED)
        traceback.print_exc()
        raise


@app.command("run")
def run(
    retrain: bool = typer.Option(
        False,
        "--retrain",
        help="Se true, re-treina modelos antes de gerar previsões/sinais.",
    ),
    signal_col: str = typer.Option(
        "signal_cls",
        "--signal-col",
        "-s",
        help="Coluna de sinal para o backtest (ex.: signal_cls, signal_reg).",
    ),
    fee_bps: float = typer.Option(
        5.0,
        "--fee-bps",
        help="Custos (fee) em bps aplicados quando há mudança de posição.",
    ),
    slippage_bps: float = typer.Option(
        2.0,
        "--slippage-bps",
        help="Slippage em bps aplicado em mudanças de posição.",
    ),
    skip_backtest: bool = typer.Option(
        False,
        "--skip-backtest",
        help="Se true, não executa o backtest ao final.",
    ),
):
    """
    Roda o pipeline fim-a-fim com defaults seguros.
    """
    typer.secho("======== PIPELINE DIÁRIO — Stock Market Analytics ========", fg=typer.colors.MAGENTA, bold=True)

    # 1) Ingestão incremental (usa defaults do .env)
    _safe_step("Ingestão de dados", ingest_run)

    # 2) Geração de features
    _safe_step("Geração de features", features_run, to_db=False)

    # 3) (Opcional) Treinamento
    if retrain:
        _safe_step("Treinamento de modelos (reg + cls)", train_run, mode="both")
    else:
        typer.secho("[pipeline] pulando treinamento (retrain=false)", fg=typer.colors.YELLOW)

    # 4) Previsões do dia (último dia disponível)
    _safe_step("Previsões diárias", predict_run, date=None)

    # 5) Sinais (usa ambos se existirem; aqui default = both)
    _safe_step("Geração de sinais (reg + cls)", signals_run, mode="both")

    # 6) Backtest
    if not skip_backtest:
        _safe_step(
            "Backtest",
            backtest_run,
            signal_col=signal_col,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
    else:
        typer.secho("[pipeline] pulando backtest (skip_backtest=true)", fg=typer.colors.YELLOW)

    typer.secho("======== PIPELINE CONCLUÍDO COM SUCESSO ========", fg=typer.colors.GREEN, bold=True)


if __name__ == "__main__":
    app()
