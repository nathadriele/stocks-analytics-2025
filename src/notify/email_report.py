"""
Envio de e-mail com resumo diário do pipeline.

- Lê métricas do backtest (data/backtests/summary.json)
- (Opcional) anexa imagens (ex.: equity curve em reports/img/)
- Usa credenciais do .env (SMTP) para envio

Uso:
    python -m src.notify.email_report --subject "Daily Market Report"
"""

from __future__ import annotations

import json
import os
import smtplib
import typer
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

from src import config


app = typer.Typer(add_completion=False, help="Envio de e-mail com resumo do backtest")


def _build_body(summary_path: Path) -> str:
    if not summary_path.exists():
        return "Resumo não encontrado. Rode o backtest antes de enviar o e-mail."

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    lines = [
        "Daily Market Report — Stock Market Analytics",
        "",
        f"Sinal: {data.get('signal_col')}",
        f"Custos: fee={data.get('fee_bps')} bps | slippage={data.get('slippage_bps')} bps",
        "",
        "Métricas:",
        f"- CAGR:        {metrics.get('CAGR')}",
        f"- Sharpe:      {metrics.get('Sharpe')}",
        f"- Vol (ann):   {metrics.get('Vol_Ann')}",
        f"- MaxDD:       {metrics.get('MaxDrawdown')}",
        f"- Período:     {metrics.get('Start')} → {metrics.get('End')}  (N={metrics.get('N_Days')})",
        "",
        "Este e-mail é gerado automaticamente para fins educacionais.",
    ]
    return "\n".join(lines)


def _attach_if_exists(msg: EmailMessage, file_path: Path, maintype: str, subtype: str) -> None:
    if file_path.exists():
        with open(file_path, "rb") as f:
            msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=file_path.name)


@app.command("run")
def run(
    subject: str = typer.Option("Daily Market Report — Stock Analytics", "--subject"),
    attach_equity: bool = typer.Option(True, "--attach-equity/--no-attach-equity"),
    equity_img_path: Optional[str] = typer.Option(None, "--equity-img", help="Caminho de imagem de equity curve"),
):
    """
    Monta e envia o e-mail com base nas variáveis .env.
    """
    smtp_server = config.SMTP_SERVER or os.getenv("SMTP_SERVER")
    smtp_port = int(config.SMTP_PORT or os.getenv("SMTP_PORT") or 587)
    user = config.EMAIL_USER or os.getenv("EMAIL_USER")
    password = config.EMAIL_PASSWORD or os.getenv("EMAIL_PASSWORD")
    to_addr = config.EMAIL_TO or os.getenv("EMAIL_TO")

    if not all([smtp_server, smtp_port, user, password, to_addr]):
        typer.secho("[email] Variáveis SMTP/EMAIL ausentes. Configure .env antes.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    summary_path = config.BACKTESTS_DIR / "summary.json"
    body = _build_body(summary_path)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr
    msg.set_content(body)

    if attach_equity:
        if equity_img_path:
            path = Path(equity_img_path)
        else:
            path = config.REPORTS_DIR / "img" / "equity_curve.png"
        _attach_if_exists(msg, path, maintype="image", subtype="png")

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(user, password)
            server.send_message(msg)
        typer.secho(f"[email] E-mail enviado para {to_addr}", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"[email] Falha no envio: {e}", fg=typer.colors.RED)
        raise


if __name__ == "__main__":
    app()
