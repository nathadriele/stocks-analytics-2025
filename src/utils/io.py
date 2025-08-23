"""
Funções utilitárias de entrada/saída de dados.
Inclui leitura/escrita em Parquet, CSV e SQLite.
"""

import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path
from typing import Optional

from src import config


def save_parquet(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """
    Salva DataFrame em formato Parquet.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=index)
    print(f"[io] DataFrame salvo em {path}")


def load_parquet(path: Path) -> Optional[pd.DataFrame]:
    """
    Lê DataFrame de um arquivo Parquet, se existir.
    """
    if not path.exists():
        print(f"[io] Arquivo não encontrado: {path}")
        return None
    return pd.read_parquet(path)


def save_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """
    Salva DataFrame em formato CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    print(f"[io] DataFrame salvo em {path}")


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    """
    Lê DataFrame de um arquivo CSV, se existir.
    """
    if not path.exists():
        print(f"[io] Arquivo não encontrado: {path}")
        return None
    return pd.read_csv(path)


def get_db_engine():
    """
    Retorna engine SQLite baseada no caminho definido em config.DB_PATH.
    """
    return create_engine(f"sqlite:///{config.DB_PATH}")


def save_to_db(df: pd.DataFrame, table_name: str, if_exists: str = "replace") -> None:
    """
    Salva DataFrame em tabela SQLite.
    """
    engine = get_db_engine()
    with engine.begin() as conn:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    print(f"[io] DataFrame salvo em tabela '{table_name}' do banco {config.DB_PATH}")


def load_from_db(table_name: str) -> Optional[pd.DataFrame]:
    """
    Lê DataFrame de uma tabela SQLite, se existir.
    """
    engine = get_db_engine()
    with engine.begin() as conn:
        try:
            return pd.read_sql(f"SELECT * FROM {table_name}", conn)
        except Exception as e:
            print(f"[io] Erro ao carregar tabela '{table_name}': {e}")
            return None
