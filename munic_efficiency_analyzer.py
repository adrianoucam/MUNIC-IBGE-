#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
munic_efficiency_analyzer.py

Programa para operacionalizar, com dados reais da MUNIC/IBGE em SQLite, uma análise
multianual de eficiência administrativa municipal inspirada no artigo:
"Análise da eficiência administrativa na gestão pública municipal - desafios e estratégias para melhoria".

Entradas:
  - Banco SQLite com tabelas: periodos, indicadores, resultados, falhas.
Saídas:
  - CSVs normalizados
  - painel anual por município
  - indicadores por dimensão
  - ranking simples de eficiência administrativa
  - gráficos PNG

Uso:
  python munic_efficiency_analyzer.py --db sqlite_munic.sqlite --out saida_munic --municipio 330330
  python munic_efficiency_analyzer.py --db sqlite_munic.sqlite --out saida_munic --todos
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DIMENSOES = {
    # Dimensão do artigo: capacidade administrativa/servidores
    "capacidade_administrativa": [
        "funcionários ativos", "servidores", "administração direta",
        "administração indireta", "nível superior", "capacitação", "escolaridade"
    ],
    # Dimensão do artigo: planejamento estratégico e instrumentos de gestão
    "planejamento_governanca": [
        "planejamento", "plano diretor", "ppa", "lei de diretrizes",
        "instrumentos de planejamento", "controle interno", "consórcio",
        "conselho", "fundo municipal", "ouvidoria"
    ],
    # Dimensão do artigo: tecnologia/TIC/digitalização
    "tecnologia_informacao": [
        "informatizado", "informatização", "internet", "portal", "site",
        "tecnologia", "governança em ti", "sistema", "banco de dados",
        "mapeamento digital"
    ],
    # Dimensão do artigo: participação social e transparência
    "participacao_transparencia": [
        "participação", "transparência", "audiência pública", "conferência",
        "conselho municipal", "orçamento participativo", "consulta pública"
    ],
    # Dimensão do artigo: estrutura organizacional e burocracia
    "estrutura_organizacional": [
        "secretaria", "órgão", "estrutura", "administração", "licitação",
        "compras", "cadastro", "patrimônio", "alvará"
    ],
}


def conectar(db_path: str) -> sqlite3.Connection:
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Banco não encontrado: {db_path}")
    return sqlite3.connect(db_path)


def carregar_base(conn: sqlite3.Connection) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indicadores = pd.read_sql_query("SELECT * FROM indicadores", conn)
    resultados = pd.read_sql_query("SELECT * FROM resultados", conn)
    periodos = pd.read_sql_query("SELECT * FROM periodos", conn)

    # Remove linhas-resumo sem localidade_id, mantendo apenas municípios/códigos reais.
    resultados["localidade_id"] = resultados["localidade_id"].fillna("").astype(str).str.strip()
    resultados = resultados[resultados["localidade_id"] != ""].copy()

    # Padronizações
    resultados["periodo"] = resultados["periodo"].astype(str)
    indicadores["periodo"] = indicadores["periodo"].astype(str)
    resultados["indicador_id"] = resultados["indicador_id"].astype(str)
    indicadores["indicador_id"] = indicadores["indicador_id"].astype(str)

    base = resultados.merge(
        indicadores[["pesquisa_id", "periodo", "indicador_id", "posicao", "indicador_nome", "classe"]],
        on=["pesquisa_id", "periodo", "indicador_id", "posicao"],
        how="left",
    )

    base["valor_texto"] = base["valor"].astype(str).str.strip()
    base["valor_num"] = base["valor_texto"].map(parse_numero)
    base["indicador_nome"] = base["indicador_nome"].fillna("")
    return base, indicadores, periodos


def parse_numero(x: object) -> float:
    """Converte números em formato brasileiro ou textual. Retorna NaN quando não for número."""
    if x is None:
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "-", "não aplicável"}:
        return np.nan

    # Remove dicionários serializados, quando existirem em linhas agregadas.
    if s.startswith("{") or s.startswith("["):
        return np.nan

    # Sim/Não como variável binária para cálculo de presença institucional.
    low = s.lower()
    if low in {"sim", "s", "existe", "existente"}:
        return 1.0
    if low in {"não", "nao", "n", "não existe", "nao existe", "inexistente"}:
        return 0.0

    # Percentuais e números pt-BR.
    s = s.replace("%", "").replace("R$", "").replace(" ", "")
    if re.match(r"^-?\d{1,3}(\.\d{3})+,\d+$", s):
        s = s.replace(".", "").replace(",", ".")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


def filtrar(base: pd.DataFrame, municipio: Optional[str], todos: bool) -> pd.DataFrame:
    if todos:
        return base.copy()
    if municipio:
        return base[base["localidade_id"].astype(str) == str(municipio)].copy()
    # Padrão: usa todos se não informado.
    return base.copy()


def detectar_dimensao(nome: str) -> Optional[str]:
    nome_low = nome.lower()
    for dim, keywords in DIMENSOES.items():
        if any(k.lower() in nome_low for k in keywords):
            return dim
    return None


def preparar_painel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dimensao"] = df["indicador_nome"].map(detectar_dimensao)
    df_dim = df[df["dimensao"].notna()].copy()

    # Valor binário/presença para indicadores qualitativos; numérico quando houver número.
    # Para texto não numérico: marca presença de resposta para indicar cobertura institucional.
    df_dim["valor_calc"] = df_dim["valor_num"]
    mask_texto = df_dim["valor_calc"].isna() & df_dim["valor_texto"].notna() & (df_dim["valor_texto"].str.strip() != "")
    df_dim.loc[mask_texto, "valor_calc"] = 1.0

    # Agrega por município, ano e dimensão.
    painel = (
        df_dim.groupby(["localidade_id", "periodo", "dimensao"], as_index=False)
        .agg(
            qtd_indicadores=("indicador_id", "nunique"),
            qtd_respostas=("valor_texto", "count"),
            media_valor=("valor_calc", "mean"),
            soma_valor=("valor_calc", "sum"),
            indicadores=("indicador_nome", lambda x: " | ".join(sorted(set(map(str, x)))[:12])),
        )
    )

    # Normalização min-max por dimensão e ano, útil quando houver vários municípios.
    painel["score_dimensao"] = painel.groupby(["periodo", "dimensao"])["media_valor"].transform(minmax)
    # Quando só há um município, minmax fica neutro em 0.5; usa presença/valor bruto normalizado por log.
    single_mask = painel["score_dimensao"].isna()
    painel.loc[single_mask, "score_dimensao"] = painel.loc[single_mask, "media_valor"].map(lambda v: np.nan if pd.isna(v) else 0.5)

    pivot = painel.pivot_table(
        index=["localidade_id", "periodo"],
        columns="dimensao",
        values="score_dimensao",
        aggfunc="mean"
    ).reset_index()

    for dim in DIMENSOES:
        if dim not in pivot.columns:
            pivot[dim] = np.nan

    pivot["score_eficiencia_administrativa"] = pivot[list(DIMENSOES.keys())].mean(axis=1, skipna=True)
    pivot["ano"] = pd.to_numeric(pivot["periodo"], errors="coerce").astype("Int64")
    pivot = pivot.sort_values(["localidade_id", "ano"])
    return painel, pivot


def minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mn, mx = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series([0.5 if not pd.isna(v) else np.nan for v in s], index=s.index)
    return (s - mn) / (mx - mn)


def gerar_ranking(pivot: pd.DataFrame) -> pd.DataFrame:
    ranking = pivot.copy()
    ranking["rank_no_ano"] = ranking.groupby("periodo")["score_eficiencia_administrativa"].rank(
        method="dense", ascending=False
    )
    return ranking.sort_values(["periodo", "rank_no_ano", "localidade_id"])


def salvar_graficos(pivot: pd.DataFrame, out: Path, municipio: Optional[str]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df = pivot.copy()
    if municipio:
        df = df[df["localidade_id"].astype(str) == str(municipio)].copy()
    if df.empty:
        return

    # Série temporal score geral
    for loc, g in df.groupby("localidade_id"):
        g = g.sort_values("ano")
        plt.figure(figsize=(10, 5))
        plt.plot(g["ano"], g["score_eficiencia_administrativa"], marker="o")
        plt.title(f"Score de eficiência administrativa - Município {loc}")
        plt.xlabel("Ano")
        plt.ylabel("Score 0-1")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / f"score_eficiencia_{loc}.png", dpi=160)
        plt.close()

        # Dimensões
        cols = list(DIMENSOES.keys())
        plt.figure(figsize=(11, 6))
        for col in cols:
            if col in g:
                plt.plot(g["ano"], g[col], marker="o", label=col)
        plt.title(f"Dimensões da eficiência administrativa - Município {loc}")
        plt.xlabel("Ano")
        plt.ylabel("Score 0-1")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out / f"dimensoes_{loc}.png", dpi=160)
        plt.close()


def diagnostico_textual(pivot: pd.DataFrame, painel: pd.DataFrame, out: Path, municipio: Optional[str]) -> None:
    linhas: List[str] = []
    linhas.append("# Relatório sintético - Eficiência administrativa MUNIC\n")
    linhas.append("Este relatório foi gerado automaticamente a partir do banco SQLite da MUNIC.\n")
    linhas.append("Dimensões usadas: " + ", ".join(DIMENSOES.keys()) + ".\n\n")

    df = pivot.copy()
    if municipio:
        df = df[df["localidade_id"].astype(str) == str(municipio)].copy()

    if df.empty:
        linhas.append("Nenhum dado encontrado para o filtro informado.\n")
    else:
        for loc, g in df.groupby("localidade_id"):
            g = g.sort_values("ano")
            linhas.append(f"## Município {loc}\n")
            linhas.append(f"Anos analisados: {', '.join(g['periodo'].astype(str).tolist())}\n")
            ult = g.dropna(subset=["score_eficiencia_administrativa"]).tail(1)
            if not ult.empty:
                row = ult.iloc[0]
                linhas.append(f"Último ano disponível: {row['periodo']}. Score geral: {row['score_eficiencia_administrativa']:.3f}.\n")
                dims = {d: row.get(d, np.nan) for d in DIMENSOES}
                dims_valid = {k: v for k, v in dims.items() if pd.notna(v)}
                if dims_valid:
                    melhor = max(dims_valid, key=dims_valid.get)
                    pior = min(dims_valid, key=dims_valid.get)
                    linhas.append(f"Maior dimensão: {melhor} ({dims_valid[melhor]:.3f}).\n")
                    linhas.append(f"Menor dimensão: {pior} ({dims_valid[pior]:.3f}).\n")
            linhas.append("\n")

    (out / "relatorio_sintetico.md").write_text("".join(linhas), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Análise multianual da eficiência administrativa municipal usando MUNIC/IBGE em SQLite.")
    parser.add_argument("--db", required=True, help="Caminho do banco SQLite da MUNIC.")
    parser.add_argument("--out", default="saida_munic", help="Pasta de saída.")
    parser.add_argument("--municipio", default=None, help="Código IBGE do município, ex.: 330330.")
    parser.add_argument("--todos", action="store_true", help="Processa todos os municípios existentes no banco.")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    conn = conectar(args.db)
    base, indicadores, periodos = carregar_base(conn)
    dados = filtrar(base, args.municipio, args.todos)

    painel_dim, pivot = preparar_painel(dados)
    ranking = gerar_ranking(pivot)

    base.to_csv(out / "munic_resultados_normalizados.csv", index=False, encoding="utf-8-sig")
    indicadores.to_csv(out / "munic_indicadores.csv", index=False, encoding="utf-8-sig")
    periodos.to_csv(out / "munic_periodos.csv", index=False, encoding="utf-8-sig")
    painel_dim.to_csv(out / "painel_dimensoes.csv", index=False, encoding="utf-8-sig")
    pivot.to_csv(out / "painel_score_eficiencia.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(out / "ranking_eficiencia_por_ano.csv", index=False, encoding="utf-8-sig")

    salvar_graficos(pivot, out, args.municipio)
    diagnostico_textual(pivot, painel_dim, out, args.municipio)

    print(f"[OK] Saídas geradas em: {out.resolve()}")
    print(f"[OK] Linhas normalizadas: {len(base)}")
    print(f"[OK] Linhas analisadas: {len(dados)}")
    print(f"[OK] Painel município-ano: {len(pivot)}")


if __name__ == "__main__":
    main()
