
# -*- coding: utf-8 -*-
"""
MUNIC ML - Qualidade de vida, segurança, saúde e educação
Autor: gerado para análise da Pesquisa de Informações Básicas Municipais (MUNIC/IBGE)

O programa:
1) Lê um banco SQLite no formato: periodos, indicadores, resultados.
2) Monta uma matriz município-ano x indicadores.
3) Codifica respostas numéricas e categóricas.
4) Cria índices compostos por temas: qualidade de vida, segurança, saúde, educação,
   governança, gestão, tecnologia, participação e meio ambiente.
5) Treina e salva modelos de aprendizado de máquina:
   - RandomForestRegressor para predizer o índice de qualidade de vida construído;
   - RandomForestClassifier para classificar maturidade municipal em baixa/média/alta.
6) Gera gráficos de evolução dos índices por município.
7) Exporta features mais importantes.

Observação metodológica:
- A MUNIC não traz diretamente IDH/IDHM em muitos layouts. Se você tiver uma tabela externa
  com IDHM, use --idh_csv com colunas: localidade_id, ano, idhm. Nesse caso, o modelo também
  treina para predizer idhm.
"""
import argparse
import json
import re
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

THEMES = {
    "saude": ["saúde", "saude", "hospital", "médic", "medic", "unidade básica", "ubs", "vigilância", "vacina", "sanit"],
    "educacao": ["educação", "educacao", "escola", "ensino", "professor", "creche", "fundeb", "merenda", "aluno"],
    "seguranca": ["segurança", "seguranca", "guarda", "defesa civil", "violência", "violencia", "risco", "desastre", "proteção", "protecao"],
    "governanca": ["transparência", "transparencia", "controle", "ouvidoria", "accountability", "conselho", "audiência", "audiencia", "lei de acesso", "lai"],
    "tecnologia": ["internet", "tecnologia", "informática", "informatica", "ti", "sistema", "digital", "eletrônico", "eletronico", "site", "portal"],
    "gestao": ["planejamento", "plano", "orçamento", "orcamento", "pessoal", "servidor", "capacitação", "capacitacao", "estrutura administrativa", "gestão", "gestao"],
    "participacao": ["participação", "participacao", "conselho", "conferência", "conferencia", "consulta pública", "consulta publica", "audiência", "audiencia"],
    "meio_ambiente": ["meio ambiente", "ambiental", "saneamento", "resíduo", "residuo", "lixo", "água", "agua", "esgoto", "drenagem", "clima"],
}

POSITIVE_WORDS = ["existência", "existencia", "sim", "possui", "tem", "implantado", "realiza", "realizado", "ativo", "funciona", "conselho", "plano", "programa", "sistema"]
NEGATIVE_WORDS = ["não", "nao", "inexist", "sem", "ausência", "ausencia", "não possui", "nao possui"]

def norm_text(x):
    return str(x or "").strip().lower()

def encode_value(v):
    """Converte respostas MUNIC para escala numérica aproximada."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "-", "..."}:
        return np.nan
    s2 = s.lower().replace(".", "").replace(",", ".")
    # número puro
    try:
        return float(s2)
    except Exception:
        pass
    # categorias frequentes
    if re.fullmatch(r"sim|s", s2):
        return 1.0
    if re.fullmatch(r"não|nao|n", s2):
        return 0.0
    if "ensino superior" in s2 or "pós" in s2 or "pos" in s2 or "superior completo" in s2:
        return 1.0
    if "ensino médio" in s2 or "ensino medio" in s2:
        return 0.65
    if "fundamental" in s2:
        return 0.35
    if "sem instru" in s2:
        return 0.05
    # presença/ausência textual
    if any(w in s2 for w in NEGATIVE_WORDS):
        return 0.0
    if any(w in s2 for w in POSITIVE_WORDS):
        return 1.0
    # categoria nominal sem ordem: codifica como ausente para índice, mas mantém NaN para evitar ruído nominal
    return np.nan

def load_munic(db_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = sqlite3.connect(db_path)
    ind = pd.read_sql_query("SELECT pesquisa_id, periodo, indicador_id, posicao, indicador_nome, classe FROM indicadores", con)
    res = pd.read_sql_query("SELECT pesquisa_id, periodo, indicador_id, posicao, localidade_id, localidade_nome, valor FROM resultados", con)
    con.close()
    # Remove linhas sem localidade quando há localidade específica duplicada
    res["localidade_id"] = res["localidade_id"].astype(str).str.strip()
    res = res[res["localidade_id"].ne("")].copy()
    res["ano"] = pd.to_numeric(res["periodo"], errors="coerce").astype("Int64")
    return ind, res

def build_matrix(ind: pd.DataFrame, res: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    meta = ind.copy()
    meta["feature"] = "i_" + meta["indicador_id"].astype(str)
    names = dict(zip(meta["feature"], meta["indicador_nome"].astype(str)))

    df = res.merge(meta[["periodo", "indicador_id", "feature", "indicador_nome"]], on=["periodo", "indicador_id"], how="left")
    df["valor_num"] = df["valor"].map(encode_value)
    # média por indicador caso haja duplicidade
    piv = df.pivot_table(index=["localidade_id", "ano"], columns="feature", values="valor_num", aggfunc="mean")
    piv = piv.reset_index()
    return piv, names

def theme_features(names: dict) -> dict[str, list[str]]:
    out = {k: [] for k in THEMES}
    for f, nm in names.items():
        n = norm_text(nm)
        for theme, keys in THEMES.items():
            if any(k in n for k in keys):
                out[theme].append(f)
    return out

def minmax_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index)
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)

def compute_indices(X: pd.DataFrame, names: dict) -> tuple[pd.DataFrame, dict]:
    tf = theme_features(names)
    idx = X[["localidade_id", "ano"]].copy()
    for theme, cols in tf.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            idx[f"idx_{theme}"] = np.nan
            continue
        # normaliza cada indicador para 0-1 e tira média temática
        normalized = X[cols].apply(minmax_series, axis=0)
        idx[f"idx_{theme}"] = normalized.mean(axis=1, skipna=True)

    components = ["idx_saude", "idx_educacao", "idx_seguranca", "idx_governanca", "idx_tecnologia", "idx_gestao", "idx_participacao", "idx_meio_ambiente"]
    weights = {
        "idx_saude": 0.18,
        "idx_educacao": 0.18,
        "idx_seguranca": 0.16,
        "idx_governanca": 0.13,
        "idx_tecnologia": 0.10,
        "idx_gestao": 0.10,
        "idx_participacao": 0.07,
        "idx_meio_ambiente": 0.08,
    }
    present = [c for c in components if c in idx]
    wsum = sum(weights[c] for c in present)
    idx["idx_qualidade_vida"] = sum(idx[c].fillna(idx[c].mean()) * weights[c] for c in present) / (wsum or 1)
    idx["classe_maturidade"] = pd.qcut(idx["idx_qualidade_vida"].rank(method="first"), q=min(3, idx["idx_qualidade_vida"].notna().sum()), labels=False, duplicates="drop")
    idx["classe_maturidade"] = idx["classe_maturidade"].fillna(0).astype(int)
    return idx, tf

def train_models(X: pd.DataFrame, indices: pd.DataFrame, out: Path, idh_csv: str|None=None):
    out.mkdir(parents=True, exist_ok=True)
    data = X.merge(indices, on=["localidade_id", "ano"], how="inner")
    if idh_csv:
        idh = pd.read_csv(idh_csv)
        idh = idh.rename(columns={"municipio":"localidade_id", "cod_mun":"localidade_id", "year":"ano", "periodo":"ano"})
        idh["localidade_id"] = idh["localidade_id"].astype(str)
        idh["ano"] = pd.to_numeric(idh["ano"], errors="coerce").astype("Int64")
        data = data.merge(idh[["localidade_id", "ano", "idhm"]], on=["localidade_id", "ano"], how="left")

    feature_cols = [c for c in X.columns if c.startswith("i_")]
    # remove colunas quase vazias
    feature_cols = [c for c in feature_cols if data[c].notna().mean() >= 0.05]
    Xf = data[feature_cols]
    y = data["idx_qualidade_vida"]

    reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
        ("model", RandomForestRegressor(n_estimators=400, random_state=42, min_samples_leaf=1))
    ])
    reg.fit(Xf, y)
    pred = reg.predict(Xf)
    metrics = {"target":"idx_qualidade_vida", "n_samples": int(len(data)), "n_features": int(len(feature_cols)), "mae_train": float(mean_absolute_error(y, pred)), "r2_train": float(r2_score(y, pred)) if len(data)>1 else None}
    joblib.dump({"pipeline": reg, "feature_cols": feature_cols, "target":"idx_qualidade_vida"}, out/"modelo_rf_qualidade_vida.joblib")

    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"))
    ])
    ycls = data["classe_maturidade"]
    clf.fit(Xf, ycls)
    joblib.dump({"pipeline": clf, "feature_cols": feature_cols, "target":"classe_maturidade"}, out/"modelo_rf_classe_maturidade.joblib")

    # Modelo opcional de IDHM se existir
    if "idhm" in data.columns and data["idhm"].notna().sum() >= 5:
        d2 = data[data["idhm"].notna()].copy()
        idh_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            ("model", RandomForestRegressor(n_estimators=400, random_state=42))
        ])
        idh_model.fit(d2[feature_cols], d2["idhm"])
        joblib.dump({"pipeline": idh_model, "feature_cols": feature_cols, "target":"idhm"}, out/"modelo_rf_idhm.joblib")
        metrics["idhm_model"] = {"n_samples": int(len(d2)), "r2_train": float(r2_score(d2["idhm"], idh_model.predict(d2[feature_cols]))) if len(d2)>1 else None}

    # Importância por RF
    importances = reg.named_steps["model"].feature_importances_
    imp = pd.DataFrame({"feature": feature_cols, "importance": importances}).sort_values("importance", ascending=False)
    imp.to_csv(out/"features_importantes_modelo.csv", index=False, encoding="utf-8-sig")
    with open(out/"metricas_modelo.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return data, imp, metrics

def plot_evolution(indices: pd.DataFrame, out: Path, top_n=20):
    out.mkdir(parents=True, exist_ok=True)
    index_cols = [c for c in indices.columns if c.startswith("idx_")]
    indices.to_csv(out/"indices_municipio_ano.csv", index=False, encoding="utf-8-sig")

    # gráfico geral: média por ano
    avg = indices.groupby("ano")[index_cols].mean(numeric_only=True).reset_index()
    for col in index_cols:
        plt.figure(figsize=(10,5))
        plt.plot(avg["ano"], avg[col], marker="o")
        plt.title(f"Evolução média - {col}")
        plt.xlabel("Ano")
        plt.ylabel("Índice 0-1")
        plt.grid(True, alpha=.3)
        plt.tight_layout()
        plt.savefig(out/f"evolucao_media_{col}.png", dpi=160)
        plt.close()

    # por município para qualidade de vida
    last = indices.sort_values("ano").groupby("localidade_id").tail(1).sort_values("idx_qualidade_vida", ascending=False).head(top_n)
    for mun in last["localidade_id"].astype(str):
        d = indices[indices["localidade_id"].astype(str)==mun].sort_values("ano")
        plt.figure(figsize=(10,5))
        for col in ["idx_qualidade_vida", "idx_saude", "idx_educacao", "idx_seguranca"]:
            if col in d:
                plt.plot(d["ano"], d[col], marker="o", label=col.replace("idx_", ""))
        plt.title(f"Evolução dos índices - município {mun}")
        plt.xlabel("Ano")
        plt.ylabel("Índice 0-1")
        plt.legend()
        plt.grid(True, alpha=.3)
        plt.tight_layout()
        plt.savefig(out/f"evolucao_municipio_{mun}.png", dpi=160)
        plt.close()

    # ranking último ano
    ano_max = indices["ano"].max()
    rank = indices[indices["ano"]==ano_max].sort_values("idx_qualidade_vida", ascending=False)
    rank.to_csv(out/"ranking_ultimo_ano.csv", index=False, encoding="utf-8-sig")
    return avg, rank

def feature_report(imp: pd.DataFrame, names: dict, out: Path, n=50):
    rep = imp.head(n).copy()
    rep["indicador_nome"] = rep["feature"].map(names)
    rep.to_csv(out/"top_features_interpretadas.csv", index=False, encoding="utf-8-sig")
    # gráfico top 20
    top = rep.head(20).iloc[::-1]
    plt.figure(figsize=(12,8))
    labels = [str(x)[:70] for x in top["indicador_nome"].fillna(top["feature"])]
    plt.barh(labels, top["importance"])
    plt.title("Top 20 features associadas ao índice de qualidade de vida")
    plt.xlabel("Importância no Random Forest")
    plt.tight_layout()
    plt.savefig(out/"top20_features_qualidade_vida.png", dpi=160)
    plt.close()
    return rep

def write_methodology(out: Path, metrics: dict, theme_cols: dict):
    txt = []
    txt.append("# Relatório metodológico - MUNIC ML\n")
    txt.append("## Objetivo\nConstruir índices municipais e modelos de aprendizado de máquina para investigar fatores associados à qualidade de vida, segurança, saúde e educação usando dados MUNIC multi-anuais.\n")
    txt.append("## Modelos salvos\n- modelo_rf_qualidade_vida.joblib: regressão do índice composto de qualidade de vida.\n- modelo_rf_classe_maturidade.joblib: classificação baixa/média/alta maturidade.\n- modelo_rf_idhm.joblib: gerado somente se for informado CSV externo com IDHM.\n")
    txt.append("## Métricas\n```json\n"+json.dumps(metrics, ensure_ascii=False, indent=2)+"\n```\n")
    txt.append("## Quantidade de indicadores por tema\n")
    for k,v in theme_cols.items():
        txt.append(f"- {k}: {len(v)} indicadores encontrados por palavras-chave.\n")
    txt.append("\n## Cuidado interpretativo\nO índice composto não é um IDHM oficial. Ele é uma aproximação empírica baseada nas variáveis disponíveis da MUNIC. Para análise científica, recomenda-se validar com IDHM, PIB per capita, mortalidade, IDEB, cobertura de saúde e dados de segurança pública externos.\n")
    (out/"RELATORIO_METODOLOGICO.md").write_text("".join(txt), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Caminho do sqlite_munic.sqlite")
    ap.add_argument("--out", default="saida_munic_ml", help="Pasta de saída")
    ap.add_argument("--idh_csv", default=None, help="CSV opcional com colunas localidade_id, ano, idhm")
    ap.add_argument("--top_n", type=int, default=20)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ind, res = load_munic(args.db)
    X, names = build_matrix(ind, res)
    X.to_csv(out/"matriz_features_municipio_ano.csv", index=False, encoding="utf-8-sig")
    indices, theme_cols = compute_indices(X, names)
    data, imp, metrics = train_models(X, indices, out, args.idh_csv)
    plot_evolution(indices, out, args.top_n)
    feature_report(imp, names, out)
    write_methodology(out, metrics, theme_cols)
    print("[OK] Processamento concluído")
    print(f"Saídas em: {out.resolve()}")
    print("Modelos salvos:")
    print(" - modelo_rf_qualidade_vida.joblib")
    print(" - modelo_rf_classe_maturidade.joblib")
    if (out/"modelo_rf_idhm.joblib").exists():
        print(" - modelo_rf_idhm.joblib")

if __name__ == "__main__":
    main()
