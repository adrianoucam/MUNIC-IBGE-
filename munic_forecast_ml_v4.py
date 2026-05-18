# -*- coding: utf-8 -*-
"""
MUNIC Forecast ML - v4.2 corrigido

Objetivo:
- Ler o banco SQLite da MUNIC.
- Calcular índices sintéticos municipais por ano.
- Separar municípios por UF e Região do Brasil.
- Gerar previsão por IA para os próximos anos para todos os índices.
- Gerar gráficos com histórico em preto e previsão em verde/vermelho:
  verde = previsão melhora em relação ao último valor histórico;
  vermelho = previsão piora em relação ao último valor histórico.

Observação metodológica:
O índice de qualidade de vida gerado aqui é um proxy administrativo-institucional derivado da MUNIC.
Ele NÃO é o IDHM oficial. Para comparação com IDHM, use base externa do Atlas Brasil/PNUD/IPEA/FJP.
"""

import argparse
import json
import re
import sqlite3
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


THEMES = {
    "saude": [
        "saude", "hospital", "medic", "unidade basica", "ubs", "vigilancia", "vacina",
        "sanit", "epidemi", "atencao basica", "sus"
    ],
    "educacao": [
        "educacao", "escola", "ensino", "professor", "creche", "fundeb", "merenda",
        "aluno", "biblioteca", "transporte escolar"
    ],
    "seguranca": [
        "seguranca", "guarda", "defesa civil", "violencia", "risco", "desastre",
        "protecao", "bombeiro", "enchente", "deslizamento"
    ],
    "governanca": [
        "transparencia", "controle", "ouvidoria", "conselho", "audiencia", "lei de acesso",
        "lai", "fiscalizacao", "prestacao", "governanca"
    ],
    "tecnologia": [
        "internet", "tecnologia", "informatica", " sistema", "digital", "eletronico", "site",
        "portal", "computador", "software", "geoprocessamento"
    ],
    "gestao": [
        "planejamento", "plano", "orcamento", "pessoal", "servidor", "capacitacao",
        "estrutura administrativa", "gestao", "consorcio", "cadastro", "administracao"
    ],
    "participacao": [
        "participacao", "conselho", "conferencia", "consulta publica", "audiencia", "forum",
        "comite", "representante"
    ],
    "meio_ambiente": [
        "meio ambiente", "ambiental", "saneamento", "residuo", "lixo", "agua", "esgoto",
        "drenagem", "clima", "defesa ambiental", "coleta seletiva"
    ],
}

UF_INFO = {
    "11": ("RO", "Norte"), "12": ("AC", "Norte"), "13": ("AM", "Norte"),
    "14": ("RR", "Norte"), "15": ("PA", "Norte"), "16": ("AP", "Norte"),
    "17": ("TO", "Norte"),
    "21": ("MA", "Nordeste"), "22": ("PI", "Nordeste"), "23": ("CE", "Nordeste"),
    "24": ("RN", "Nordeste"), "25": ("PB", "Nordeste"), "26": ("PE", "Nordeste"),
    "27": ("AL", "Nordeste"), "28": ("SE", "Nordeste"), "29": ("BA", "Nordeste"),
    "31": ("MG", "Sudeste"), "32": ("ES", "Sudeste"), "33": ("RJ", "Sudeste"),
    "35": ("SP", "Sudeste"),
    "41": ("PR", "Sul"), "42": ("SC", "Sul"), "43": ("RS", "Sul"),
    "50": ("MS", "Centro-Oeste"), "51": ("MT", "Centro-Oeste"),
    "52": ("GO", "Centro-Oeste"), "53": ("DF", "Centro-Oeste"),
}

MISSING = {
    "", "nan", "none", "null", "-", "...", "ignorado", "não sabe", "nao sabe",
    "não informado", "nao informado"
}


def strip_accents(s):
    return "".join(
        c for c in unicodedata.normalize("NFKD", str(s))
        if not unicodedata.combining(c)
    )


def norm(s):
    return re.sub(r"\s+", " ", strip_accents(str(s or "")).lower().strip())


def safe_filename(value, max_len=100):
    """Remove caracteres problemáticos em nomes de arquivos no Windows/Linux."""
    text = strip_accents(str(value or "sem_nome"))
    text = re.sub(r"[^0-9A-Za-z_-]+", "_", text)
    text = text.strip("._- ")
    if not text:
        text = "sem_nome"
    return text[:max_len]


def encode_binary(v):
    if pd.isna(v):
        return np.nan
    s = norm(str(v).strip())
    if s in MISSING:
        return np.nan
    if s in {"sim", "s"}:
        return 1.0
    if s in {"nao", "não", "n"}:
        return 0.0
    if any(k in s for k in ["nao possui", "não possui", "inexist", "sem ", "ausencia", "ausente"]):
        return 0.0
    if any(k in s for k in ["possui", "existe", "existente", "implant", "implement", "realiza", "ativo", "funciona"]):
        return 1.0
    if "superior" in s or "pos-gradu" in s or "pós-gradu" in s:
        return 1.0
    if "medio" in s or "médio" in s:
        return 0.65
    if "fundamental" in s:
        return 0.35
    if "sem instr" in s:
        return 0.05
    return np.nan


def encode_numeric(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().replace(".", "").replace(",", ".")
    if norm(s) in MISSING:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def uf_region_from_code(localidade_id):
    code = re.sub(r"\D", "", str(localidade_id or ""))
    if len(code) < 2:
        return "NA", "Não identificada"
    return UF_INFO.get(code[:2], ("NA", "Não identificada"))


def load(db_path):
    con = sqlite3.connect(db_path)
    ind = pd.read_sql(
        "SELECT pesquisa_id, periodo, indicador_id, posicao, indicador_nome, classe FROM indicadores",
        con,
    )
    res = pd.read_sql(
        "SELECT pesquisa_id, periodo, indicador_id, posicao, localidade_id, localidade_nome, valor FROM resultados",
        con,
    )
    con.close()

    res["localidade_id"] = res["localidade_id"].astype(str).str.strip()
    res = res[res["localidade_id"].ne("")].copy()
    res["ano"] = pd.to_numeric(res["periodo"], errors="coerce").astype("Int64")

    ind["feature"] = (
        "i_"
        + ind["indicador_id"].astype(str)
        + "__"
        + ind["posicao"].astype(str).str.replace(r"[^0-9A-Za-z_]+", "_", regex=True)
    )
    return ind, res


def feature_names(ind):
    return dict(zip(ind["feature"], ind["indicador_nome"].astype(str)))


def build_matrix(ind, res):
    meta = ind[["pesquisa_id", "periodo", "indicador_id", "posicao", "feature", "indicador_nome"]].drop_duplicates()
    df = res.merge(meta, on=["pesquisa_id", "periodo", "indicador_id", "posicao"], how="left")
    df["valor_bin"] = df["valor"].map(encode_binary)
    df["valor_num"] = df["valor"].map(encode_numeric)

    bin_piv = df.pivot_table(
        index=["localidade_id", "ano"], columns="feature", values="valor_bin", aggfunc="mean"
    ).reset_index()
    num_piv = df.pivot_table(
        index=["localidade_id", "ano"], columns="feature", values="valor_num", aggfunc="mean"
    ).reset_index()
    num_piv = num_piv.rename(
        columns={c: "num_" + c for c in num_piv.columns if c not in ["localidade_id", "ano"]}
    )
    X = bin_piv.merge(num_piv, on=["localidade_id", "ano"], how="outer")

    names_df = df[["localidade_id", "localidade_nome"]].drop_duplicates()
    names_df = names_df.groupby("localidade_id", as_index=False)["localidade_nome"].first()
    X = X.merge(names_df, on="localidade_id", how="left")
    return X, df


def theme_features(names):
    out = {k: [] for k in THEMES}
    for f, nm in names.items():
        n = norm(nm)
        for theme, keys in THEMES.items():
            if any(k in n for k in keys):
                out[theme].append(f)
    return out


def add_geo_columns(df):
    uf_reg = df["localidade_id"].map(uf_region_from_code)
    df["uf"] = [x[0] for x in uf_reg]
    df["regiao"] = [x[1] for x in uf_reg]
    return df


def compute_indices(X, names, out):
    tf = theme_features(names)
    base_cols = ["localidade_id", "ano"]
    if "localidade_nome" in X.columns:
        base_cols.append("localidade_nome")
    idx = X[base_cols].copy()
    idx = add_geo_columns(idx)

    diag = []
    for theme, cols in tf.items():
        cols = [c for c in cols if c in X.columns]
        valid_cols = [c for c in cols if X[c].notna().sum() > 0]
        diag.append({
            "tema": theme,
            "indicadores_por_palavra_chave": len(cols),
            "indicadores_com_resposta_binaria": len(valid_cols),
        })
        if not valid_cols:
            idx[f"idx_{theme}"] = np.nan
            idx[f"n_{theme}"] = 0
        else:
            idx[f"n_{theme}"] = X[valid_cols].notna().sum(axis=1)
            idx[f"idx_{theme}"] = X[valid_cols].mean(axis=1, skipna=True)

    pd.DataFrame(diag).to_csv(
        out / "diagnostico_indicadores_temas.csv", index=False, encoding="utf-8-sig"
    )

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
    for c in weights:
        if c not in idx:
            idx[c] = np.nan

    filled = []
    for c, w in weights.items():
        m = idx[c].mean(skipna=True)
        fill_value = m if pd.notna(m) else 0.0
        filled.append(idx[c].fillna(fill_value) * w)
    idx["idx_qualidade_vida"] = sum(filled) / sum(weights.values())

    if idx["idx_qualidade_vida"].nunique(dropna=True) >= 3:
        idx["classe_maturidade"] = pd.qcut(
            idx["idx_qualidade_vida"].rank(method="first"),
            q=3,
            labels=False,
            duplicates="drop",
        )
    else:
        idx["classe_maturidade"] = pd.cut(
            idx["idx_qualidade_vida"], bins=[-0.01, 0.33, 0.66, 1.01], labels=[0, 1, 2]
        )
    idx["classe_maturidade"] = idx["classe_maturidade"].astype("Int64").fillna(0).astype(int)
    return idx, tf


def save_indices(idx, out):
    idx_cols = [
        "localidade_id", "localidade_nome", "uf", "regiao", "ano",
        "idx_saude", "idx_educacao", "idx_seguranca", "idx_governanca",
        "idx_tecnologia", "idx_gestao", "idx_participacao", "idx_meio_ambiente",
        "idx_qualidade_vida", "classe_maturidade",
    ]
    for c in idx_cols:
        if c not in idx.columns:
            idx[c] = "" if c in {"localidade_nome", "uf", "regiao"} else np.nan
    idx[idx_cols].to_csv(out / "indices_municipio_ano.csv", index=False, encoding="utf-8-sig")

    ncols = [c for c in idx.columns if c.startswith("n_")]
    idx[["localidade_id", "ano", "uf", "regiao"] + ncols].to_csv(
        out / "cobertura_indicadores_por_tema.csv", index=False, encoding="utf-8-sig"
    )


def normalize_numeric_features(data, feature_cols):
    data = data.copy()
    for c in feature_cols:
        if c.startswith("num_"):
            s = pd.to_numeric(data[c], errors="coerce")
            if s.notna().sum() >= 2 and s.max() != s.min():
                data[c] = (s - s.min()) / (s.max() - s.min())
            else:
                data[c] = np.nan
    return data


def train_quality_models(X, idx, out, names, idh_csv=None):
    geo_cols = ["localidade_id", "ano", "idx_qualidade_vida", "classe_maturidade"]
    data = X.merge(idx[geo_cols], on=["localidade_id", "ano"], how="inner")

    if idh_csv:
        idh = pd.read_csv(idh_csv)
        idh = idh.rename(columns={
            "municipio": "localidade_id",
            "cod_mun": "localidade_id",
            "codigo_municipio": "localidade_id",
            "year": "ano",
            "periodo": "ano",
            "IDHM": "idhm",
            "idh": "idhm",
        })
        idh["localidade_id"] = idh["localidade_id"].astype(str)
        idh["ano"] = pd.to_numeric(idh["ano"], errors="coerce").astype("Int64")
        data = data.merge(idh[["localidade_id", "ano", "idhm"]], on=["localidade_id", "ano"], how="left")

    feature_cols = [
        c for c in X.columns
        if c not in ["localidade_id", "ano", "localidade_nome"] and data[c].notna().mean() >= 0.03
    ]
    data = normalize_numeric_features(data, feature_cols)
    feature_cols = [c for c in feature_cols if data[c].notna().sum() > 0]

    if len(data) < 3 or len(feature_cols) == 0:
        metrics = {
            "alerta": "Dados insuficientes para treinar modelo de qualidade de vida.",
            "n_samples": int(len(data)),
            "n_features": int(len(feature_cols)),
        }
        with open(out / "metricas_modelo_qualidade.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        return data, pd.DataFrame(), metrics

    Xf = data[feature_cols]
    y = data["idx_qualidade_vida"]
    reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=500, random_state=42, min_samples_leaf=1, n_jobs=-1)),
    ])
    reg.fit(Xf, y)
    pred = reg.predict(Xf)

    metrics = {
        "target": "idx_qualidade_vida",
        "n_samples": int(len(data)),
        "n_municipios": int(data["localidade_id"].nunique()),
        "n_features": int(len(feature_cols)),
        "mae_train": float(mean_absolute_error(y, pred)),
        "r2_train": float(r2_score(y, pred)) if len(data) > 1 else None,
        "alerta": "Se houver poucos municípios no SQLite, o modelo mede evolução temporal interna. Para comparação regional real, carregue todos os municípios da MUNIC.",
    }
    joblib.dump(
        {"pipeline": reg, "feature_cols": feature_cols, "target": "idx_qualidade_vida"},
        out / "modelo_rf_qualidade_vida.joblib",
    )

    if data["classe_maturidade"].nunique() >= 2:
        clf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced", n_jobs=-1)),
        ])
        clf.fit(Xf, data["classe_maturidade"])
        joblib.dump(
            {"pipeline": clf, "feature_cols": feature_cols, "target": "classe_maturidade"},
            out / "modelo_rf_classe_maturidade.joblib",
        )

    if "idhm" in data.columns and data["idhm"].notna().sum() >= 5:
        d2 = data[data["idhm"].notna()].copy()
        idhm_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)),
        ])
        idhm_model.fit(d2[feature_cols], d2["idhm"])
        joblib.dump(
            {"pipeline": idhm_model, "feature_cols": feature_cols, "target": "idhm"},
            out / "modelo_rf_idhm.joblib",
        )
        metrics["idhm_model"] = {
            "n_samples": int(len(d2)),
            "r2_train": float(r2_score(d2["idhm"], idhm_model.predict(d2[feature_cols]))) if len(d2) > 1 else None,
        }

    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": reg.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False)
    imp.to_csv(out / "features_importantes_modelo.csv", index=False, encoding="utf-8-sig")

    def label_feature(f):
        base = f.replace("num_", "")
        return names.get(base, f)

    if not imp.empty:
        imp2 = imp.copy()
        imp2["indicador_nome"] = imp2["feature"].map(label_feature)
        imp2.to_csv(out / "top_features_interpretadas.csv", index=False, encoding="utf-8-sig")

    with open(out / "metricas_modelo_qualidade.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return data, imp, metrics


def plot_features(imp, names, out):
    if imp is None or imp.empty:
        return

    def label(f):
        base = f.replace("num_", "")
        return names.get(base, f)

    rep = imp.head(100).copy()
    rep["indicador_nome"] = rep["feature"].map(label)
    rep.to_csv(out / "top_features_interpretadas.csv", index=False, encoding="utf-8-sig")

    top = rep.head(20).iloc[::-1]
    plt.figure(figsize=(12, 8))
    plt.barh([str(x)[:85] for x in top["indicador_nome"]], top["importance"])
    plt.title("Top 20 features associadas ao índice de qualidade de vida")
    plt.xlabel("Importância no Random Forest")
    plt.tight_layout()
    plt.savefig(out / "top20_features_qualidade_vida.png", dpi=160)
    plt.close()


def plot_evolucao_regioes(idx, out):
    index_cols = [c for c in idx.columns if c.startswith("idx_")]
    regional = idx.groupby(["regiao", "ano"], as_index=False)[index_cols].mean(numeric_only=True)
    regional.to_csv(out / "media_regiao_ano.csv", index=False, encoding="utf-8-sig")

    for col in index_cols:
        plt.figure(figsize=(11, 6))
        for reg, d in regional.groupby("regiao"):
            d = d.sort_values("ano")
            plt.plot(d["ano"], d[col], marker="o", label=reg)
        plt.ylim(0, 1)
        plt.title(f"Evolução regional - {col}")
        plt.xlabel("Ano")
        plt.ylabel("Índice médio 0-1")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / f"evolucao_regioes_{safe_filename(col)}.png", dpi=160)
        plt.close()


def compare_region(idx, out, regiao=None, ano=None, top=30):
    d = idx.copy()
    if regiao and regiao.lower() != "brasil":
        d = d[d["regiao"].str.lower() == regiao.lower()]
    if d.empty:
        return None
    if ano is None:
        ano = int(d["ano"].max())
    d = d[d["ano"] == ano]
    if d.empty:
        return None

    tag = safe_filename(regiao or "Brasil")
    ranking = d.sort_values("idx_qualidade_vida", ascending=False).copy()
    ranking.to_csv(out / f"ranking_{tag}_{ano}.csv", index=False, encoding="utf-8-sig")
    ranking.head(top).to_csv(out / f"top{top}_{tag}_{ano}.csv", index=False, encoding="utf-8-sig")

    topd = ranking.head(top).iloc[::-1]
    labels = topd["localidade_nome"].fillna("").astype(str).str.strip()
    labels = np.where(labels == "", topd["localidade_id"].astype(str), labels)
    plt.figure(figsize=(12, max(6, top * 0.32)))
    plt.barh(labels, topd["idx_qualidade_vida"])
    plt.xlim(0, 1)
    plt.title(f"Top {top} municípios - {regiao or 'Brasil'} - {ano}")
    plt.xlabel("Índice de qualidade de vida sintético")
    plt.tight_layout()
    plt.savefig(out / f"top{top}_{tag}_{ano}_qualidade_vida.png", dpi=170)
    plt.close()
    return ranking


def compare_selected_municipios(idx, out, municipios):
    if not municipios:
        return
    ids = [str(x).strip() for x in municipios.split(",") if str(x).strip()]
    d = idx[idx["localidade_id"].astype(str).isin(ids)].copy()
    if d.empty:
        return
    d.to_csv(out / "comparacao_municipios_selecionados.csv", index=False, encoding="utf-8-sig")
    cols = [
        "idx_qualidade_vida", "idx_saude", "idx_educacao", "idx_seguranca",
        "idx_governanca", "idx_tecnologia",
    ]
    for col in cols:
        if col not in d.columns:
            continue
        plt.figure(figsize=(11, 6))
        for mun, dm in d.groupby("localidade_id"):
            if dm["localidade_nome"].notna().any():
                name = str(dm["localidade_nome"].dropna().iloc[0])
                if not name.strip():
                    name = str(mun)
            else:
                name = str(mun)
            dm = dm.sort_values("ano")
            plt.plot(dm["ano"], dm[col], marker="o", label=name)
        plt.ylim(0, 1)
        plt.title(f"Comparação temporal - {col}")
        plt.xlabel("Ano")
        plt.ylabel("Índice 0-1")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / f"comparacao_municipios_{safe_filename(col)}.png", dpi=160)
        plt.close()


def cluster_last_year(idx, out, regiao=None, k=4):
    cols = [
        "idx_saude", "idx_educacao", "idx_seguranca", "idx_governanca",
        "idx_tecnologia", "idx_gestao", "idx_participacao", "idx_meio_ambiente",
    ]
    d = idx.copy()
    if regiao and regiao.lower() != "brasil":
        d = d[d["regiao"].str.lower() == regiao.lower()]
    if d.empty:
        return
    ano = int(d["ano"].max())
    d = d[d["ano"] == ano].copy()
    if len(d) < 3:
        d["cluster"] = 0
        d.to_csv(out / "clusters_ultimo_ano.csv", index=False, encoding="utf-8-sig")
        return

    kk = max(2, min(k, len(d)))
    Xc = d[cols].copy()
    Xc = SimpleImputer(strategy="median").fit_transform(Xc)
    Xs = StandardScaler().fit_transform(Xc)
    km = KMeans(n_clusters=kk, random_state=42, n_init=20)
    d["cluster"] = km.fit_predict(Xs)

    pca = PCA(n_components=2, random_state=42)
    pts = pca.fit_transform(Xs)
    d["pca1"] = pts[:, 0]
    d["pca2"] = pts[:, 1]
    d.to_csv(out / "clusters_ultimo_ano.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 7))
    for cl, dc in d.groupby("cluster"):
        plt.scatter(dc["pca1"], dc["pca2"], label=f"Cluster {cl}")
    plt.title(f"Clusters municipais - {regiao or 'Brasil'} - {ano}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out / "clusters_pca_ultimo_ano.png", dpi=160)
    plt.close()


def correlation_heatmap(idx, out, regiao=None):
    cols = [
        "idx_saude", "idx_educacao", "idx_seguranca", "idx_governanca",
        "idx_tecnologia", "idx_gestao", "idx_participacao", "idx_meio_ambiente",
        "idx_qualidade_vida",
    ]
    d = idx.copy()
    if regiao and regiao.lower() != "brasil":
        d = d[d["regiao"].str.lower() == regiao.lower()]
    if len(d) < 3:
        return
    corr = d[cols].corr()
    corr.to_csv(out / "correlacao_indices.csv", encoding="utf-8-sig")
    plt.figure(figsize=(9, 7))
    im = plt.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(cols)), [c.replace("idx_", "") for c in cols], rotation=45, ha="right")
    plt.yticks(range(len(cols)), [c.replace("idx_", "") for c in cols])
    plt.title(f"Correlação entre índices - {regiao or 'Brasil'}")
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.values[i, j]
            if pd.notna(val):
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "heatmap_correlacao_indices.png", dpi=160)
    plt.close()


def forecast_all_indices(idx, out, horizonte=5, min_years=3):
    """Treina um RandomForestRegressor global por índice e prevê os próximos anos por município."""
    out = Path(out)
    modelos_dir = out / "modelos_previsao"
    modelos_dir.mkdir(parents=True, exist_ok=True)

    df = idx.copy()
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
    df = df.dropna(subset=["ano", "localidade_id"])
    df["ano"] = df["ano"].astype(int)

    index_cols = [c for c in df.columns if c.startswith("idx_")]
    all_forecasts = []
    metrics = []

    def make_supervised(base, target):
        base = base.sort_values(["localidade_id", "ano"]).copy()
        base["y"] = base[target]
        base["lag1"] = base.groupby("localidade_id")["y"].shift(1)
        base["lag2"] = base.groupby("localidade_id")["y"].shift(2)
        base["roll3"] = base.groupby("localidade_id")["y"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        )
        base["trend"] = base["lag1"] - base["lag2"]
        base["t"] = base["ano"] - base.groupby("localidade_id")["ano"].transform("min")
        base["cod_uf"] = base["localidade_id"].astype(str).str[:2]
        return base.dropna(subset=["y"])

    for target in index_cols:
        needed = ["localidade_id", "localidade_nome", "uf", "regiao", "ano", target]
        sup = make_supervised(df[needed].copy(), target)
        sup = sup.dropna(subset=["y"])
        if sup["localidade_id"].nunique() == 0 or len(sup) < 5:
            continue

        feat_num = ["ano", "t", "lag1", "lag2", "roll3", "trend"]
        feat_cat = ["uf", "regiao", "cod_uf"]
        pre = ColumnTransformer([
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                feat_num,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                feat_cat,
            ),
        ])
        model = Pipeline([
            ("preprocess", pre),
            ("rf", RandomForestRegressor(
                n_estimators=350,
                random_state=42,
                min_samples_leaf=2,
                n_jobs=-1,
            )),
        ])

        years = sorted(sup["ano"].dropna().unique())
        if len(years) >= 4:
            test_year = years[-1]
            train = sup[sup["ano"] < test_year]
            test = sup[sup["ano"] == test_year]
            if len(train) >= 5 and len(test) >= 1:
                model.fit(train[feat_num + feat_cat], train["y"])
                pred = model.predict(test[feat_num + feat_cat])
                metrics.append({
                    "indice": target,
                    "validacao_ano": int(test_year),
                    "mae": float(mean_absolute_error(test["y"], pred)),
                    "r2": float(r2_score(test["y"], pred)) if len(test) > 1 else np.nan,
                    "n_treino": int(len(train)),
                    "n_teste": int(len(test)),
                })

        model.fit(sup[feat_num + feat_cat], sup["y"])
        model_name = f"modelo_previsao_{safe_filename(target)}.joblib"
        joblib.dump(model, modelos_dir / model_name)

        for mun, g in df.groupby("localidade_id"):
            g = g.sort_values("ano").copy()
            hist = g[["ano", target]].dropna()
            if len(hist) < min_years:
                continue

            if g["localidade_nome"].notna().any():
                nome = g["localidade_nome"].dropna().iloc[-1]
            else:
                nome = str(mun)
            uf = g["uf"].dropna().iloc[-1] if g["uf"].notna().any() else "NA"
            reg = g["regiao"].dropna().iloc[-1] if g["regiao"].notna().any() else "Não identificada"

            last_year = int(hist["ano"].max())
            series = list(hist[target].astype(float).values)
            years_hist = list(hist["ano"].astype(int).values)
            first_year = min(years_hist)
            last_value = float(series[-1])

            for step in range(1, horizonte + 1):
                year = last_year + step
                lag1 = series[-1] if len(series) >= 1 else np.nan
                lag2 = series[-2] if len(series) >= 2 else np.nan
                roll3 = float(np.nanmean(series[-3:])) if len(series) else np.nan
                trend = lag1 - lag2 if np.isfinite(lag1) and np.isfinite(lag2) else 0.0
                row = pd.DataFrame([{
                    "ano": year,
                    "t": year - first_year,
                    "lag1": lag1,
                    "lag2": lag2,
                    "roll3": roll3,
                    "trend": trend,
                    "uf": uf,
                    "regiao": reg,
                    "cod_uf": str(mun)[:2],
                }])
                yhat = float(model.predict(row[feat_num + feat_cat])[0])
                yhat = max(0.0, min(1.0, yhat))
                series.append(yhat)

                all_forecasts.append({
                    "localidade_id": mun,
                    "localidade_nome": nome,
                    "uf": uf,
                    "regiao": reg,
                    "indice": target,
                    "ano": year,
                    "valor_previsto": yhat,
                    "ultimo_valor_historico": last_value,
                    "delta_vs_ultimo_historico": yhat - last_value,
                    "tendencia": "melhora" if yhat >= last_value else "piora",
                })

    pred_df = pd.DataFrame(all_forecasts)
    metrics_df = pd.DataFrame(metrics)

    if not pred_df.empty:
        max_pred_year = int(pred_df["ano"].max())
        max_hist_year = int(df["ano"].max())
        anos_previstos = max_pred_year - max_hist_year
        pred_filename = f"previsoes_{anos_previstos}_anos.csv"
        pred_df.to_csv(out / pred_filename, index=False, encoding="utf-8-sig")
        pred_df.to_csv(out / "previsoes_5_anos.csv", index=False, encoding="utf-8-sig")

        last_future = pred_df["ano"].max()
        ranking_last = pred_df[pred_df["ano"] == last_future].sort_values(
            ["indice", "valor_previsto"], ascending=[True, False]
        )
        ranking_last.to_csv(out / "ranking_previsao_ultimo_ano.csv", index=False, encoding="utf-8-sig")

    metrics_df.to_csv(out / "metricas_validacao_previsao.csv", index=False, encoding="utf-8-sig")
    return pred_df, metrics_df


def make_forecast_series(idx, pred_df, out):
    index_cols = [c for c in idx.columns if c.startswith("idx_")]
    hist_long = idx.melt(
        id_vars=["localidade_id", "localidade_nome", "uf", "regiao", "ano"],
        value_vars=index_cols,
        var_name="indice",
        value_name="valor",
    )
    hist_long["tipo"] = "historico"
    hist_long["tendencia"] = "historico"

    if pred_df is None or pred_df.empty:
        hist_long.to_csv(out / "series_historico_previsao.csv", index=False, encoding="utf-8-sig")
        return hist_long

    fut = pred_df.rename(columns={"valor_previsto": "valor"})[
        ["localidade_id", "localidade_nome", "uf", "regiao", "ano", "indice", "valor", "tendencia"]
    ].copy()
    fut["tipo"] = "previsao"
    all_series = pd.concat([hist_long, fut], ignore_index=True)
    all_series.to_csv(out / "series_historico_previsao.csv", index=False, encoding="utf-8-sig")
    return all_series


def plot_forecasts(idx, pred_df, out, max_municipios=None):
    plot_dir = out / "graficos_previsao"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if pred_df is None or pred_df.empty:
        return

    index_cols = [c for c in idx.columns if c.startswith("idx_")]
    munics = sorted(pred_df["localidade_id"].unique())
    if max_municipios:
        munics = munics[:max_municipios]

    for mun in munics:
        hist_m = idx[idx["localidade_id"] == mun].sort_values("ano")
        if hist_m.empty:
            continue
        if hist_m["localidade_nome"].notna().any():
            nome = str(hist_m["localidade_nome"].dropna().iloc[-1])
        else:
            nome = str(mun)
        safe_nome = safe_filename(nome, 80)
        mdir = plot_dir / f"{safe_filename(mun, 20)}_{safe_nome}"
        mdir.mkdir(parents=True, exist_ok=True)

        for ind in index_cols:
            h = hist_m[["ano", ind]].dropna()
            f = pred_df[
                (pred_df["localidade_id"] == mun) & (pred_df["indice"] == ind)
            ].sort_values("ano")
            if len(h) < 2 or f.empty:
                continue
            last_hist = float(h[ind].iloc[-1])
            last_pred = float(f["valor_previsto"].iloc[-1])
            color = "green" if last_pred >= last_hist else "red"
            label_pred = "Previsão IA: melhora" if color == "green" else "Previsão IA: piora"

            plt.figure(figsize=(9, 4.8))
            plt.plot(h["ano"], h[ind], marker="o", color="black", label="Histórico MUNIC")
            fx = [int(h["ano"].iloc[-1])] + list(f["ano"].astype(int))
            fy = [last_hist] + list(f["valor_previsto"].astype(float))
            plt.plot(fx, fy, marker="o", color=color, label=label_pred)
            plt.ylim(-0.03, 1.03)
            plt.title(f"{nome} - {ind} - previsão")
            plt.xlabel("Ano")
            plt.ylabel("Índice sintético MUNIC")
            plt.grid(True, alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(mdir / f"{safe_filename(ind)}.png", dpi=160)
            plt.close()

    agg_dir = plot_dir / "agregado_regional"
    agg_dir.mkdir(exist_ok=True)
    for ind in index_cols:
        h = idx.groupby(["regiao", "ano"], as_index=False)[ind].mean()
        f = pred_df[pred_df["indice"] == ind].groupby(["regiao", "ano"], as_index=False)["valor_previsto"].mean()
        for reg in sorted(h["regiao"].dropna().unique()):
            hh = h[h["regiao"] == reg].sort_values("ano")
            ff = f[f["regiao"] == reg].sort_values("ano")
            if len(hh) < 2 or ff.empty:
                continue
            last_h = float(hh[ind].iloc[-1])
            last_f = float(ff["valor_previsto"].iloc[-1])
            color = "green" if last_f >= last_h else "red"

            plt.figure(figsize=(9, 4.8))
            plt.plot(hh["ano"], hh[ind], marker="o", color="black", label="Histórico médio")
            fx = [int(hh["ano"].iloc[-1])] + list(ff["ano"].astype(int))
            fy = [last_h] + list(ff["valor_previsto"].astype(float))
            plt.plot(fx, fy, marker="o", color=color, label="Previsão média")
            plt.ylim(-0.03, 1.03)
            plt.title(f"{reg} - {ind} - previsão média regional")
            plt.xlabel("Ano")
            plt.ylabel("Índice sintético MUNIC")
            plt.grid(True, alpha=0.25)
            plt.legend()
            plt.tight_layout()
            safe_reg = safe_filename(reg)
            safe_ind = safe_filename(ind)
            filename = f"{safe_reg}_{safe_ind}.png"
            plt.savefig(agg_dir / filename, dpi=160)
            plt.close()


def write_forecast_report(out, idx, pred_df, metrics, regiao, horizonte, quality_metrics=None):
    lines = []
    lines.append("# Relatório de previsão MUNIC com IA\n")
    lines.append(f"Horizonte de previsão: {horizonte} anos.\n")
    lines.append(f"Recorte regional: {regiao or 'Brasil'}.\n")
    lines.append("\n## Metodologia\n")
    lines.append("1. Os indicadores da MUNIC foram agrupados por temas: saúde, educação, segurança, governança, tecnologia, gestão, participação e meio ambiente.\n")
    lines.append("2. Cada tema gerou um índice sintético entre 0 e 1, pela proporção de respostas positivas/estruturantes disponíveis no banco.\n")
    lines.append("3. O índice de qualidade de vida é uma composição ponderada desses índices temáticos. Ele não é IDHM oficial; é um proxy administrativo-institucional derivado da MUNIC.\n")
    lines.append("4. Para cada índice foi treinado um RandomForestRegressor global, usando ano, UF, região, defasagens temporais, média móvel e tendência recente.\n")
    lines.append("5. A previsão é recursiva: cada ano previsto alimenta o ano seguinte.\n")
    lines.append("\n## Interpretação das cores\n")
    lines.append("- Preto: anos históricos observados na MUNIC.\n")
    lines.append("- Verde: previsão maior ou igual ao último valor histórico, indicando melhora.\n")
    lines.append("- Vermelho: previsão menor que o último valor histórico, indicando piora.\n")

    if quality_metrics:
        lines.append("\n## Métricas do modelo de qualidade de vida\n")
        lines.append("```json\n")
        lines.append(json.dumps(quality_metrics, ensure_ascii=False, indent=2))
        lines.append("\n```\n")

    if metrics is not None and not metrics.empty:
        lines.append("\n## Validação temporal dos modelos de previsão\n")
        lines.append(metrics.to_markdown(index=False))
        lines.append("\n")

    if pred_df is not None and not pred_df.empty:
        last_year = int(pred_df["ano"].max())
        lines.append(f"\n## Último ano previsto: {last_year}\n")
        qv = pred_df[(pred_df["ano"] == last_year) & (pred_df["indice"] == "idx_qualidade_vida")].copy()
        if not qv.empty:
            lines.append("\n### Top 20 municípios previstos em qualidade de vida\n")
            top = qv.sort_values("valor_previsto", ascending=False).head(20)
            lines.append(top[["localidade_id", "localidade_nome", "uf", "regiao", "valor_previsto", "tendencia"]].to_markdown(index=False))
            lines.append("\n")
            lines.append("\n### 20 municípios com maior queda prevista em qualidade de vida\n")
            down = qv.sort_values("delta_vs_ultimo_historico", ascending=True).head(20)
            lines.append(down[["localidade_id", "localidade_nome", "uf", "regiao", "ultimo_valor_historico", "valor_previsto", "delta_vs_ultimo_historico"]].to_markdown(index=False))
            lines.append("\n")

    (Path(out) / "RELATORIO_PREVISAO_MUNIC.md").write_text("\n".join(lines), encoding="utf-8")


def write_method_report(out, metrics, tf, idx, regiao=None):
    lines = []
    lines.append("# Relatório metodológico - MUNIC Forecast ML v4.2\n\n")
    lines.append("## Objetivo\n")
    lines.append("Comparar municípios por região brasileira, calcular índices sintéticos, identificar features associadas à qualidade de vida administrativa municipal e prever os próximos anos.\n\n")
    lines.append("## Arquivos principais\n")
    lines.append("- `indices_municipio_ano.csv`: índices por município, UF, região e ano.\n")
    lines.append("- `previsoes_5_anos.csv`: previsões para todos os municípios e todos os índices.\n")
    lines.append("- `series_historico_previsao.csv`: série longa com histórico + previsão.\n")
    lines.append("- `graficos_previsao/`: gráficos individuais e agregados regionais.\n")
    lines.append("- `modelos_previsao/`: modelos `.joblib` por índice.\n")
    lines.append("- `features_importantes_modelo.csv`: features mais importantes para qualidade de vida.\n")
    lines.append("- `RELATORIO_PREVISAO_MUNIC.md`: relatório final.\n\n")
    lines.append("## Métricas do modelo de qualidade de vida\n")
    lines.append("```json\n")
    lines.append(json.dumps(metrics, ensure_ascii=False, indent=2))
    lines.append("\n```\n\n")
    lines.append("## Cobertura por tema\n")
    for k, v in tf.items():
        lines.append(f"- {k}: {len(v)} indicadores candidatos encontrados por palavras-chave.\n")
    lines.append("\n## Municípios e regiões no banco\n")
    lines.append(f"- Municípios distintos: {idx['localidade_id'].nunique()}\n")
    lines.append(f"- Regiões: {', '.join(sorted(idx['regiao'].dropna().unique()))}\n")
    if idx["localidade_id"].nunique() <= 2:
        lines.append("\n**Alerta:** o SQLite analisado possui poucos municípios válidos. Para comparação regional real, carregue todos os municípios/UFs no banco `resultados`. O programa já está preparado para isso.\n")
    if regiao:
        lines.append(f"\nFiltro regional solicitado: **{regiao}**.\n")
    (Path(out) / "RELATORIO_METODOLOGICO.md").write_text("".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Caminho do SQLite da MUNIC")
    ap.add_argument("--out", default="saida_munic_forecast", help="Pasta de saída")
    ap.add_argument("--regiao", default="Brasil", help="Brasil, Sudeste, Sul, Nordeste, Norte ou Centro-Oeste")
    ap.add_argument("--horizonte", type=int, default=5, help="Quantidade de anos futuros")
    ap.add_argument("--municipios", default=None, help="Lista de códigos separados por vírgula para comparação temporal")
    ap.add_argument("--idh_csv", default=None, help="CSV externo opcional com colunas localidade_id/ano/idhm")
    ap.add_argument("--max_graficos_municipios", type=int, default=None, help="Limita gráficos individuais por município. O CSV sempre sai completo.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ind, res = load(args.db)
    names = feature_names(ind)
    X, _ = build_matrix(ind, res)
    idx, tf = compute_indices(X, names, out)

    if args.regiao and args.regiao.lower() != "brasil":
        idx_run = idx[idx["regiao"].str.lower() == args.regiao.lower()].copy()
    else:
        idx_run = idx.copy()

    if idx_run.empty:
        print(f"Nenhum município encontrado para a região: {args.regiao}")
        return

    save_indices(idx_run, out)
    plot_evolucao_regioes(idx_run, out)
    compare_region(idx_run, out, regiao=args.regiao)
    compare_selected_municipios(idx_run, out, args.municipios)
    cluster_last_year(idx_run, out, regiao="Brasil")
    correlation_heatmap(idx_run, out, regiao="Brasil")

    # Modelo explicativo da qualidade de vida com as features originais.
    # O modelo explicativo usa todos os dados carregados; as saídas são úteis para interpretar fatores associados.
    X_geo = X.merge(idx[["localidade_id", "ano", "regiao"]], on=["localidade_id", "ano"], how="left")
    if args.regiao and args.regiao.lower() != "brasil":
        selected_ids = set(idx_run["localidade_id"].astype(str))
        X_geo = X_geo[X_geo["localidade_id"].astype(str).isin(selected_ids)].copy()
    data_model, imp, quality_metrics = train_quality_models(X_geo.drop(columns=["regiao"], errors="ignore"), idx_run, out, names, idh_csv=args.idh_csv)
    plot_features(imp, names, out)

    pred_df, forecast_metrics = forecast_all_indices(idx_run, out, horizonte=args.horizonte)
    make_forecast_series(idx_run, pred_df, out)
    plot_forecasts(idx_run, pred_df, out, max_municipios=args.max_graficos_municipios)
    write_forecast_report(out, idx_run, pred_df, forecast_metrics, args.regiao, args.horizonte, quality_metrics)
    write_method_report(out, quality_metrics, tf, idx_run, regiao=args.regiao)

    print("OK - previsão MUNIC gerada em:", out.resolve())
    print("Arquivos principais:")
    print("- indices_municipio_ano.csv")
    print("- previsoes_5_anos.csv")
    print("- series_historico_previsao.csv")
    print("- graficos_previsao/")
    print("- modelos_previsao/")
    print("- RELATORIO_PREVISAO_MUNIC.md")
    print("- RELATORIO_METODOLOGICO.md")


if __name__ == "__main__":
    main()
