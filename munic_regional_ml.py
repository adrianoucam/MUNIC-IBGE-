# -*- coding: utf-8 -*-
"""
MUNIC Regional ML - v3

Objetivo:
- Calcular índices municipais por ano a partir do BD SQLite da MUNIC.
- Separar municípios por UF e Região do Brasil.
- Comparar municípios de uma região, por exemplo Sudeste.
- Treinar modelos Random Forest para identificar features associadas à qualidade de vida.
- Gerar gráficos regionais, rankings e arquivos CSV.

Observação:
O índice de qualidade de vida é um índice sintético derivado da presença de instrumentos,
planos, conselhos, estruturas, sistemas e políticas públicas capturadas na MUNIC. Não é IDHM oficial.
"""

import argparse
import json
import re
import sqlite3
import unicodedata
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

THEMES = {
    "saude": ["saude", "hospital", "medic", "unidade basica", "ubs", "vigilancia", "vacina", "sanit", "epidemi", "atencao basica", "sus"],
    "educacao": ["educacao", "escola", "ensino", "professor", "creche", "fundeb", "merenda", "aluno", "biblioteca", "transporte escolar"],
    "seguranca": ["seguranca", "guarda", "defesa civil", "violencia", "risco", "desastre", "protecao", "bombeiro", "enchente", "deslizamento"],
    "governanca": ["transparencia", "controle", "ouvidoria", "conselho", "audiencia", "lei de acesso", "lai", "fiscalizacao", "prestacao", "governanca"],
    "tecnologia": ["internet", "tecnologia", "informatica", " sistema", "digital", "eletronico", "site", "portal", "computador", "software", "geoprocessamento"],
    "gestao": ["planejamento", "plano", "orcamento", "pessoal", "servidor", "capacitacao", "estrutura administrativa", "gestao", "consorcio", "cadastro", "administracao"],
    "participacao": ["participacao", "conselho", "conferencia", "consulta publica", "audiencia", "forum", "comite", "representante"],
    "meio_ambiente": ["meio ambiente", "ambiental", "saneamento", "residuo", "lixo", "agua", "esgoto", "drenagem", "clima", "defesa ambiental", "coleta seletiva"],
}

UF_INFO = {
    "11": ("RO", "Norte"), "12": ("AC", "Norte"), "13": ("AM", "Norte"), "14": ("RR", "Norte"), "15": ("PA", "Norte"), "16": ("AP", "Norte"), "17": ("TO", "Norte"),
    "21": ("MA", "Nordeste"), "22": ("PI", "Nordeste"), "23": ("CE", "Nordeste"), "24": ("RN", "Nordeste"), "25": ("PB", "Nordeste"), "26": ("PE", "Nordeste"), "27": ("AL", "Nordeste"), "28": ("SE", "Nordeste"), "29": ("BA", "Nordeste"),
    "31": ("MG", "Sudeste"), "32": ("ES", "Sudeste"), "33": ("RJ", "Sudeste"), "35": ("SP", "Sudeste"),
    "41": ("PR", "Sul"), "42": ("SC", "Sul"), "43": ("RS", "Sul"),
    "50": ("MS", "Centro-Oeste"), "51": ("MT", "Centro-Oeste"), "52": ("GO", "Centro-Oeste"), "53": ("DF", "Centro-Oeste"),
}

MISSING = {"", "nan", "none", "null", "-", "...", "ignorado", "não sabe", "nao sabe", "não informado", "nao informado"}


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


def norm(s):
    return re.sub(r"\s+", " ", strip_accents(str(s or "")).lower().strip())


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
        return ("NA", "Não identificada")
    return UF_INFO.get(code[:2], ("NA", "Não identificada"))


def load(db_path):
    con = sqlite3.connect(db_path)
    ind = pd.read_sql("select pesquisa_id, periodo, indicador_id, posicao, indicador_nome, classe from indicadores", con)
    res = pd.read_sql("select pesquisa_id, periodo, indicador_id, posicao, localidade_id, localidade_nome, valor from resultados", con)
    con.close()
    res["localidade_id"] = res["localidade_id"].astype(str).str.strip()
    res = res[res["localidade_id"].ne("")].copy()
    res["ano"] = pd.to_numeric(res["periodo"], errors="coerce").astype("Int64")
    ind["feature"] = "i_" + ind["indicador_id"].astype(str) + "__" + ind["posicao"].astype(str).str.replace(r"[^0-9A-Za-z_]+", "_", regex=True)
    return ind, res


def feature_names(ind):
    return dict(zip(ind["feature"], ind["indicador_nome"].astype(str)))


def build_matrix(ind, res):
    meta = ind[["pesquisa_id", "periodo", "indicador_id", "posicao", "feature", "indicador_nome"]].drop_duplicates()
    df = res.merge(meta, on=["pesquisa_id", "periodo", "indicador_id", "posicao"], how="left")
    df["valor_bin"] = df["valor"].map(encode_binary)
    df["valor_num"] = df["valor"].map(encode_numeric)

    bin_piv = df.pivot_table(index=["localidade_id", "ano"], columns="feature", values="valor_bin", aggfunc="mean").reset_index()
    num_piv = df.pivot_table(index=["localidade_id", "ano"], columns="feature", values="valor_num", aggfunc="mean").reset_index()
    num_piv = num_piv.rename(columns={c: "num_" + c for c in num_piv.columns if c not in ["localidade_id", "ano"]})
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
        diag.append({"tema": theme, "indicadores_por_palavra_chave": len(cols), "indicadores_com_resposta_binaria": len(valid_cols)})
        if not valid_cols:
            idx[f"idx_{theme}"] = np.nan
            idx[f"n_{theme}"] = 0
        else:
            idx[f"n_{theme}"] = X[valid_cols].notna().sum(axis=1)
            idx[f"idx_{theme}"] = X[valid_cols].mean(axis=1, skipna=True)

    pd.DataFrame(diag).to_csv(out / "diagnostico_indicadores_temas.csv", index=False, encoding="utf-8-sig")

    weights = {
        "idx_saude": .18,
        "idx_educacao": .18,
        "idx_seguranca": .16,
        "idx_governanca": .13,
        "idx_tecnologia": .10,
        "idx_gestao": .10,
        "idx_participacao": .07,
        "idx_meio_ambiente": .08,
    }
    for c in weights:
        if c not in idx:
            idx[c] = np.nan
    filled = []
    for c, w in weights.items():
        m = idx[c].mean(skipna=True)
        filled.append(idx[c].fillna(m if pd.notna(m) else 0.0) * w)
    idx["idx_qualidade_vida"] = sum(filled) / sum(weights.values())

    if idx["idx_qualidade_vida"].nunique(dropna=True) >= 3:
        idx["classe_maturidade"] = pd.qcut(idx["idx_qualidade_vida"].rank(method="first"), q=3, labels=False, duplicates="drop")
    else:
        idx["classe_maturidade"] = pd.cut(idx["idx_qualidade_vida"], bins=[-0.01, .33, .66, 1.01], labels=[0, 1, 2])
    idx["classe_maturidade"] = idx["classe_maturidade"].astype("Int64").fillna(0).astype(int)
    return idx, tf


def normalize_numeric_features(data, feature_cols):
    for c in feature_cols:
        if c.startswith("num_"):
            s = pd.to_numeric(data[c], errors="coerce")
            if s.notna().sum() >= 2 and s.max() != s.min():
                data[c] = (s - s.min()) / (s.max() - s.min())
            else:
                data[c] = np.nan
    return data


def train_models(X, idx, out, names, idh_csv=None):
    geo_cols = ["localidade_id", "ano", "idx_qualidade_vida", "classe_maturidade"]
    data = X.merge(idx[geo_cols], on=["localidade_id", "ano"], how="inner")

    if idh_csv:
        idh = pd.read_csv(idh_csv)
        idh = idh.rename(columns={"municipio": "localidade_id", "cod_mun": "localidade_id", "codigo_municipio": "localidade_id", "year": "ano", "periodo": "ano", "IDHM": "idhm", "idh": "idhm"})
        idh["localidade_id"] = idh["localidade_id"].astype(str)
        idh["ano"] = pd.to_numeric(idh["ano"], errors="coerce").astype("Int64")
        data = data.merge(idh[["localidade_id", "ano", "idhm"]], on=["localidade_id", "ano"], how="left")

    feature_cols = [c for c in X.columns if c not in ["localidade_id", "ano", "localidade_nome"] and data[c].notna().mean() >= 0.03]
    data = normalize_numeric_features(data, feature_cols)
    feature_cols = [c for c in feature_cols if data[c].notna().sum() > 0]

    Xf = data[feature_cols]
    y = data["idx_qualidade_vida"]
    reg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=500, random_state=42, min_samples_leaf=1)),
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
    joblib.dump({"pipeline": reg, "feature_cols": feature_cols, "target": "idx_qualidade_vida"}, out / "modelo_rf_qualidade_vida.joblib")

    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced")),
    ])
    clf.fit(Xf, data["classe_maturidade"])
    joblib.dump({"pipeline": clf, "feature_cols": feature_cols, "target": "classe_maturidade"}, out / "modelo_rf_classe_maturidade.joblib")

    if "idhm" in data.columns and data["idhm"].notna().sum() >= 5:
        d2 = data[data["idhm"].notna()].copy()
        idhm_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=500, random_state=42)),
        ])
        idhm_model.fit(d2[feature_cols], d2["idhm"])
        joblib.dump({"pipeline": idhm_model, "feature_cols": feature_cols, "target": "idhm"}, out / "modelo_rf_idhm.joblib")
        metrics["idhm_model"] = {"n_samples": int(len(d2)), "r2_train": float(r2_score(d2["idhm"], idhm_model.predict(d2[feature_cols]))) if len(d2) > 1 else None}

    imp = pd.DataFrame({"feature": feature_cols, "importance": reg.named_steps["model"].feature_importances_}).sort_values("importance", ascending=False)
    imp.to_csv(out / "features_importantes_modelo.csv", index=False, encoding="utf-8-sig")
    with open(out / "metricas_modelo.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return data, imp, metrics


def save_indices(idx, out):
    idx_cols = [
        "localidade_id", "localidade_nome", "uf", "regiao", "ano",
        "idx_saude", "idx_educacao", "idx_seguranca", "idx_governanca", "idx_tecnologia", "idx_gestao", "idx_participacao", "idx_meio_ambiente", "idx_qualidade_vida", "classe_maturidade",
    ]
    for c in idx_cols:
        if c not in idx.columns:
            idx[c] = "" if c in {"localidade_nome", "uf", "regiao"} else np.nan
    idx[idx_cols].to_csv(out / "indices_municipio_ano_regional.csv", index=False, encoding="utf-8-sig")
    ncols = [c for c in idx.columns if c.startswith("n_")]
    idx[["localidade_id", "ano", "uf", "regiao"] + ncols].to_csv(out / "cobertura_indicadores_por_tema.csv", index=False, encoding="utf-8-sig")


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
        plt.grid(True, alpha=.3)
        plt.tight_layout()
        plt.savefig(out / f"evolucao_regioes_{col}.png", dpi=160)
        plt.close()


def compare_region(idx, out, regiao=None, ano=None, top=30):
    d = idx.copy()
    if regiao:
        d = d[d["regiao"].str.lower() == regiao.lower()]
    if ano is None:
        ano = int(d["ano"].max()) if len(d) else None
    if ano is not None:
        d = d[d["ano"] == ano]

    if d.empty:
        return None

    tag = (regiao or "Brasil").replace(" ", "_")
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
    cols = ["idx_qualidade_vida", "idx_saude", "idx_educacao", "idx_seguranca", "idx_governanca", "idx_tecnologia"]
    for col in cols:
        plt.figure(figsize=(11, 6))
        for mun, dm in d.groupby("localidade_id"):
            name = str(dm["localidade_nome"].dropna().iloc[0]) if dm["localidade_nome"].notna().any() and str(dm["localidade_nome"].dropna().iloc[0]).strip() else mun
            dm = dm.sort_values("ano")
            plt.plot(dm["ano"], dm[col], marker="o", label=name)
        plt.ylim(0, 1)
        plt.title(f"Comparação temporal - {col}")
        plt.xlabel("Ano")
        plt.ylabel("Índice 0-1")
        plt.legend()
        plt.grid(True, alpha=.3)
        plt.tight_layout()
        plt.savefig(out / f"comparacao_municipios_{col}.png", dpi=160)
        plt.close()


def plot_features(imp, names, out):
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


def cluster_last_year(idx, out, regiao=None, k=4):
    cols = ["idx_saude", "idx_educacao", "idx_seguranca", "idx_governanca", "idx_tecnologia", "idx_gestao", "idx_participacao", "idx_meio_ambiente"]
    d = idx.copy()
    if regiao:
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
    plt.grid(True, alpha=.25)
    plt.tight_layout()
    plt.savefig(out / "clusters_pca_ultimo_ano.png", dpi=160)
    plt.close()


def correlation_heatmap(idx, out, regiao=None):
    cols = ["idx_saude", "idx_educacao", "idx_seguranca", "idx_governanca", "idx_tecnologia", "idx_gestao", "idx_participacao", "idx_meio_ambiente", "idx_qualidade_vida"]
    d = idx.copy()
    if regiao:
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


def report(out, metrics, tf, idx, regiao=None):
    lines = []
    lines.append("# Relatório metodológico - MUNIC Regional ML v3\n\n")
    lines.append("## Objetivo\nComparar municípios por região brasileira, calcular índices sintéticos e identificar features associadas à qualidade de vida administrativa municipal.\n\n")
    lines.append("## Arquivos principais\n")
    lines.append("- `indices_municipio_ano_regional.csv`: índices por município, UF, região e ano.\n")
    lines.append("- `media_regiao_ano.csv`: evolução média por região.\n")
    lines.append("- `ranking_<REGIAO>_<ANO>.csv`: ranking regional no último ano ou ano escolhido.\n")
    lines.append("- `top_features_interpretadas.csv`: indicadores MUNIC mais associados ao índice.\n")
    lines.append("- `modelo_rf_qualidade_vida.joblib`: modelo Random Forest salvo.\n")
    lines.append("- `clusters_ultimo_ano.csv`: clusterização municipal no último ano disponível.\n\n")
    lines.append("## Métricas do modelo\n```json\n" + json.dumps(metrics, ensure_ascii=False, indent=2) + "\n```\n\n")
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
    (out / "RELATORIO_REGIONAL.md").write_text("".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Caminho do SQLite da MUNIC")
    ap.add_argument("--out", default="saida_munic_regional", help="Pasta de saída")
    ap.add_argument("--regiao", default="Sudeste", help="Região para ranking/comparação: Sudeste, Sul, Nordeste, Norte, Centro-Oeste ou Brasil")
    ap.add_argument("--ano", type=int, default=None, help="Ano para ranking; padrão = último ano disponível na região")
    ap.add_argument("--top", type=int, default=30, help="Quantidade de municípios no gráfico de ranking")
    ap.add_argument("--municipios", default=None, help="Lista de códigos separados por vírgula para comparação temporal")
    ap.add_argument("--idh_csv", default=None, help="CSV opcional com colunas localidade_id, ano, idhm")
    ap.add_argument("--clusters", type=int, default=4, help="Número máximo de clusters")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    regiao = None if args.regiao.lower() == "brasil" else args.regiao

    ind, res = load(args.db)
    X, df = build_matrix(ind, res)
    names = feature_names(ind)
    X.to_csv(out / "matriz_features_municipio_ano.csv", index=False, encoding="utf-8-sig")

    idx, tf = compute_indices(X, names, out)
    save_indices(idx, out)

    data, imp, metrics = train_models(X, idx, out, names, args.idh_csv)
    plot_evolucao_regioes(idx, out)
    compare_region(idx, out, regiao=regiao, ano=args.ano, top=args.top)
    compare_selected_municipios(idx, out, args.municipios)
    plot_features(imp, names, out)
    cluster_last_year(idx, out, regiao=regiao, k=args.clusters)
    correlation_heatmap(idx, out, regiao=regiao)
    report(out, metrics, tf, idx, regiao=regiao)

    print("[OK] MUNIC Regional ML v3 concluído")
    print("Saída:", out.resolve())
    print("Arquivo principal: indices_municipio_ano_regional.csv")
    print("Ranking regional: ranking_<regiao>_<ano>.csv")


if __name__ == "__main__":
    main()
