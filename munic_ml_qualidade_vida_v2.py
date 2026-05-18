
# -*- coding: utf-8 -*-
"""
MUNIC ML - Qualidade de vida v2 corrigida

Correção principal da v1:
- A v1 fazia normalização min-max por indicador. Quando o banco contém apenas um município
  (como o arquivo anexado, com 330330), muitos indicadores ficam constantes e viram 0.5.
- Esta v2 calcula os índices principalmente por presença/ausência temática (Sim=1, Não=0),
  o que gera variação temporal mesmo com um único município.

Saídas principais:
- matriz_features_municipio_ano.csv
- indices_municipio_ano.csv
- diagnostico_indicadores_temas.csv
- modelo_rf_qualidade_vida.joblib
- modelo_rf_classe_maturidade.joblib
- features_importantes_modelo.csv
- top_features_interpretadas.csv
- gráficos PNG
"""
import argparse, json, re, sqlite3, unicodedata
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

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

NEG = {"nao", "não", "n", "inexistente", "nao possui", "não possui", "sem", "ausente"}
POS = {"sim", "s", "possui", "existe", "existente", "implantado", "implementado", "realiza", "realizado", "ativo", "funciona"}
MISSING = {"", "nan", "none", "null", "-", "...", "ignorado", "não sabe", "nao sabe", "não informado", "nao informado"}

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(s)) if not unicodedata.combining(c))

def norm(s):
    return re.sub(r"\s+", " ", strip_accents(str(s or "")).lower().strip())

def encode_binary(v):
    """Codifica respostas qualitativas em 0/1. Números são tratados em função separada."""
    if pd.isna(v): return np.nan
    raw = str(v).strip()
    s = norm(raw)
    if s in MISSING: return np.nan
    if s in {"sim", "s"}: return 1.0
    if s in {"nao", "não", "n"}: return 0.0
    if any(k in s for k in ["nao possui", "não possui", "inexist", "sem ", "ausencia", "ausente"]): return 0.0
    if any(k in s for k in ["possui", "existe", "existente", "implant", "implement", "realiza", "ativo", "funciona"]): return 1.0
    # escolaridade do prefeito/gestor, quando aparecer
    if "superior" in s or "pos-gradu" in s or "pós-gradu" in s: return 1.0
    if "medio" in s or "médio" in s: return 0.65
    if "fundamental" in s: return 0.35
    if "sem instr" in s: return 0.05
    return np.nan

def encode_numeric(v):
    if pd.isna(v): return np.nan
    s=str(v).strip().replace('.', '').replace(',', '.')
    if norm(s) in MISSING: return np.nan
    try: return float(s)
    except Exception: return np.nan

def load(db_path):
    con=sqlite3.connect(db_path)
    ind=pd.read_sql("select pesquisa_id, periodo, indicador_id, posicao, indicador_nome, classe from indicadores", con)
    res=pd.read_sql("select pesquisa_id, periodo, indicador_id, posicao, localidade_id, localidade_nome, valor from resultados", con)
    con.close()
    res['localidade_id']=res['localidade_id'].astype(str).str.strip()
    res=res[res['localidade_id'].ne('')].copy()
    res['ano']=pd.to_numeric(res['periodo'], errors='coerce').astype('Int64')
    ind['feature']='i_'+ind['indicador_id'].astype(str)+'__'+ind['posicao'].astype(str).str.replace(r'[^0-9A-Za-z_]+','_',regex=True)
    return ind,res

def feature_names(ind):
    return dict(zip(ind['feature'], ind['indicador_nome'].astype(str)))

def build_matrix(ind,res):
    meta=ind[['pesquisa_id','periodo','indicador_id','posicao','feature','indicador_nome']].drop_duplicates()
    df=res.merge(meta,on=['pesquisa_id','periodo','indicador_id','posicao'],how='left')
    df['valor_bin']=df['valor'].map(encode_binary)
    df['valor_num']=df['valor'].map(encode_numeric)
    bin_piv=df.pivot_table(index=['localidade_id','ano'], columns='feature', values='valor_bin', aggfunc='mean').reset_index()
    num_piv=df.pivot_table(index=['localidade_id','ano'], columns='feature', values='valor_num', aggfunc='mean').reset_index()
    # prefixa matriz numérica para não confundir; modelos usam binário + numérico normalizado
    num_piv=num_piv.rename(columns={c:'num_'+c for c in num_piv.columns if c not in ['localidade_id','ano']})
    X=bin_piv.merge(num_piv,on=['localidade_id','ano'],how='outer')
    return X, df

def theme_features(names):
    out={k:[] for k in THEMES}
    for f,nm in names.items():
        n=norm(nm)
        for theme,keys in THEMES.items():
            if any(k in n for k in keys): out[theme].append(f)
    return out

def compute_indices(X,names,out):
    tf=theme_features(names)
    idx=X[['localidade_id','ano']].copy()
    diag=[]
    for theme, cols in tf.items():
        cols=[c for c in cols if c in X.columns]
        valid_cols=[c for c in cols if X[c].notna().sum()>0]
        diag.append({'tema':theme,'indicadores_por_palavra_chave':len(cols),'indicadores_com_resposta_binaria':len(valid_cols)})
        if not valid_cols:
            idx[f'idx_{theme}']=np.nan
            idx[f'n_{theme}']=0
        else:
            idx[f'n_{theme}']=X[valid_cols].notna().sum(axis=1)
            # Média das respostas positivas entre indicadores disponíveis no tema.
            idx[f'idx_{theme}']=X[valid_cols].mean(axis=1, skipna=True)
    pd.DataFrame(diag).to_csv(out/'diagnostico_indicadores_temas.csv', index=False, encoding='utf-8-sig')
    comps=['idx_saude','idx_educacao','idx_seguranca','idx_governanca','idx_tecnologia','idx_gestao','idx_participacao','idx_meio_ambiente']
    weights={'idx_saude':.18,'idx_educacao':.18,'idx_seguranca':.16,'idx_governanca':.13,'idx_tecnologia':.10,'idx_gestao':.10,'idx_participacao':.07,'idx_meio_ambiente':.08}
    for c in comps:
        if c not in idx: idx[c]=np.nan
    # Se algum tema faltar em determinado ano, substitui pela média do próprio tema no período analisado.
    filled=[]
    for c in comps:
        m=idx[c].mean(skipna=True)
        filled.append(idx[c].fillna(m if pd.notna(m) else 0.0)*weights[c])
    idx['idx_qualidade_vida']=sum(filled)/sum(weights.values())
    # Classes por tercis; se houver poucos dados, usa cortes fixos.
    if idx['idx_qualidade_vida'].nunique(dropna=True)>=3:
        idx['classe_maturidade']=pd.qcut(idx['idx_qualidade_vida'].rank(method='first'), q=3, labels=False, duplicates='drop')
    else:
        idx['classe_maturidade']=pd.cut(idx['idx_qualidade_vida'], bins=[-0.01,.33,.66,1.01], labels=[0,1,2])
    idx['classe_maturidade']=idx['classe_maturidade'].astype('Int64').fillna(0).astype(int)
    return idx, tf

def normalize_numeric_features(data, feature_cols):
    # Mantém binários como estão; normaliza numéricos em 0-1 por coluna, preservando NaN.
    for c in feature_cols:
        if c.startswith('num_'):
            s=pd.to_numeric(data[c], errors='coerce')
            if s.notna().sum()>=2 and s.max()!=s.min():
                data[c]=(s-s.min())/(s.max()-s.min())
            else:
                data[c]=np.nan
    return data

def train(X,idx,out,names,idh_csv=None):
    data=X.merge(idx,on=['localidade_id','ano'],how='inner')
    if idh_csv:
        idh=pd.read_csv(idh_csv)
        idh=idh.rename(columns={'municipio':'localidade_id','cod_mun':'localidade_id','codigo_municipio':'localidade_id','year':'ano','periodo':'ano','IDHM':'idhm'})
        idh['localidade_id']=idh['localidade_id'].astype(str)
        idh['ano']=pd.to_numeric(idh['ano'],errors='coerce').astype('Int64')
        data=data.merge(idh[['localidade_id','ano','idhm']],on=['localidade_id','ano'],how='left')
    feature_cols=[c for c in X.columns if c not in ['localidade_id','ano'] and data[c].notna().mean()>=0.03]
    data=normalize_numeric_features(data, feature_cols)
    # Remove colunas que ficaram totalmente vazias após normalização
    feature_cols=[c for c in feature_cols if data[c].notna().sum()>0]
    Xf=data[feature_cols]
    y=data['idx_qualidade_vida']
    reg=Pipeline([('imputer',SimpleImputer(strategy='median')),('model',RandomForestRegressor(n_estimators=500, random_state=42, min_samples_leaf=1))])
    reg.fit(Xf,y)
    pred=reg.predict(Xf)
    metrics={'target':'idx_qualidade_vida','n_samples':int(len(data)),'n_features':int(len(feature_cols)),'mae_train':float(mean_absolute_error(y,pred)),'r2_train':float(r2_score(y,pred)) if len(data)>1 else None,
             'alerta':'Com apenas um município, o modelo mede evolução temporal interna; para generalizar, inclua todos os municípios.'}
    joblib.dump({'pipeline':reg,'feature_cols':feature_cols,'target':'idx_qualidade_vida'}, out/'modelo_rf_qualidade_vida.joblib')
    clf=Pipeline([('imputer',SimpleImputer(strategy='median')),('model',RandomForestClassifier(n_estimators=400, random_state=42, class_weight='balanced'))])
    clf.fit(Xf,data['classe_maturidade'])
    joblib.dump({'pipeline':clf,'feature_cols':feature_cols,'target':'classe_maturidade'}, out/'modelo_rf_classe_maturidade.joblib')
    if 'idhm' in data.columns and data['idhm'].notna().sum()>=5:
        d2=data[data['idhm'].notna()].copy()
        idhm=Pipeline([('imputer',SimpleImputer(strategy='median')),('model',RandomForestRegressor(n_estimators=500, random_state=42))])
        idhm.fit(d2[feature_cols], d2['idhm'])
        joblib.dump({'pipeline':idhm,'feature_cols':feature_cols,'target':'idhm'}, out/'modelo_rf_idhm.joblib')
        metrics['idhm_model']={'n_samples':int(len(d2)),'r2_train':float(r2_score(d2['idhm'],idhm.predict(d2[feature_cols]))) if len(d2)>1 else None}
    imp=pd.DataFrame({'feature':feature_cols,'importance':reg.named_steps['model'].feature_importances_}).sort_values('importance',ascending=False)
    imp.to_csv(out/'features_importantes_modelo.csv',index=False,encoding='utf-8-sig')
    with open(out/'metricas_modelo.json','w',encoding='utf-8') as f: json.dump(metrics,f,ensure_ascii=False,indent=2)
    return data,imp,metrics

def plots(idx,imp,names,out,top_n=20):
    clean_cols=['localidade_id','ano','idx_saude','idx_educacao','idx_seguranca','idx_governanca','idx_tecnologia','idx_gestao','idx_participacao','idx_meio_ambiente','idx_qualidade_vida','classe_maturidade']
    idx[clean_cols].to_csv(out/'indices_municipio_ano.csv',index=False,encoding='utf-8-sig')
    ncols=[c for c in idx.columns if c.startswith('n_')]
    idx[['localidade_id','ano']+ncols].to_csv(out/'cobertura_indicadores_por_tema.csv',index=False,encoding='utf-8-sig')
    index_cols=[c for c in clean_cols if c.startswith('idx_')]
    avg=idx.groupby('ano')[index_cols].mean(numeric_only=True).reset_index()
    for col in index_cols:
        plt.figure(figsize=(10,5)); plt.plot(avg['ano'], avg[col], marker='o')
        plt.ylim(0,1); plt.title(f'Evolução média - {col}'); plt.xlabel('Ano'); plt.ylabel('Índice 0-1'); plt.grid(True,alpha=.3); plt.tight_layout()
        plt.savefig(out/f'evolucao_media_{col}.png',dpi=160); plt.close()
    for mun in idx['localidade_id'].astype(str).drop_duplicates().head(top_n):
        d=idx[idx['localidade_id'].astype(str)==mun].sort_values('ano')
        plt.figure(figsize=(11,6))
        for col in ['idx_qualidade_vida','idx_saude','idx_educacao','idx_seguranca','idx_governanca','idx_tecnologia']:
            plt.plot(d['ano'], d[col], marker='o', label=col.replace('idx_',''))
        plt.ylim(0,1); plt.title(f'Evolução dos índices - município {mun}'); plt.xlabel('Ano'); plt.ylabel('Índice 0-1'); plt.legend(); plt.grid(True,alpha=.3); plt.tight_layout()
        plt.savefig(out/f'evolucao_municipio_{mun}.png',dpi=160); plt.close()
    ano_max=idx['ano'].max(); idx[idx['ano']==ano_max].sort_values('idx_qualidade_vida',ascending=False).to_csv(out/'ranking_ultimo_ano.csv',index=False,encoding='utf-8-sig')
    rep=imp.head(80).copy()
    def label(f):
        base=f.replace('num_','')
        return names.get(base, f)
    rep['indicador_nome']=rep['feature'].map(label)
    rep.to_csv(out/'top_features_interpretadas.csv',index=False,encoding='utf-8-sig')
    top=rep.head(20).iloc[::-1]
    plt.figure(figsize=(12,8)); plt.barh([str(x)[:80] for x in top['indicador_nome']], top['importance'])
    plt.title('Top 20 features associadas ao índice de qualidade de vida'); plt.xlabel('Importância no Random Forest'); plt.tight_layout()
    plt.savefig(out/'top20_features_qualidade_vida.png',dpi=160); plt.close()

def report(out,metrics,tf,idx):
    lines=['# Relatório metodológico - MUNIC ML v2\n\n',
           '## Correção aplicada\nA versão v2 evita a normalização min-max que deixava todos os índices em 0,5 quando há apenas um município. Os índices agora usam a proporção de respostas positivas disponíveis em cada tema.\n\n',
           '## Interpretação\nO índice varia de 0 a 1. Valores maiores indicam maior presença de estruturas, políticas, sistemas, conselhos, planos ou instrumentos associados ao tema. Não é IDHM oficial.\n\n',
           '## Métricas\n```json\n'+json.dumps(metrics,ensure_ascii=False,indent=2)+'\n```\n\n',
           '## Indicadores por tema encontrados por palavras-chave\n']
    for k,v in tf.items(): lines.append(f'- {k}: {len(v)} indicadores candidatos\n')
    lines.append('\n## Variação observada nos índices\n')
    for c in [x for x in idx.columns if x.startswith('idx_')]:
        lines.append(f'- {c}: min={idx[c].min():.3f}, max={idx[c].max():.3f}, média={idx[c].mean():.3f}\n')
    lines.append('\n## Limitação\nSeu SQLite contém, pelo diagnóstico, apenas a localidade 330330 com várias rodadas anuais. Para ranking nacional e modelo generalizável, é necessário baixar/gravar todos os municípios em resultados.\n')
    (out/'RELATORIO_METODOLOGICO.md').write_text(''.join(lines),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--db',required=True)
    ap.add_argument('--out',default='saida_munic_ml_v2')
    ap.add_argument('--idh_csv',default=None)
    ap.add_argument('--top_n',type=int,default=20)
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    ind,res=load(args.db)
    X,df=build_matrix(ind,res)
    names=feature_names(ind)
    X.to_csv(out/'matriz_features_municipio_ano.csv',index=False,encoding='utf-8-sig')
    idx,tf=compute_indices(X,names,out)
    data,imp,metrics=train(X,idx,out,names,args.idh_csv)
    plots(idx,imp,names,out,args.top_n)
    report(out,metrics,tf,idx)
    print('[OK] MUNIC ML v2 concluído')
    print('Saída:', out.resolve())
    print('Verifique indices_municipio_ano.csv e diagnostico_indicadores_temas.csv')
if __name__=='__main__': main()
