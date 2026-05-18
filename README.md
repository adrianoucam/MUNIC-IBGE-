# MUNIC ML Qualidade de Vida v2

Esta versão corrige o problema dos índices todos em 0,5.

## Como executar
```bash
pip install pandas numpy matplotlib scikit-learn joblib
python munic_ml_qualidade_vida_v2.py --db sqlite_munic.sqlite --out saida_munic_ml_v2
```

## Saídas
- `indices_municipio_ano.csv`: índices corrigidos por ano.
- `cobertura_indicadores_por_tema.csv`: quantos indicadores entraram em cada tema por ano.
- `diagnostico_indicadores_temas.csv`: diagnóstico dos temas.
- `modelo_rf_qualidade_vida.joblib`: modelo Random Forest salvo.
- `top_features_interpretadas.csv`: variáveis mais importantes com nomes dos indicadores.

## Observação
O banco anexado parece conter apenas uma localidade municipal (`330330`) ao longo dos anos. Assim, o modelo aprende evolução temporal, não comparação nacional. Para ranking nacional, é preciso popular o SQLite com todos os municípios.
