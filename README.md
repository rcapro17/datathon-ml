# Datathon – Passos Mágicos (Machine Learning Engineering)

Projeto desenvolvido para o **Datathon – Case Passos Mágicos**, com o objetivo de apoiar a missão de transformação social por meio da educação.

A Associação Passos Mágicos atua há décadas promovendo educação de qualidade para crianças e jovens em vulnerabilidade social. Neste projeto, construímos um ciclo completo de **Machine Learning Engineering / MLOps**, desde o treinamento do modelo até a disponibilização em produção via API, com empacotamento Docker, testes automatizados, logging e monitoramento de drift.

---

## 🎯 Objetivo do Projeto (Problema de Negócio)

Construir um **modelo preditivo** capaz de **estimar o risco de defasagem escolar** por estudante, para permitir:

- identificação precoce de alunos com risco de defasagem;
- priorização de intervenções pedagógicas;
- acompanhamento contínuo das mudanças nos dados (drift).

---

## ✅ Solução Proposta

Implementamos uma pipeline completa de Machine Learning:

1. **Ingestão de dados** (`src/data_load.py`)
2. **Pré-processamento / normalização** (`src/predict.py`)
3. **Treinamento e avaliação do modelo** (`src/train.py`)
4. **Serialização do modelo** com `joblib` (modelo salvo em `app/model/model.joblib`)
5. **Deploy via API Flask** com endpoints:
   - `GET /health`
   - `POST /predict`
6. **Empacotamento Docker** para execução replicável
7. **Testes unitários + cobertura** com `pytest + pytest-cov`
8. **Monitoramento contínuo**:
   - logging de requests do `/predict` em `monitoring/logs/requests.parquet`
   - relatório HTML com Evidently: `monitoring/reports/drift_report.html`

---

## 🧰 Stack Tecnológica

- **Linguagem**: Python 3.x
- **Data/ML**: pandas, numpy, scikit-learn
- **Serialização**: joblib
- **API**: Flask + gunicorn
- **Testes**: pytest, pytest-cov
- **Containerização**: Docker
- **Monitoramento**: logging + Evidently (Data Drift)

---

## 📦 Estrutura do Projeto

```bash
datathon-ml/
  app/
    main.py
    routes.py
    schemas.py
    model/
      model.joblib
  src/
    config.py
    data_load.py
    processing.py
    features.py
    train.py
    predict.py
  monitoring/
    drift_report.py
    seed_requests.py
    logs/
    reports/
  tests/
    test_preprocessing.py
    test_predict.py
    test_api.py
  data/
    raw/
    processed/
  artifacts/
    metrics.json
    schema.json
  requirements.txt
  Dockerfile
  README.md
  .gitignore
```
