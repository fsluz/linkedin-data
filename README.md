# 🎓 UNIGRAM ML Pipeline — TCC

Dashboard + API FastAPI para os 4 pipelines de ML (Classificação, Regressão, Clusterização, Recomendação).

## ⚡ Setup rápido

```bash
# 1. Python 3.9 obrigatório (PyCaret não suporta 3.10+)
python3.9 -m venv venv

# Ativar o ambiente virtual:
source venv/bin/activate          # Linux/macOS
venv\Scripts\Activate.ps1         # Windows (PowerShell)

# 2. Instalar dependências
pip install --upgrade pip
pip install -r requirements.txt

# 3. Rodar o servidor (sempre da raiz do projeto)
uvicorn src.ml_server:app --reload --port 8000

# 4. Abrir o dashboard no navegador
# Abra o arquivo dashboard/ml-dashboard.html diretamente no navegador
```

## 📁 Estrutura

```
linkedin-data/
├── dashboard/
│   ├── ml-dashboard.html   ← Dashboard (abra no navegador)
│   ├── ml-dashboard.js     ← Lógica e gráficos do dashboard
│   └── power bi.html       ← Referência visual Power BI
│
├── models/                 ← Coloque os .pkl do Colab aqui
│   └── LEIA-ME.txt
│
├── notebooks/              ← Coloque o .ipynb do Colab aqui
│
├── src/
│   ├── ml_server.py        ← API FastAPI (rode este)
│   ├── save_models.py      ← Cole no Colab para salvar os modelos
│   └── test_api.py         ← Testa todos os endpoints
│
├── venv/                   ← Ambiente virtual (não subir pro Git)
├── requirements.txt        ← Dependências Python 3.9
└── README.md
```

## 🔗 Endpoints

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/health` | Status da API |
| GET | `/ml/status` | Status de cada pipeline |
| GET | `/ml/classification/metrics` | Métricas de classificação |
| GET | `/ml/regression/metrics` | Métricas de regressão |
| GET | `/ml/clustering/metrics` | Métricas de clusterização |
| GET | `/ml/recommendation/metrics` | Métricas de recomendação |
| GET | `/ml/classification/confusion-matrix` | Matriz de confusão |
| GET | `/ml/clustering/characteristics` | Características dos clusters |
| GET | `/ml/recommendation/top-skills` | Top habilidades recomendadas |
| POST | `/ml/classification/predict` | Predição de classificação |
| POST | `/ml/regression/predict` | Predição de regressão |
| POST | `/ml/recommendation/predict` | Recomendação de vagas |

Docs interativos: **http://127.0.0.1:8000/docs**

## 🤖 Integrar modelos do Colab

1. No Colab, siga as instruções em `src/save_models.py`
2. Baixe os `.pkl` gerados
3. Coloque-os em `models/`
4. Reinicie o servidor — ele detecta e carrega automaticamente
5. O dashboard muda de "mock" para "real" instantaneamente

## 🧪 Testar

```bash
# Rodar da raiz do projeto com o venv ativo
python src/test_api.py

# ou em outra porta:
python src/test_api.py --url http://localhost:8001
```
