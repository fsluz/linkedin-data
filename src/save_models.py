"""
save_models.py — Cole este código no seu Colab para salvar os modelos
======================================================================
Após rodar seus pipelines no Colab, execute este bloco para salvar
os modelos e baixá-los para colocar na pasta models/ do projeto.

Instruções:
  1. Rode seus pipelines normalmente no Colab
  2. Cole e execute este script no Colab
  3. Baixe os arquivos .pkl gerados
  4. Coloque-os em:  projeto-ml-tcc/models/
  5. Reinicie o servidor (ele carrega os reais automaticamente)
"""

import os

# ============================================================
# 1. Classificação
# ============================================================
# Substitua `best_clf` pelo nome da variável do seu melhor modelo

# from pycaret.classification import save_model
# save_model(best_clf, 'models/classification_pipeline')
# print("✅ Classificação salva em models/classification_pipeline.pkl")


# ============================================================
# 2. Regressão
# ============================================================
# Substitua `best_reg` pelo nome da variável do seu melhor modelo

# from pycaret.regression import save_model
# save_model(best_reg, 'models/regression_pipeline')
# print("✅ Regressão salva em models/regression_pipeline.pkl")


# ============================================================
# 3. Clusterização
# ============================================================
# Substitua `best_clust` pelo nome da variável do seu melhor modelo

# from pycaret.clustering import save_model
# save_model(best_clust, 'models/clustering_pipeline')
# print("✅ Clusterização salva em models/clustering_pipeline.pkl")


# ============================================================
# 4. Recomendação (cosine similarity / SVD)
# ============================================================
# Se o seu modelo de recomendação não for PyCaret nativo,
# salve com joblib:

# import joblib
# joblib.dump(rec_model, 'models/recommendation_pipeline.pkl')
# print("✅ Recomendação salva em models/recommendation_pipeline.pkl")


# ============================================================
# 5. Atualizar métricas reais no servidor
# ============================================================
# Após treinar, atualize o dict `self.metrics` em ml_server.py
# com os valores reais, por exemplo:
#
# model_provider.metrics["classification"] = {
#     "accuracy": float(pull()["Accuracy"].iloc[-1]),
#     "precision": float(pull()["Prec."].iloc[-1]),
#     "recall": float(pull()["Recall"].iloc[-1]),
#     "f1_score": float(pull()["F1"].iloc[-1]),
# }


# ============================================================
# Download (rode no Colab para baixar os arquivos)
# ============================================================
# from google.colab import files
# import glob
# for f in glob.glob('models/*.pkl'):
#     files.download(f)
#     print(f"⬇️  {f}")
