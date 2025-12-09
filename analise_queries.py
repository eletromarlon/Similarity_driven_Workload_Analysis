#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
queries_pipeline.py
===================
Pipeline de análise de carga de trabalho SQL:
1. Pré-processamento e normalização das queries
2. Geração de embeddings semânticos
3. Agrupamento com DBSCAN
4. Construção de séries temporais
5. Extração de features com catch22

Autor: Marlon Duarte
Data: 2025-11-09
"""

import re
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
# Use non-interactive backend for headless environments (servers/CI)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from pycatch22 import catch22_all


# ======================================================
# === Etapa 1 — Pré-processamento das queries SQL ======
# ======================================================

def normalize_query(query: str) -> str:
    """Normaliza consultas SQL: lowercase, remove valores específicos e uniformiza espaços."""
    if not isinstance(query, str):
        return ""
    #q = query.lower()
    #q = re.sub(r"'[^']*'", "?", q)        # substitui strings
    #q = re.sub(r"\b\d+\b", "?", q)        # substitui números
    #q = re.sub(r"\b[0-9a-fA-F-]{36}\b", "?", q)  # UUIDs
    #q = re.sub(r"%s", "?", q)             # placeholders SQL
    #q = re.sub(r"\s+", " ", q).strip()    # espaços extras
    return query


def preprocess_queries(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica normalização nas queries e garante colunas esperadas."""
    df = df.copy()
    df["query_norm"] = df["query_text"].apply(normalize_query)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    return df


# ======================================================
# === Etapa 2 — Embeddings das queries ================
# ======================================================

def generate_embeddings(queries, model_name="all-MiniLM-L6-v2", save_path="embeddings.npy"):
    """Gera embeddings vetoriais para as queries normalizadas."""
    print(f"[{datetime.now()}] Gerando embeddings com modelo '{model_name}' ...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(queries, show_progress_bar=True)
    np.save(save_path, embeddings)
    print(f"[✓] Embeddings salvos em {save_path}")
    return embeddings


# ======================================================
# === Etapa 3 — Clustering DBSCAN =====================
# ======================================================

def cluster_queries(embeddings, eps=0.4, min_samples=2):
    """Agrupa queries semanticamente equivalentes usando DBSCAN."""
    print(f"[{datetime.now()}] Executando clustering DBSCAN ...")
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = db.fit_predict(embeddings)
    print(f"[✓] Clusters encontrados: {len(set(labels)) - (1 if -1 in labels else 0)} (+ outliers)")
    return labels


# ======================================================
# === Etapa 4 — Séries temporais ======================
# ======================================================

def build_time_series(df: pd.DataFrame):
    """Gera séries temporais e um gráfico de eventos (timestamp vs cluster_id).

    - Recebe apenas o DataFrame `df` com pelo menos as colunas `timestamp`,
      `cluster_id` e `duration`.
    - Cria dois CSVs agregados com frequência de 1 minuto: `series_count.csv`
      (contagem de queries por cluster) e `series_duration.csv` (duração média).
    - Gera um gráfico de eventos (`series_event_plot.png`) onde o eixo x é o
      timestamp real de cada query e o eixo y é o `cluster_id`. A cor indica
      o cluster e o tamanho do ponto é proporcional à duração.

    Retorna: (ts_count, ts_duration)
    """
    print(f"[{datetime.now()}] Construindo séries temporais e gráfico de eventos ...")

    # Garantias e conversões
    df = df.copy()
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame deve conter a coluna 'timestamp'.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "cluster_id" not in df.columns:
        raise ValueError("DataFrame deve conter a coluna 'cluster_id'.")
    if "duration" not in df.columns:
        # se não houver duration, criar coluna com 1.0 para permitir plot
        df["duration"] = 1.0

    df.sort_values("timestamp", inplace=True)

    # Agregações em janela de 1 minuto (mantemos estes CSVs para compatibilidade)
    ts_count = df.groupby([pd.Grouper(key="timestamp", freq="1Min"), "cluster_id"]).size().unstack(fill_value=0)
    ts_duration = df.groupby([pd.Grouper(key="timestamp", freq="1Min"), "cluster_id"])['duration'].mean().unstack(fill_value=0)

    ts_count.to_csv("series_count.csv")
    ts_duration.to_csv("series_duration.csv")

    # ----- Gráfico de eventos: timestamp (x) vs cluster_id (y) -----
    # Mapear cluster_id para índices (fatorizar) para cores consistentes
    cluster_labels, inv = pd.factorize(df['cluster_id'])
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % cmap.N) for i in cluster_labels]

    # Tamanho proporcional à duração (normalizada)
    durations = df['duration'].astype(float).fillna(0).values
    if durations.max() > 0:
        sizes = 20 + 80 * (durations / durations.max())
    else:
        sizes = np.full_like(durations, 30, dtype=float)

    plt.figure(figsize=(14, 6))
    plt.scatter(df['timestamp'], df['cluster_id'], c=colors, s=sizes, alpha=0.7, edgecolors='w', linewidth=0.2)
    plt.xlabel('Timestamp')
    plt.ylabel('Cluster ID')
    plt.title('Event plot: queries (timestamp x) vs cluster_id (y)')
    plt.grid(axis='x', linestyle='--', alpha=0.4)

    # Ajustar y-ticks para mostrar todos os cluster_id únicos, na ordem natural
    unique_clusters = np.unique(df['cluster_id'])
    plt.yticks(unique_clusters)

    # Legenda simples: mostrar número de eventos por cluster
    counts_by_cluster = df['cluster_id'].value_counts().sort_index()
    # construir texto da legenda
    legend_text = '\n'.join([f"Cluster {cid}: {int(counts_by_cluster.get(cid, 0))} events" for cid in unique_clusters])
    plt.gcf().text(0.99, 0.5, legend_text, fontsize=9, va='center', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    out_png = 'series_event_plot.png'
    plt.tight_layout(rect=(0, 0, 0.97, 1.0))
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"[✓] Séries salvas: series_count.csv e series_duration.csv")
    print(f"[✓] Gráfico de eventos salvo em {out_png}")
    return ts_count, ts_duration


# ======================================================
# === Etapa 5 — Extração de features com catch22 =======
# ======================================================

def extract_catch22_features(ts_df: pd.DataFrame, prefix="count"):
    """Extrai 22 features descritivas das séries temporais por cluster (versão robusta)."""
    print(f"[{datetime.now()}] Extraindo features com catch22 (robusto) ...")
    features = []
    
    for cluster_id in ts_df.columns:
        serie = ts_df[cluster_id].values.astype(float)
        serie = serie[~np.isnan(serie)]
        if len(serie) < 10 or np.all(serie == serie[0]):
            print(f"[!] Série do cluster {cluster_id} ignorada (curta ou constante)")
            continue
        serie = (serie - np.mean(serie)) / (np.std(serie) + 1e-8)
        feat_dict = catch22_all(serie)
        feat_dict["cluster_id"] = cluster_id
        features.append(feat_dict)
    
    feat_df = pd.DataFrame(features)
    feat_df.to_csv(f"catch22_features_{prefix}.csv", index=False)
    print(f"[✓] Features salvas em catch22_features_{prefix}.csv")
    return feat_df


# ======================================================
# === Função principal ================================
# ======================================================

def main():
    input_path = "dataset_csv_renamed/dbp4_postgresql-2016-11-29.csv"
    if not os.path.exists(input_path):
        print(f"[ERRO] Arquivo {input_path} não encontrado.")
        return
    
    print(f"\n[{datetime.now()}] Iniciando pipeline de análise de queries SQL ...\n")
    
    # carregando o csv que não possui cabeçalho
    df = pd.read_csv(input_path, header=None)
    #df = preprocess_queries(df)
     
    print(df.head())

    # Tratar os dados da coluna 3. Considerar os dados apenas após o ":"
    df['query_text'] = df[3].apply(lambda x: x.split(":", 1)[1].strip() if isinstance(x, str) and ":" in x else "")

    # Imprime na tela valores da coluna 'query_text' que possuem ":" no conteúdo
    if df['query_text'] == ":":
        print("Consultas com ':' no conteúdo:")
        print(df['query_text'][df['query_text'] == ":"])

    print(df.head())

    # 2. Embeddings
    '''embeddings = generate_embeddings(df["query_norm"].tolist())
    
    # 3. Clustering
    labels = cluster_queries(embeddings)
    df["cluster_id"] = labels
    df.to_csv("queries_clustered.csv", index=False)
    print(f"[✓] Resultado salvo em queries_clustered.csv")
    
    # 4. Séries temporais
    ts_count, ts_duration = build_time_series(df)
    
    # 5. Extração de features (exemplo: contagem)
    #extract_catch22_features(ts_count, prefix="count")
    
    print(f"\n[{datetime.now()}] Pipeline concluído com sucesso! ✅\n")

'''
# ======================================================
# === Execução direta =================================
# ======================================================

if __name__ == "__main__":
    main()
