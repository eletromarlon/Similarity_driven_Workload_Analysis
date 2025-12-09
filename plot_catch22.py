import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # Necessário para tratar Infinity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_feature_map(df_features):
    print("-> Gerando Mapa de Comportamento (PCA)...")
    
    # 1. Separar apenas as colunas numéricas (features do catch22)
    # Removemos metadados como cluster_id, example_query, count
    feature_cols = df_features.drop(columns=['cluster_id', 'example_query', 'count'], errors='ignore')
    
    # --- LIMPEZA DE DADOS (NOVO) ---
    # O catch22 pode gerar NaNs ou Infinitos em séries muito curtas ou constantes.
    # O PCA não aceita isso. Vamos substituir por 0.
    feature_cols = feature_cols.fillna(0)
    feature_cols = feature_cols.replace([np.inf, -np.inf], 0)
    # -------------------------------

    # 2. Normalizar os dados (O catch22 gera escalas muito diferentes)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_cols)
    
    # 3. Reduzir para 2 Dimensões com PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    
    # 4. Adicionar ao DF para plotagem
    df_features['PCA1'] = components[:, 0]
    df_features['PCA2'] = components[:, 1]
    
    # 5. Plotar
    plt.figure(figsize=(12, 8))
    
    # Usamos o tamanho da bolinha (s) para representar a frequência (count)
    # Usamos log no count para bolinhas gigantes não taparem as pequenas
    sizes = np.log1p(df_features['count']) * 20 
    
    scatter = plt.scatter(
        df_features['PCA1'], 
        df_features['PCA2'], 
        s=sizes, 
        alpha=0.6, 
        c=df_features['cluster_id'], 
        cmap='viridis'
    )
    
    plt.title('Mapa de Comportamento das Queries (Baseado no Catch22)', fontsize=14)
    plt.xlabel('Componente Principal 1 (Variação de Padrão)')
    plt.ylabel('Componente Principal 2')
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True, alpha=0.3)
    
    # Opcional: Anotar os IDs dos maiores clusters
    top_clusters = df_features.nlargest(5, 'count')
    for idx, row in top_clusters.iterrows():
        plt.text(row['PCA1'], row['PCA2'], f"ID {row['cluster_id']}", fontsize=9, weight='bold')

    plt.show()


def plot_top_clusters_series(df_original, df_features, top_n=3):
    print(f"-> Plotando séries temporais dos Top {top_n} clusters...")
    
    # Pegar os IDs dos clusters mais frequentes
    top_ids = df_features.nlargest(top_n, 'count')['cluster_id'].values
    
    plt.figure(figsize=(15, 4 * top_n))
    
    for i, c_id in enumerate(top_ids):
        # Filtrar dados brutos desse cluster
        cluster_data = df_original[df_original['cluster_id'] == c_id].copy()
        cluster_data = cluster_data.sort_values('timestamp')
        
        # Recalcular delta_t para garantir
        cluster_data['delta_t'] = cluster_data['timestamp'].diff().dt.total_seconds()
        
        # Subplot
        plt.subplot(top_n, 1, i+1)
        
        # Plotar: X = Hora, Y = Tempo desde a última query (Delta T)
        plt.plot(cluster_data['timestamp'], cluster_data['delta_t'], 
                 marker='.', linestyle='-', linewidth=0.5, markersize=4, alpha=0.7)
        
        # Título com a query (cortada para não poluir)
        query_sample = df_features[df_features['cluster_id'] == c_id]['example_query'].values[0]
        plt.title(f"Cluster {c_id} (N={len(cluster_data)})\nQuery: {query_sample}", fontsize=10)
        plt.ylabel("Intervalo (segundos)")
        plt.grid(True, alpha=0.3)
        
        # Se houver outliers muito grandes no eixo Y, usar escala log ajuda a ver melhor
        # plt.yscale('log') 

    plt.tight_layout()
    plt.show()


def plot_specific_clusters(df_original, df_features, cluster_ids):
    """
    Plota as séries temporais apenas para os IDs de cluster especificados na lista.
    Exemplo de uso: plot_specific_clusters(df, df_features, [0, 5, 12])
    """
    print(f"-> Plotando clusters específicos: {cluster_ids}...")
    
    # Validação: garantir que os IDs existem
    valid_ids = [cid for cid in cluster_ids if cid in df_features['cluster_id'].values]
    
    if not valid_ids:
        print("Nenhum ID válido encontrado nas features.")
        return

    n_plots = len(valid_ids)
    
    # AJUSTE VISUAL: 
    # A altura da figura agora é dinâmica: 4 polegadas de altura para CADA gráfico.
    # Se você pedir 10 gráficos, a imagem terá 40 polegadas de altura.
    plt.figure(figsize=(15, 4 * n_plots))
    
    for i, c_id in enumerate(valid_ids):
        # Filtrar dados brutos desse cluster
        cluster_data = df_original[df_original['cluster_id'] == c_id].copy()
        cluster_data = cluster_data.sort_values('timestamp')
        
        # Recalcular delta_t
        cluster_data['delta_t'] = cluster_data['timestamp'].diff().dt.total_seconds()
        
        # Subplot (n_plots linhas, 1 coluna, índice atual)
        plt.subplot(n_plots, 1, i+1)
        
        # Plotagem
        plt.plot(cluster_data['timestamp'], cluster_data['delta_t'], 
                 marker='.', linestyle='-', linewidth=0.5, markersize=4, alpha=0.7)
        
        # Título Informativo
        # Tenta pegar a query de exemplo, se falhar usa string vazia
        try:
            query_sample = df_features[df_features['cluster_id'] == c_id]['example_query'].values[0]
        except IndexError:
            query_sample = "Query não encontrada nas features"
            
        plt.title(f"Cluster ID: {c_id} (Total Events: {len(cluster_data)})\nQuery: {query_sample}", 
                  fontsize=11, fontweight='bold')
        plt.ylabel("Intervalo (segundos)")
        plt.grid(True, alpha=0.3)
        
        # Dica: Se quiser ver melhor picos baixos vs altos, descomente a linha abaixo
        # plt.yscale('log') 

    plt.tight_layout()
    plt.show()
