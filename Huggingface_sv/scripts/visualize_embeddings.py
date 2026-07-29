import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap

def load_data():
    index_path = "cbir_index.npz"
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"{index_path} not found. Ensure the index is built.")
    data = np.load(index_path, allow_pickle=True)
    return data['embeddings'], data['labels']

def plot_embeddings(embeddings, labels, method='tsne'):
    print(f"Running {method.upper()} on {len(embeddings)} embeddings...")
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    else:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        
    projections = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    
    # Map labels to colors
    unique_labels = list(set(labels))
    palette = sns.color_palette("husl", len(unique_labels))
    
    for idx, label in enumerate(unique_labels):
        mask = (labels == label)
        plt.scatter(
            projections[mask, 0], 
            projections[mask, 1], 
            color=palette[idx], 
            label=label.capitalize(),
            alpha=0.7,
            s=40,
            edgecolors='w',
            linewidth=0.5
        )
        
    plt.title(f"{method.upper()} Projection of MobileNetV2 Embeddings (CBIR Index)")
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    plt.legend(title="Class")
    plt.tight_layout()
    
    os.makedirs("../results", exist_ok=True)
    out_path = f"../results/{method}_plot.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

def main():
    try:
        embeddings, labels = load_data()
    except Exception as e:
        print(e)
        return
        
    plot_embeddings(embeddings, labels, 'tsne')
    plot_embeddings(embeddings, labels, 'umap')

if __name__ == "__main__":
    main()
