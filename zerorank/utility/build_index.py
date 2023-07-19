import faiss 
from fire import Fire
from os.path import join
import numpy as np

def main(artifact_dir : str, output : str, query_only : bool = False, quantized_dim : int = 256, ncoarse : int = 100):
    queries = np.load(join(artifact_dir, 'q_embeddings.npy'))
    if not query_only:
        docs = np.load(join(artifact_dir, 'd_embeddings.npy'))
        embeddings = np.concatenate([queries, docs], axis=1)
    else:
        embeddings = queries

    print(embeddings.shape)

    idx = np.load(join(artifact_dir, 'idx.npy'))

    pca_matrix = faiss.PCAMatrix(embeddings.shape[-1], quantized_dim, 0, False) 
    coarse_quantizer = faiss.IndexFlatL2(quantized_dim)
    sub_index = faiss.IndexIVFPQ(coarse_quantizer, quantized_dim, ncoarse, 16, 8)

    index = faiss.IndexPreTransform(pca_matrix, sub_index)
    index.train(embeddings)
    index.add_with_ids(embeddings, idx)

    identifier = 'q' if query_only else 'qd'
    faiss.write_index(index, join(output, f'index_{identifier}_{quantized_dim}_{ncoarse}.index'))

if __name__ == "__main__":
    Fire(main)