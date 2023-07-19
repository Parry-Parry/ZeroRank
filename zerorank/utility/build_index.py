import faiss 
from fire import Fire
from os.path import join
import numpy as np
import json

def main(artifact_dir : str, query_only : bool = False, nprobe : int = 10, nlist : int = 100, niter : int = 5, n : int = 10, quantized_dim : int = 256):
    queries = np.load(join(artifact_dir, 'q_embeddings.npy'))
    if not query_only:
        docs = np.load(join(artifact_dir, 'd_embeddings.npy'))
        embeddings = np.concatenate([queries, docs], axis=1)
    else:
        embeddings = queries

    lookup = np.load(join(artifact_dir, 'idx.npy'))

    coarse_quantizer = faiss.IndexFlatL2 (quantized_dim)
    sub_index = faiss.IndexIVFPQ (coarse_quantizer, quantized_dim, ncoarse, 16, 8)
    


if __name__ == "__main__":
    Fire(main)