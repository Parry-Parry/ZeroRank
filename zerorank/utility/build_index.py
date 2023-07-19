import faiss 
from fire import Fire
from os.path import join

def main(artifact_dir : str, query_only : bool = False, nprobe : int = 10, nlist : int = 100, niter : int = 5, n : int = 10):
    queries = np.load(join(artifact_dir, 'q_embeddings.npy'))
    

if __name__ == "__main__":
    Fire(main)