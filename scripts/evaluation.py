import pyterrier as pt
pt.init()

from fire import Fire
import logging
import torch
from zerorank.reranker.model import FLANT5
from zerorank.reranker.reranker import ZeroRanker
from sentence_transformers import SentenceTransformer
from trec23.evaluation import generate_experiment
from trec23.pipelines.baselines import load_pisa

import faiss

def main(model_name_or_path : str,
         embedding_model_name_or_path : str, 
         pisa_path : str,
         index_path : str, 
         eval_ds : str, 
         output : str,
         batch_size : int,
         query_only : bool = False,
         k : int = 3,
         nprobe : int = 10, 
         ):
    
    bm25 = load_pisa(dataset=pisa_path).bm25()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = SentenceTransformer(embedding_model_name_or_path, device=device)
    model = FLANT5(model_name_or_path, device=device)
    index = faiss.read_index(index_path)
    reranker = ZeroRanker(model, embedding_model, index, query_only=query_only, k=k, nprobe=nprobe, batch_size=batch_size)

    dataset = pt.get_dataset('irds:msmarco-passage/train/triples-small')
    pipe = bm25 >> pt.get_text(dataset, "text") >> reranker

    ds = pt.get_dataset(eval_ds)

    res = generate_experiment(bm25, pipe, dataset=ds, names = ["BM25", "BM25 >> ZeroRank"], baseline=0)
    res.to_csv(output, sep='\t', index=False, header=False)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Fire(main)