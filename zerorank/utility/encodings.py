from fire import Fire
import os
from os.path import join
import json
import sentence_transformers
import numpy as np
import ir_datasets as irds
from zerorank.index.util import construct_query_lookup
import pandas as pd
import gc
import torch

def main(data : str,
         corpus : str, 
         output : str, 
         model_name : str = 'bert-base-uncased',
         batch_size : int = 32):
    
    ds = irds.load(corpus)
    queries = pd.DataFrame(ds.queries_iter()).set_index('query_id')['query'].to_dict()
    docs = pd.DataFrame(ds.docs_iter()).set_index('docno')['text'].to_dict()

    frame = pd.read_csv(data, sep='\t', index=False, header=None, names=['query_id', 'doc_id_a', 'doc_id_b'])
    frame['query'] = frame['qid'].apply(lambda x : queries[x])
    frame['text_a'] = frame['doc_id_a'].apply(lambda x : docs[x])

    frame.reset_index(drop=True, inplace=True)

    del queries
    del docs
    gc.collect()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = sentence_transformers.SentenceTransformer(model_name, device=device)

    q_embeddings = model(frame['query'].tolist(), batch_size=batch_size, convert_to_numpy=True)  
    np.save(join(output, 'q_embeddings.npy'), q_embeddings)
    del q_embeddings
    a_embeddings = model(frame['text_a'].tolist(), batch_size=batch_size, convert_to_numpy=True)
    np.save(join(output, 'd_embeddings.npy'), a_embeddings)
    del a_embeddings

    lookup = construct_query_lookup(frame)
    json.dump(lookup, open(join(output, 'lookup.json'), 'w'))


if __name__ == '__main__':
    Fire(main)