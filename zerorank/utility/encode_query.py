from fire import Fire
import os
from os.path import join
import json
import sentence_transformers
import numpy as np
from zerorank.index.util import format_dataset, construct_query_lookup

def main(corpus : str, 
         output : str, 
         cutoff : int = 10000, 
         model_name : str = 'bert-base-uncased',
         batch_size : int = 32):
     
    os.makedirs(output, exist_ok=True)
    
    df = format_dataset(corpus, cutoff=cutoff//2, keep_triples=True)
    df = df.drop_duplicates(subset=['qid']).reset_index(drop=True)
    lookup = construct_query_lookup(df)
    model = sentence_transformers.SentenceTransformer(model_name)
    df['embedding'] = model(df['query'].tolist(), batch_size=batch_size, convert_to_numpy=True)

    embedding = df['embedding'].to_numpy()
    np.save(join(output, 'embeddings.npy'), embedding)
    json.dump(lookup, open(join(output, 'lookup.json'), 'w'))

if __name__ == '__main__':
    Fire(main)