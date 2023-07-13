from fire import Fire
import os
from os.path import join
import json
import sentence_transformers
import numpy as np
from zerorank.index.util import format_dataset, encode_querydoc, construct_id_lookup

def main(corpus : str, 
         output : str, 
         cutoff : int = 10000, 
         model_name : str = 'bert-base-uncased',
         batch_size : int = 32):
     
    os.makedirs(output, exist_ok=True)
    df = format_dataset(corpus, cutoff=cutoff//2)
    model = sentence_transformers.SentenceTransformer(model_name)

    df = encode_querydoc(df, model, batch_size=batch_size)
    df['embedding'] = df.apply(lambda x : np.concatenate(x['q_embedding'], x['d_embedding']), axis=1)
    lookup = construct_id_lookup(df)

    embedding = df['embedding'].to_numpy()
    np.save(join(output, 'embeddings.npy'), embedding)
    json.dump(lookup, open(join(output, 'lookup.json'), 'w'))

if __name__ == '__main__':
    Fire(main)