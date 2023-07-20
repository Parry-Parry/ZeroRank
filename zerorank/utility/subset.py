from fire import Fire
import ir_datasets as irds
import pandas as pd
from os.path import join

def main(dataset : str = '/data/triples.tsv',
         output_dir : str = '/',
         cutoff : int = 1000000):
    docpairs = pd.read_csv(dataset, sep='\t', header=None, names=['query_id', 'doc_id_a', 'doc_id_b'], dtype={'query_id' : str, 'doc_id_a' : str, 'doc_id_b' : str})
    docpairs = docpairs.drop_duplicates(subset=['query_id'])
    subset = docpairs.sample(n=cutoff).reset_index(drop=True)
    subset.to_csv(join(output_dir, f'subset{cutoff}.tsv'), sep='\t', index=False, header=False)

if __name__ == '__main__':
    Fire(main)