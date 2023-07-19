from fire import Fire
import ir_datasets as irds
import pandas as pd
from os.path import join

def main(dataset : str = 'msmarco-passage',
         output_dir : str = '/',
         cutoff : int = 1e6):
    ds = irds.load(dataset)
    docpairs = pd.DataFrame(ds.docpairs_iter())

    subset = docpairs.sample(n=cutoff).reset_index(drop=True)
    subset.to_csv(join(output_dir, f'subset{cutoff}.csv'), index=False, header=False)

if __name__ == '__main__':
    Fire(main)