import numpy as np
import faiss
from fire import Fire

class faissMap():
    def __init__(self, index, idxtotxt) -> None:
        self.index = index
        self.idxtotxt = idxtotxt

    def add(self, items, idx):
        self.index.add_with_ids(items, idx)

    def search(self, items, n=10, nprobe=10):
        D, I = self.index.search(items, n, nprobe)
        return [[self.idxtotxt[i] for i in l] for l in I]


def main(out : str, sentence_encoder : str, dataset : str, cut : int, dim : int, nprobe : int, nlist : int, niter : int, n : int):
    import zerorank.index.util as util
    import pandas as pd
    pass 

if __name__ == "__main__":
    Fire(main)