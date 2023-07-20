from pyterrier import transformer
import numpy as np
from pyterrier.model import add_ranks

class ZeroRanker(transformer):
    def __init__(self, 
                 model, 
                 embedding_model,
                 memory_structure, 
                 prompt,
                 query_only = False,
                 text_attr = 'text',
                 k=3,
                 nprobe=10,
                 batch_size=32,
                 **kwargs):
        super().__init__(**kwargs)

        self.model = model
        self.embedding_model = embedding_model
        self.memory_structure = memory_structure
        self.prompt = prompt
        self.query_only = query_only
        self.batch_size = batch_size
    
        self.text_attr = text_attr
        self.k = k
        self.nprobe = nprobe

    def construct_prompt(self, query, document, context):
        return self.prompt.construct(query=query, document=document, context=self.prompt.examples(context))
    
    def make_queries(self, inputs):
        query_lookup = inputs[['qid', 'query']].drop_duplicates()
        query_embeddings = self.embedding_model.encode(query_lookup['query'].tolist(), batch_size=self.batch_size, convert_to_numpy=True)
        query_lookup['embedding'] = [v.numpy() for v in query_embeddings]
        query_lookup = query_lookup.set_index('qid')['embedding'].to_dict()

        if not self.query_only:
            doc_lookup = inputs[['docno', self.text_attr]].drop_duplicates()
            doc_embeddings = self.embedding_model.encode(doc_lookup[self.text_attr].tolist(), batch_size=self.batch_size)
            doc_lookup['embedding'] = [v.numpy() for v in doc_embeddings]
            doc_lookup = doc_lookup.set_index('docno')['embedding'].to_dict()

            index_queries = inputs.apply(lambda x : np.concatenate(query_lookup[x['qid']], doc_lookup[x['docno']]), axis=1)
        else:
            index_queries = inputs.apply(lambda x : query_lookup[x['qid']], axis=1)

        return index_queries

    def transform(self, inputs):

        inputs = inputs.copy()
        intermediate = inputs.copy()
        
        index_queries = self.make_queries(inputs)
        intermediate['context'] = self.memory_structure.search(index_queries.to_numpy(), self.k, self.nprobe)
        intermediate['prompt'] = intermediate.apply(lambda x : self.construct_prompt(x['query'], x[self.text_attr], x['context']), axis=1)
        
        scores =  self.model.generate(intermediate['prompt'], batch_size=self.batch_size)
        inputs = inputs.drop(columns=['score', 'rank'], errors='ignore').assign(score=scores)
        inputs = add_ranks(inputs)

        return inputs
