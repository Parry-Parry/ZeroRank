from pyterrier import transformer
import numpy as np

class ZeroRanker(transformer):
    def __init__(self, 
                 model, 
                 embedding_model,
                 memory_structure, 
                 prompt,
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
        self.batch_size = batch_size
    
        self.text_attr = text_attr
        self.k = k
        self.nprobe = nprobe

    def construct_prompt(self, query, document, context):
        return self.prompt.construct(query=query, document=document, context=self.prompt.examples(context))

    def transform(self, inputs):

        inputs = inputs.copy()
        intermediate = inputs.copy()
        
        query_lookup = inputs[['qid', 'query']].drop_duplicates()
        query_lookup['embedding'] = self.embedding_model(query_lookup['query'].tolist(), convert_to_numpy=True)
        query_lookup = query_lookup.set_index('qid')['embedding'].to_dict()

        doc_lookup = inputs[['docno', self.text_attr]].drop_duplicates()
        doc_lookup['embedding'] = self.embedding_model(doc_lookup[self.text_attr].tolist(), convert_to_numpy=True)
        doc_lookup = doc_lookup.set_index('docno')['embedding'].to_dict()

        index_queries = inputs.apply(lambda x : np.concatenate(query_lookup[x['qid']], doc_lookup[x['docno']]), axis=1)
        intermediate['context'] = self.memory_structure.search(index_queries.to_numpy(), self.k, self.nprobe)

        intermediate['prompt'] = intermediate.apply(lambda x : self.construct_prompt(x['query'], x[self.text_attr], x['context']), axis=1)
        
        inputs['score'] = intermediate['prompt'].apply(lambda x : self.model(x, batch_size=self.batch_size))

        return inputs
