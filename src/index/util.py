import pandas as pd
import numpy as np

def construct_id_lookup(inputs):
    lookup = {}
    for row in inputs.itertuples():
        lookup[row.Index] = {'query' : row.query, 'document' : row.text, 'relevance' : row.relevance}
    return lookup

def format_dataset(corpus):
    docpairs = pd.DataFrame(corpus.docpairs_iter())
    docs = pd.DataFrame(corpus.docs_iter()).set_index('doc_id').text.to_dict()
    queries = pd.DataFrame(corpus.queries_iter()).set_index('query_id').text.to_dict()

    docpairs['query'] = docpairs['query_id'].apply(lambda x : queries[x])
    docpairs['text_a'] = docpairs['doc_id_a'].apply(lambda x : docs[x])
    docpairs['text_b'] = docpairs['doc_id_b'].apply(lambda x : docs[x])

    tmp = []

    for row in docpairs.itertuples():
        tmp.append({'qid' : row.query_id, 'query' : row.query, 'docno' : row.doc_id_a, 'text' : row.text_a, 'relevance' : "TRUE"})
        tmp.append({'qid' : row.query_id, 'query' : row.query, 'docno' : row.doc_id_b, 'text' : row.text_b, 'relevance' : "FALSE"})

    return pd.DataFrame(tmp)

def encode_dataset(dataset, model, batch_size=32):
    query_lookup = dataset[['qid', 'query']].drop_duplicates()
    query_lookup['embedding'] = model(query_lookup['query'].tolist(), batch_size=batch_size, convert_to_numpy=True)
    query_lookup = query_lookup.set_index('qid')['embedding'].to_dict()

    doc_lookup = dataset[['docno', 'text']].drop_duplicates()
    doc_lookup['embedding'] = model(doc_lookup['text'].tolist(), batch_size=batch_size, convert_to_numpy=True)
    doc_lookup = doc_lookup.set_index('docno')['embedding'].to_dict()

    dataset['embedding'] = dataset.apply(lambda x : np.concatenate([query_lookup[x['qid']], doc_lookup[x['docno']]]), axis=1)

    return dataset

