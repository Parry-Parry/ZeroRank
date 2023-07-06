import pandas as pd

def construct_id_lookup(inputs):
    lookup = {}
    for row in inputs.itertuples():
        lookup[row.Index] = {'query' : row.query, 'document' : row.text, 'relevance' : row.relevance}
    return lookup

def construct_query_lookup(inputs):
    lookup = {}
    for row in inputs.itertuples():
        lookup[row.Index] = {'query' : row.query, 'document_a' : row.text_a, 'document_b' : row.text_b, 'relevance' : row.relevance}
    return lookup

def format_dataset(corpus, cutoff=0, keep_triples=False):
    docpairs = pd.DataFrame(corpus.docpairs_iter())
    if cutoff: docpairs = docpairs.sample(n=cutoff, random_state=42)
    docs = pd.DataFrame(corpus.docs_iter()).set_index('doc_id').text.to_dict()
    queries = pd.DataFrame(corpus.queries_iter()).set_index('query_id').text.to_dict()

    docpairs['query'] = docpairs['query_id'].apply(lambda x : queries[x])
    docpairs['text_a'] = docpairs['doc_id_a'].apply(lambda x : docs[x])
    docpairs['text_b'] = docpairs['doc_id_b'].apply(lambda x : docs[x])

    tmp = []
    if keep_triples:
        for row in docpairs.itertuples():
            tmp.append({
                'qid' : row.query_id,
                'query' : row.query,
                'docno_a' : row.doc_id_a,
                'text_a' : row.text_a,
                'docno_b' : row.doc_id_b,
                'text_b' : row.text_b,
            })
    else:  
        for row in docpairs.itertuples():
            tmp.append({'qid' : row.query_id, 'query' : row.query, 'docno' : row.doc_id_a, 'text' : row.text_a, 'relevance' : "true"})
            tmp.append({'qid' : row.query_id, 'query' : row.query, 'docno' : row.doc_id_b, 'text' : row.text_b, 'relevance' : "false"})

    return pd.DataFrame(tmp)

def encode_triples(dataset, model, batch_size=32):
    query_lookup = dataset[['qid', 'query']].drop_duplicates()
    query_lookup['embedding'] = model(query_lookup['query'].tolist(), batch_size=batch_size, convert_to_numpy=True)
    query_lookup = query_lookup.set_index('qid')['embedding'].to_dict()

    docs = dataset[['docno_a', 'text_a']].drop_duplicates().tolist() + dataset[['docno_b', 'text_b']].drop_duplicates().tolist()
    doc_lookup = pd.DataFrame(docs, columns=['docno', 'text']).drop_duplicates()
    doc_lookup['embedding'] = model(doc_lookup['text'].tolist(), batch_size=batch_size, convert_to_numpy=True)
    doc_lookup = doc_lookup.set_index('docno')['embedding'].to_dict()

    dataset['q_embedding'] = dataset['query_id'].apply(lambda x : query_lookup[x], axis=1)
    dataset['a_embedding'] = dataset['doc_id_a'].apply(lambda x : doc_lookup[x], axis=1)
    dataset['b_embedding'] = dataset['doc_id_b'].apply(lambda x : doc_lookup[x], axis=1)

    return dataset.reset_index(drop=True)

def encode_querydoc(dataset, model, batch_size=32):
    query_lookup = dataset[['qid', 'query']].drop_duplicates()
    query_lookup['embedding'] = model(query_lookup['query'].tolist(), batch_size=batch_size, convert_to_numpy=True)
    query_lookup = query_lookup.set_index('qid')['embedding'].to_dict()

    doc_lookup = dataset[['docno', 'text']].drop_duplicates()
    doc_lookup['embedding'] = model(doc_lookup['text'].tolist(), batch_size=batch_size, convert_to_numpy=True)
    doc_lookup = doc_lookup.set_index('docno')['embedding'].to_dict()

    dataset['q_embedding'] = dataset['qid'].apply(lambda x : query_lookup[x], axis=1)
    dataset['d_embedding'] = dataset['docno'].apply(lambda x : doc_lookup[x], axis=1)
    return dataset.reset_index(drop=True)