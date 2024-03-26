from dataclasses import dataclass
from datasets import load_dataset
import json
from typing import List
import numpy as np

import chromadb
import lancedb
import redis

from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('embeddings_csv', '', 'Embeddings CSV file')
flags.DEFINE_string('system', 'redis', 'redis | chroma | lance')

_INDEX = 'idx-vars'
_DIM = 384
_LIMIT = 1000


@dataclass
class Data:
    dcids: List[str]
    sentences: List[str]
    embeddings: List[List[float]]


def get_data(embeddings_file: str) -> Data:
    ds = load_dataset('csv', data_files=embeddings_file)
    df = ds['train'].to_pandas()
    dcids = df['dcid'].values.tolist()
    df = df.drop('dcid', axis=1)
    sentences = df['sentence'].values.tolist()
    df = df.drop('sentence', axis=1)
    embeddings = df.to_numpy().tolist()
    return Data(dcids[:_LIMIT], sentences[:_LIMIT], embeddings[:_LIMIT])


def _create_redis_index(client):
    try:
        # check to see if index exists
        client.ft(_INDEX).info()
        print('Index already exists!')
        client.ft(_INDEX).dropindex(delete_documents=True)
    except:
        pass

    # schema
    schema = (
        TextField('$.dcid', no_stem=True, as_name='dcid'),
        TextField('$.sentence', no_stem=True, as_name='sentence'),
        VectorField('$.embeddings',
                    'FLAT', {
                        'TYPE': 'FLOAT32',
                        'DIM': _DIM,
                        'DISTANCE_METRIC': 'COSINE',
                    },
                    as_name='vector'),
    )

    # index Definition
    definition = IndexDefinition(prefix=['vars:'], index_type=IndexType.JSON)

    # create Index
    client.ft('idx:vars').create_index(fields=schema, definition=definition)

    # Check the index
    info = client.ft(_INDEX).info()
    print(info)
    num_docs = info['num_docs']
    indexing_failures = info['hash_indexing_failures']
    total_indexing_time = info['total_indexing_time']
    percent_indexed = float(info['percent_indexed']) * 100
    print(
        f"{num_docs} documents ({percent_indexed} percent) indexed with {indexing_failures} failures in {float(total_indexing_time):.2f} milliseconds"
    )


def _search_redis_index(client, query_embeddings):
    query = (Query('(*)=>[KNN 3 @vector $query_vector AS vector_score]').
             sort_by('vector_score').return_fields('vector_score', 'dcid',
                                                   'sentence').dialect(2))
    result_docs = client.ft(_INDEX).search(query, {
        'query_vector':
        np.array(query_embeddings, dtype=np.float32).tobytes()
    } | {}).docs
    for doc in result_docs:
        vector_score = round(1 - float(doc.vector_score), 2)
        print({
            'query': query,
            'score': vector_score,
            'id': doc.dcid,
            'brand': doc.sentence,
        })


def run_redis(data: Data):
    client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    client.ping()

    pipeline = client.pipeline()
    keys = []
    for i, s, e in zip(data.dcids, data.sentences, data.embeddings):
        redis_key = f'v:{i:s}'
        pipeline.json().set(redis_key, '$', {
            'dcid': i,
            'sentence': s,
            'embeddings': e
        })
        keys.append(redis_key)

    pipeline.execute()
    print(json.dumps(client.json().get(keys[0]), indent=2))

    _create_redis_index(client)
    _search_redis_index(client, data.embeddings[0])


def run_chroma(data: Data):
    client = chromadb.Client()
    collection = client.create_collection(name=_INDEX,
                                          metadata={"hnsw:space": "cosine"})
    collection.add(
        embeddings=data.embeddings,
        documents=data.dcids,
        ids=[f'{i}:{s}' for i, s in zip(data.dcids, data.sentences)])
    print('Indexed docs into Chroma')

    result = collection.query(
        query_embeddings=data.embeddings[0],
        n_results=10,
    )
    print(result)

    return


def _search_lance(db, embeddings: List[float]):
    tbl = db.open_table(_INDEX)
    result = tbl.search(embeddings).limit(10).to_list()
    print(result)


def run_lance(data: Data):
    db = lancedb.connect("~/.lancedb")

    records = []
    for d, s, e in zip(data.dcids, data.sentences, data.embeddings):
        records.append({'vector': e, 'dcid': d, 'sentence': s})
    db.create_table(_INDEX, records)
    _search_lance(db, data.embeddings[0])
    return


def main(_):
    data = get_data(FLAGS.embeddings_csv)
    if FLAGS.system == 'redis':
        run_redis(data)
    elif FLAGS.system == 'chroma':
        run_chroma(data)
    elif FLAGS.system == 'lance':
        run_lance(data)


if __name__ == "__main__":
    app.run(main)
