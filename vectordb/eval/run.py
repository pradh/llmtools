from dataclasses import asdict, dataclass, fields
from datasets import load_dataset
from sentence_transformers.util import semantic_search
from typing import List
import numpy as np

import asyncio
import csv
import logging
import os
import time
import torch
import shutil

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('store_embeddings', '', 'Store Embeddings CSV file')
flags.DEFINE_string('test_embeddings', '', 'Test Embeddings CSV file')
flags.DEFINE_string('system', 'all', 'baseline | redis | chroma | lance | all')

_INDEX = 'idx-vars'
_DIM = 384
_LIMIT = 100000
_TOPK = 40
_BATCH_SZ = 50


@dataclass
class Data:
    dcids: List[str]
    sentences: List[str]
    embeddings: List[List[float]]


@dataclass
class Stats:
    system: str = ''
    connection_time_sec: float = -1
    index_creation_time_sec: float = -1
    total_rows: int = 0
    loading_time_sec: float = -1
    num_queries: int = 0
    num_matches: int = 0
    search_time_sec: float = -1
    search_api_latency_sec: float = -1


#
# Helpers
#


def write_stats(stats: List[Stats]):
    with open('stats.csv', 'w') as fp:
        cols = [f.name for f in fields(Stats())]
        csvw = csv.DictWriter(fp, fieldnames=cols)
        csvw.writeheader()
        csvw.writerows([asdict(it) for it in stats])


def load_embeddings(name: str,
                    embeddings_file: str,
                    lim: int = _LIMIT) -> Data:
    ds = load_dataset('csv', data_files=embeddings_file)
    df = ds['train'].to_pandas()
    dcids = df['dcid'].values.tolist()
    df = df.drop('dcid', axis=1)
    sentences = df['sentence'].values.tolist()
    df = df.drop('sentence', axis=1)
    embeddings = df.to_numpy().tolist()
    logging.info(f'Loaded {len(dcids)} "{name}" embeddings!')
    return Data(dcids[:lim], sentences[:lim], embeddings[:lim])


class Timeit:

    def __init__(self):
        self.start = time.time()

    def reset(self):
        ret = time.time() - self.start
        self.start = ret + self.start
        return ret


#
# Sentence Transformer
#
def run_baseline(store: Data, test: Data) -> Stats:
    stats = Stats(system='Baseline (SentenceTransformer)')

    store = torch.tensor(store.embeddings, dtype=torch.float)
    query = torch.tensor(test.embeddings, dtype=torch.float)

    t = Timeit()
    hits = semantic_search(query, store, top_k=_TOPK)
    stats.num_matches = 0
    for hit in hits:
        stats.num_matches += len(hit)
    stats.search_time_sec = t.reset()
    stats.search_api_latency_sec = stats.search_time_sec / len(test.embeddings)
    logging.info(f'Baseline returned {stats.num_matches} results')

    return stats


#
#  REDIS Stuff
#
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query


def _create_redis_index(client):
    try:
        # check to see if index exists
        client.ft(_INDEX).info()
        client.ft(_INDEX).dropindex(delete_documents=True)
        logging.info('Redis index already exists, deleting!')
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

    # create Index
    client.ft(_INDEX).create_index(
        fields=schema, definition=IndexDefinition(index_type=IndexType.JSON))


def _search_redis_index(client, test: Data):
    query = (Query('(*)=>[KNN 40 @vector $query_vector AS vector_score]').
             sort_by('vector_score').return_fields('vector_score', 'dcid',
                                                   'sentence').dialect(2))
    tot_results = 0
    for emb in test.embeddings:
        result_docs = client.ft(_INDEX).search(
            query, {
                'query_vector': np.array(emb, dtype=np.float32).tobytes()
            } | {}).docs
        tot_results += len(result_docs)
    logging.info(f'Redis returned {tot_results} results')

    return tot_results


def _load_redis_index(client, data: Data):
    pipeline = client.pipeline()
    keys = []
    for i, s, e in zip(data.dcids, data.sentences, data.embeddings):
        redis_key = f'{i}:{s}'
        pipeline.json().set(redis_key, '$', {
            'dcid': i,
            'sentence': s,
            'embeddings': e
        })
        keys.append(redis_key)
    result = pipeline.execute()
    logging.info(f'Loaded {sum(result)} docs into Redis successfully!')
    # logging.info(json.dumps(client.json().get(keys[0]), indent=2))

    # Check the index
    info = client.ft(_INDEX).info()
    num_docs = info['num_docs']
    indexing_failures = info['hash_indexing_failures']
    total_indexing_time = info['total_indexing_time']
    percent_indexed = float(info['percent_indexed']) * 100
    logging.info(
        f"{num_docs} documents ({percent_indexed} percent) indexed with {indexing_failures} failures in {float(total_indexing_time):.2f} milliseconds"
    )


def run_redis(store: Data, test: Data) -> Stats:
    stats = Stats(system='Redis-VSS')

    t = Timeit()
    client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    client.ping()
    stats.connection_time_sec = t.reset()

    _create_redis_index(client)
    stats.index_creation_time_sec = t.reset()

    _load_redis_index(client, store)
    stats.loading_time_sec = t.reset()

    stats.num_matches = _search_redis_index(client, test)
    stats.search_time_sec = t.reset()
    stats.search_api_latency_sec = stats.search_time_sec / len(test.embeddings)
    return stats


#
# Chroma Stuff
#
import chromadb


def run_chroma(store: Data, test: Data) -> Stats:
    stats = Stats(system='Chroma')
    if os.path.exists('/tmp/.chroma'):
        shutil.rmtree('/tmp/.chroma')

    t = Timeit()
    client = chromadb.PersistentClient('/tmp/.chroma')
    stats.connection_time_sec = t.reset()

    collection = client.create_collection(name=_INDEX,
                                          metadata={"hnsw:space": "cosine"})
    stats.index_creation_time_sec = t.reset()

    nadded = 0
    while nadded < len(store.dcids):
        s, e = nadded, nadded + 5000
        collection.add(
            embeddings=store.embeddings[s:e],
            documents=store.dcids[s:e],
            ids=[
                f'{i}:{s}'
                for i, s in zip(store.dcids[s:e], store.sentences[s:e])
            ])
        nadded += len(store.dcids[s:e])

    stats.loading_time_sec = t.reset()
    logging.info(f'Indexed {len(store.dcids)} docs into Chroma')

    stats.num_matches = 0
    for emb in test.embeddings:
        results = collection.query(
            query_embeddings=emb,
            n_results=_TOPK,
        )
        stats.num_matches += len(results)
    stats.search_time_sec = t.reset()
    stats.search_api_latency_sec = stats.search_time_sec / len(test.embeddings)
    logging.info(f'Chroma returned {stats.num_matches} results')

    return stats


#
# Lance Stuff
#

import lancedb


async def _search_lance_async_int(tbl, emb) -> int:
    t = Timeit()
    df = await tbl.vector_search(emb).distance_type('cosine').limit(
        _TOPK).to_pandas()
    return len(df.values.tolist()), t.reset()


async def _search_async_lance(db, test: Data) -> int:
    tbl = await db.open_table(_INDEX)

    ndone = 0
    tot_results = 0
    tot_latency = 0.0
    while ndone < len(test.embeddings):
        s, e = ndone, ndone + 10
        sl = test.embeddings[s:e]
        results = []
        for emb in sl:
            results.append(_search_lance_async_int(tbl, emb))
        results = await asyncio.gather(*results)
        tot_results += sum([r[0] for r in results])
        tot_latency += sum([r[1] for r in results])
        ndone += len(sl)

    logging.info(f'Lance returned {tot_results}')
    return tot_results, tot_latency / ndone


async def _run_async_lance(store: Data, test: Data) -> Stats:
    stats = Stats(system='LanceDB Async')

    if os.path.exists('/tmp/.lancedb'):
        shutil.rmtree('/tmp/.lancedb')

    t = Timeit()
    db = await lancedb.connect_async("/tmp/.lancedb")
    stats.connection_time_sec = t.reset()

    records = []
    for d, s, e in zip(store.dcids, store.sentences, store.embeddings):
        records.append({'vector': e, 'dcid': d, 'sentence': s})
    await db.create_table(_INDEX, records)
    stats.loading_time_sec = t.reset()
    logging.info(f'Indexed {len(store.dcids)} docs into LanceDB')

    stats.num_matches, stats.search_api_latency_sec = await _search_async_lance(
        db, test)
    stats.search_time_sec = t.reset()

    return stats


def run_async_lance(store: Data, test: Data) -> Stats:
    return asyncio.run(_run_async_lance(store, test))


def _search_sync_lance(db, test: Data) -> int:
    tbl = db.open_table(_INDEX)

    tot_results = 0
    for i, emb in enumerate(test.embeddings):
        li = tbl.search(emb).metric('cosine').limit(_TOPK).to_list()
        tot_results += len(li)
        if i % 1000 == 999:
            logging.info(f'... {i} queried')

    logging.info(f'Lance returned {tot_results}')
    return tot_results


def run_sync_lance(store: Data, test: Data) -> Stats:
    stats = Stats(system='LanceDB Sync')

    if os.path.exists('/tmp/.lancedb'):
        shutil.rmtree('/tmp/.lancedb')

    t = Timeit()
    db = lancedb.connect("/tmp/.lancedb")
    stats.connection_time_sec = t.reset()

    records = []
    for d, s, e in zip(store.dcids, store.sentences, store.embeddings):
        records.append({'vector': e, 'dcid': d, 'sentence': s})
    db.create_table(_INDEX, records)
    stats.loading_time_sec = t.reset()
    logging.info(f'Indexed {len(store.dcids)} docs into LanceDB')

    stats.num_matches = _search_sync_lance(db, test)
    stats.search_time_sec = t.reset()
    stats.search_api_latency_sec = stats.search_time_sec / len(test.embeddings)

    return stats


#
# Vertex AI
#
from google.cloud import aiplatform_v1

# Set variables for the current deployed index.
_API_ENDPOINT = "302175072.us-central1-496370955550.vdb.vertexai.goog"
_INDEX_ENDPOINT = "projects/496370955550/locations/us-central1/indexEndpoints/8500794985312944128"
_DEPLOYED_INDEX_ID = "dc_all_minilm_l6_v2_ft_1709655496660"


def _vertex_search(client, test: Data):
    ndone = 0
    tot_results = 0
    ncalls = 0
    tot_latency = 0.0
    while ndone < len(test.embeddings):
        s, e = ndone, ndone + _BATCH_SZ
        sl = test.embeddings[s:e]
        queries = []
        for emb in sl:
            # Build FindNeighborsRequest object
            queries.append(
                aiplatform_v1.FindNeighborsRequest.Query(
                    datapoint=aiplatform_v1.IndexDatapoint(feature_vector=emb),
                    # The number of nearest neighbors to be retrieved
                    neighbor_count=_TOPK))
        request = aiplatform_v1.FindNeighborsRequest(
            index_endpoint=_INDEX_ENDPOINT,
            deployed_index_id=_DEPLOYED_INDEX_ID,
            # Request can have multiple queries
            queries=queries,
            return_full_datapoint=False,
        )
        # Execute the request
        t = Timeit()
        response = client.find_neighbors(request)
        tot_latency += t.reset()
        ncalls += 1
        for ns in response.nearest_neighbors:
            tot_results += len(ns.neighbors)
        ndone += len(sl)

    logging.info(f'Vertex AI returned {tot_results}')
    return tot_results, tot_latency / ncalls


def run_vertex(_, test: Data) -> Stats:
    stats = Stats(system='Vertex AI')

    # Configure Vector Search client
    t = Timeit()
    vector_search_client = aiplatform_v1.MatchServiceClient(
        client_options={"api_endpoint": _API_ENDPOINT})
    stats.connection_time_sec = t.reset()

    stats.num_matches, stats.search_api_latency_sec = _vertex_search(
        vector_search_client, test)
    stats.search_time_sec = t.reset()

    return stats


#
# Main
#


def main(_):
    _ROUTER = {
        'baseline': [run_baseline],
        'redis': [run_redis],
        'chroma': [run_chroma],
        'lance_async': [run_async_lance],
        'lance_sync': [run_sync_lance],
        'vertex': [run_vertex],
        'all': [
            run_baseline, run_redis, run_chroma, run_async_lance,
            run_sync_lance, run_vertex
        ],
    }
    stats = []
    data = load_embeddings('store', FLAGS.store_embeddings)
    test = load_embeddings('test', FLAGS.test_embeddings)
    for fn in _ROUTER[FLAGS.system]:
        s = fn(data, test)
        s.total_rows = len(data.embeddings)
        s.num_queries = len(test.embeddings)
        stats.append(s)
    write_stats(stats)


if __name__ == "__main__":
    app.run(main)
