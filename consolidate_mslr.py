"""Pack 31531 per-query npz files into two consolidated files for fast loading."""
import numpy as np
import scipy.sparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'mslr')
FILE_DIR = os.path.join(DATA_DIR, 'mslr_processed')
OUT_REL  = os.path.join(DATA_DIR, 'mslr_all_rel.npy')
OUT_FEAT = os.path.join(DATA_DIR, 'mslr_all_feat.npz')

meta = np.load(os.path.join(DATA_DIR, 'mslr.npz'), allow_pickle=True)
n_queries = len(meta['docsPerQuery'])
print(f"Packing {n_queries} queries from {FILE_DIR}", flush=True)

def load_query(qID):
    with np.load(os.path.join(FILE_DIR, f'{qID}_rel.npz')) as f:
        rel = f['relevances'].copy()
    feat = scipy.sparse.load_npz(os.path.join(FILE_DIR, f'{qID}_feat.npz'))
    return qID, rel, feat

N_WORKERS = 32
results = [None] * n_queries

with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futures = {ex.submit(load_query, i): i for i in range(n_queries)}
    done = 0
    for fut in as_completed(futures):
        qID, rel, feat = fut.result()
        results[qID] = (rel, feat)
        done += 1
        if done % 2000 == 0:
            print(f"  {done}/{n_queries}", flush=True)

print("Stacking...", flush=True)
all_rel  = np.concatenate([r[0] for r in results])
all_feat = scipy.sparse.vstack([r[1] for r in results], format='csr')

print(f"Saving rel  ({all_rel.shape})  -> {OUT_REL}", flush=True)
np.save(OUT_REL, all_rel)
print(f"Saving feat ({all_feat.shape}) -> {OUT_FEAT}", flush=True)
scipy.sparse.save_npz(OUT_FEAT, all_feat)
print("Done.", flush=True)
