from fastcore.all import *
from dataclasses import dataclass
import pickle
import pathlib
from natsort import natsorted
from more_itertools import flatten
import pandas as pd

@dataclass
class QueryDictWrapper:
    start_idx: int
    query_dicts: list


def load_pkl(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data

top100_qds = []
sorted_top100_paths = natsorted([path for path in pathlib.Path("/Users/ymh/Documents/Git_repos/NLP/gpt_exps/data/pkls/top100/").iterdir()], key=str)
sorted_nxt300_paths = natsorted([path for path in pathlib.Path("/Users/ymh/Documents/Git_repos/NLP/gpt_exps/data/pkls/top101_to_400/").iterdir()], key=str)

top100_pkls = list(map(lambda p: load_pkl(p).query_dicts, sorted_top100_paths))
nxt300_pkls = list(map(lambda p: load_pkl(p).query_dicts, sorted_nxt300_paths))
pkls = L(flatten(m) for m in (top100_pkls, nxt300_pkls)).concat()
df = pd.DataFrame(pkls)
df.to_csv("top_400.tsv", sep="\t")

# Wed Jul 7 2021: Loading orig_qds_400_to_2600.p
qds_ent_400_to_2600 = load_pkl("/Users/ymh/Documents/Git_repos/NLP/gpt_exps/data/extracted_from_kensho/qds_pkls/orig_qds_400_to_2600.p")
qds=  qds_ent_400_to_2600
assert len(qds) == 263760

qds[0]
qds[263759]