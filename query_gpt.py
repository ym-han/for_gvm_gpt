# Notes re process: jaxlib needs to be 0.1.67.
# when using a different version of jaxlib, error when running CausalTransformer: RuntimeError: Invalid argument: Argument does not match host shape or layout of computation parameter 0: want s32[]{:T(256)}, got s32[]

from utils_for_query import logging_config, extract_nm_fr_resp, rm_white_space, download_blob, upload_blob, tg_notify

import logging
from notifiers.logging import NotificationHandler
import ujson
import pathlib
from functools import partial
import argparse
import os
import itertools as itls
from fastcore.all import *

from dataclasses import dataclass
import pickle
from typing import Dict, Optional, Tuple, List
from copy import deepcopy
from tqdm import tqdm

from google.cloud import storage
from smart_open import open

import numpy as np 

from jax.config import config

try: 
    import time

    import jax
    from jax.experimental import maps
    import numpy as np
    import optax
    import transformers

    from mesh_transformer.checkpoint import read_ckpt
    from mesh_transformer.sampling import nucleaus_sample
    from mesh_transformer.transformer_shard import CausalTransformer
except: 
    import time

    import jax
    from jax.experimental import maps
    import numpy as np
    import optax
    import transformers

    from mesh_transformer.checkpoint import read_ckpt
    from mesh_transformer.sampling import nucleaus_sample
    from mesh_transformer.transformer_shard import CausalTransformer


# LOGGING

## Non-telegram:
logging.config.dictConfig(logging_config)
logger = logging.getLogger("root")
logger.handlers[0] = RichHandler(markup=True)

## For telegram-logging:
config_tg_path = pathlib.Path("configs/telegram.json")
if not config_tg_path.is_file():  
    download_blob("coref_gpt", "misc/telegram.json", config_tg_path)
tg_params = ujson.load(open(config_tg_path))

tg_hdlr = NotificationHandler('telegram', defaults=tg_params)
tg_hdlr.setLevel(logging.ERROR)
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
tg_hdlr.setFormatter(c_format)
logger.addHandler(tg_hdlr)

# 

std_params = {"layers": 28,
              "d_model": 4096,
              "n_heads": 16,
              "n_vocab": 50400,
              "norm": "layernorm",
              "pe": "rotary",
              "pe_rotary_dims": 64,
              "sampler": nucleaus_sample,
              "optimizer": optax.scale(0), #from colab version
              "seq": 256,
              "tpu_size": 8}

# @dataclass
# class QueryDictWrapper:
#     start_idx: int
#     query_dicts: list


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Location of config file for setting up gpt-j")
    parser.add_argument("--startidx", type=int, default=None, help="start idx of the list for this tpu")
    parser.add_argument("--tpunm", type=int, default=None, help="name of tpu")
    args = parser.parse_args()
    return args


# Non-util funcs
def setup_gpt(setup_params):
    bucket, model_dir = setup_params["bucket"], setup_params["model_dir"]
    per_replica_batch = setup_params["per_replica_batch"]
    cores_per_replica = setup_params["cores_per_replica"]

    mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
    devices = np.array(jax.devices()).reshape(mesh_shape)
    maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(devices, ('dp', 'mp')))

    tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')

    """Here we create the network and load the parameters from the downloaded files. Expect this to take around 5 minutes."""
    total_batch = per_replica_batch * jax.device_count() // cores_per_replica

    network = CausalTransformer(setup_params)

    model_path = f"../{model_dir}/"
    network.state = read_ckpt(network.state, model_path, devices.shape[1])
    network.state = network.move_xmap(network.state, np.zeros(cores_per_replica))

    return tokenizer, network, total_batch


"""For interactive inference """
def infer(setup_params:Dict=None, tokenizer=None, network=None, total_batch=8, context=None, top_p=0.9, temp=0.9, gen_len=10):
    seq = setup_params["seq"]

    tokens = tokenizer.encode(context)

    provided_ctx = len(tokens)
    pad_amount = seq - provided_ctx

    padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
    batched_tokens = np.array([padded_tokens] * total_batch)
    length = np.ones(total_batch, dtype=np.uint32) * len(tokens)

    start = time.time()
    output = network.generate(batched_tokens, length, gen_len, {"top_p": np.ones(total_batch) * top_p, "temp": np.ones(total_batch) * temp})

    samples = []
    decoded_tokens = output[1][0]

    for o in decoded_tokens[:, :, 0]:
      samples.append(f"\033[1m{context}\033[0m{tokenizer.decode(o)}")

    print(f"completion done in {time.time() - start:06}s")
    return samples

def ask_gpt(setup_params:Dict=None, tokenizer=None, network=None, total_batch=8, context=None, top_p=0.9, temp=0.9, gen_len=10):
    print(f"top_p is {top_p};temp is {temp}\n")
    seq = setup_params["seq"]

    tokens = tokenizer.encode(context)

    provided_ctx = len(tokens)
    pad_amount = seq - provided_ctx

    padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
    batched_tokens = np.array([padded_tokens] * total_batch)
    length = np.ones(total_batch, dtype=np.uint32) * len(tokens)

    output = network.generate(batched_tokens, length, gen_len, {"top_p": np.ones(total_batch) * top_p, "temp": np.ones(total_batch) * temp})

    samples = []
    decoded_tokens = output[1][0]

    for o in decoded_tokens[:, :, 0]:
      samples.append(f"{tokenizer.decode(o)}")

    return samples

def make_k_copies(obj, k: int): 
    return L([ deepcopy(obj) for _ in range(k) ])

def add_resp_to_qdict(qdict, response_str:str):
    # qdict, response_str = qdict_resp_tuple
    qdict["response"] = extract_nm_fr_resp(response_str)
    return qdict
  

def run_queries(batch_sz: int, ask_func, query_dicts : List[Dict]) -> List[Dict]:  

    # 1. For each qdict `qd`, make batch_sz number of replicates, and feed `qd` into GPT to get a list of batch_sz responses
    # 2. Augment the batch_sz replicates with the responses
    # 3. Return list of replicates
    logger.debug(f"batch_sz is {batch_sz}")

    if len(query_dicts) == 0: 
        raise Exception("`run_queries` got mt qd_dict list as input?!")

    ret_qdicts = []
    
    for qd_idx, qd in enumerate(tqdm(query_dicts)):
        logger.debug(f"Processing idx {qd_idx} of query dicts")
        gpt_outs = ask_func(context=qd["prompt"], top_p=qd["top_p"], temp=qd["temp"])
        batch_qds = make_k_copies(qd, batch_sz)
        lst_qds_with_responses = batch_qds.map_zipwith(add_resp_to_qdict, gpt_outs)

        ret_qdicts.extend(lst_qds_with_responses)
      
    return ret_qdicts


if __name__ == "__main__":
    # Init params
    args = parse_args()
    setup_params = std_params
    setup_params.update(ujson.load(open(args.config)))
    bucket, orig_qd_path = setup_params["bucket"], setup_params["orig_qd_path"]
    qd_save_dir = setup_params["qd_save_dir"]

    # Check that batch_sz matches seq hyperparam/config
    infer_batch_sz = int(setup_params["per_replica_batch"])
    if infer_batch_sz == 8 and int(setup_params["seq"]) != 256:
        raise Exception("seq needs to be equal to 2048/8 if per_replica_batch (the batch sz) == 8!") 

    # Set up model and model query function
    tokenizer, network, total_batch = setup_gpt(setup_params)
    ask = partial(ask_gpt, setup_params=setup_params, tokenizer=tokenizer, network=network, total_batch=total_batch)

    # Load query dicts
    dest_qd_path = pathlib.Path(orig_qd_path)
    if not dest_qd_path.parent.is_dir(): dest_qd_path.parent.mkdir() 
    download_blob(bucket, orig_qd_path,  orig_qd_path)

    start_idx = int(args.startidx)
    n_qdicts_to_infer = int(setup_params["n_qdicts_to_infer_per_tpu"])
    end_idx = start_idx + n_qdicts_to_infer

    all_query_dicts = pickle.load( open( dest_qd_path, "rb" ) )
    qdicts_to_infer = all_query_dicts[start_idx: end_idx]

    # Qeury GPT and get updated query dicts

    try:
        ret_qdicts = run_queries(infer_batch_sz, ask, qdicts_to_infer)
    except Exception as inst:
        logger.exception("Fatal error while trying to process query_dicts")

    # Save updated query dicts
    ## TO DO: Save the data in json instead, and make sure to stream it / save it lazily, instead of keeping everything in memory and saving only at the end.




    # qdw = QueryDictWrapper(start_idx, ret_qdicts)
    # logger.debug(qdw)

    tg.notify(f"DONE - {args.tpunm}")

    qdw_savefnm = f"qdw_{start_idx}_to_{end_idx}.p"
    pickle.dump( qdw, open(qdw_savefnm, "wb") )
    upload_blob(bucket, qdw_savefnm, f"{qd_save_dir}/"+qdw_savefnm)


# # stream from GCS
# for line in open('gs://my_bucket/my_file.txt'):
#     print(line)

# # stream content *into* GCS (write mode):
# with open('gs://my_bucket/my_file.txt', 'wb') as fout:
#     fout.write(b'hello world')