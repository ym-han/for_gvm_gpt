# Notes re process: jaxlib needs to be 0.1.67.
# when using a different version of jaxlib, error when running CausalTransformer: RuntimeError: Invalid argument: Argument does not match host shape or layout of computation parameter 0: want s32[]{:T(256)}, got s32[]

import logging
logging.basicConfig(level=logging.INFO)

import pathlib
import argparse
import json
import os
from functools import partial
import itertools as itls
from fastcore.all import *

from dataclasses import dataclass
import pickle
from typing import Dict, Optional, Tuple, List
from copy import deepcopy
from tqdm import tqdm

from google.cloud import storage
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


@dataclass
class QueryDictWrapper:
    start_idx: int 
    query_dicts: list


# Util funcs
def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )



def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    logging.info(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Location of config file for setting up gpt-j")
    parser.add_argument("--startidx", type=int, default=None, help="start idx of the list for this tpu")

    args = parser.parse_args()
    return args

""" Extracts the name from the output. 
    This assumes that the name is enclosed in brackets """
def extract_nm_fr_resp(response:str = "") -> str:
    if "]" not in response: return "no closing ]"

    end_idx = response.find("]")
    return response[: end_idx]

def rm_white_space(strg: str) -> str:
    lines_without_trailing = strg.strip().split("\n")
    no_whitespace = "\n".join( [ln.strip() for ln in lines_without_trailing] )

    return no_whitespace


# Non-util funcs
def setup_gpt(setup_params):
    bucket, model_dir = setup_params["bucket"], setup_params["model_dir"]
    per_replica_batch = setup_params["per_replica_batch"]
    cores_per_replica = setup_params["cores_per_replica"]

    setup_params["sampler"] = nucleaus_sample
    setup_params["optimizer"] = optax.scale(0) #from colab version

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
def infer(setup_params, tokenizer, network, total_batch, context, top_p=0.9, temp=0.9, gen_len=10):
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

def ask_gpt(setup_params, tokenizer, network, context, top_p=0.9, temp=0.9, gen_len=10):
    #print(f"top_p is {top_p};temp is {temp}\n")
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

    if len(query_dicts) == 0: logging.debug("`run_queries` got mt qd_dict list as input?!")

    ret_qdicts = []

    for qd in query_dicts:
        gpt_outs = ask_func(qd["prompt"], top_p=qd["top_p"], temp=qd["temp"])
        batch_qds = make_k_copies(qd, batch_sz)
        lst_qds_with_responses = batch_qds.map_zipwith(add_resp_to_qdict, gpt_outs)

        ret_qdicts.extend(lst_qds_with_responses)
  
    return ret_qdicts




if __name__ == "__main__":
    # Init params
    args = parse_args()
    setup_params = json.load(open(args.config))
    bucket, orig_qd_path = setup_params["bucket"], setup_params["orig_qd_path"]
    qd_save_dir = setup_params["qd_save_dir"]
    
    # Set up model and model query function
    tokenizer, network, total_batch = setup_gpt(setup_params)
    ask = partial(ask_gpt, setup_params, tokenizer, network, total_batch)

    # Load query dicts
    dest_qd_path = pathlib.Path(orig_qd_path)
    if not dest_qd_path.parent.is_dir(): dest_qd_path.parent.mkdir() 
    download_blob(bucket, orig_qd_path,  orig_qd_path)

    start_idx = int(args.startidx)
    n_qdicts_to_infer = int(setup_params["n_qdicts_to_infer_per_tpu"])
    end_idx = start_idx + n_qdicts_to_infer

    all_query_dicts = pickle.load( open( dest_qd_path, "rb" ) )
    qdicts_to_infer = all_query_dicts[start_idx: end_idx]

    # Qeury GPT and update query dicts
    infer_batch_sz = int(setup_params["cores_per_replica"])
    run_queries(infer_batch_sz, ask, qdicts_to_infer)

    # Save updated query dicts
    qdw = QueryDictWrapper(start_idx, qdicts_to_infer)

    qdw_savefnm = f"qdw_{start_idx}.p"
    pickle.dump( qdw, open(qdw_savefnm, "wb") )
    upload_blob(bucket, qdw_savefnm, "{qd_save_dir}/"+qdw_savefnm)














