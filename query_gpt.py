# Notes re process: jaxlib needs to be 0.1.67.
# when using a different version of jaxlib, error when running CausalTransformer: RuntimeError: Invalid argument: Argument does not match host shape or layout of computation parameter 0: want s32[]{:T(256)}, got s32[]

from utils_for_query import init_log_config, extract_nm_fr_resp, rm_white_space, download_blob, upload_blob, tg_notify, error_log_path, info_log_path

from datetime import datetime
import logging.config
from rich.logging import RichHandler
from notifiers.logging import NotificationHandler
import wandb

import ujson
from pathlib import Path
import argparse
from copy import copy
from tqdm import tqdm
from google.cloud import storage
from smart_open import open
import multiprocessing as mp
from multiprocessing import get_context

from functools import partial
import itertools as itls
from fastcore.all import *
# from functional import pseq 

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import dill as pickle

import pytz
import os
import sys
import signal

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

# ===== Consts ===== #

stop_received = False
bkt_nm = "coref_gpt"
bkt_pre = f"gs://{bkt_nm}/"

std_params = {"layers": 28,
              "d_model": 4096,
              "n_heads": 16,
              "n_vocab": 50400,
              "norm": "layernorm",
              "pe": "rotary",
              "pe_rotary_dims": 64,
              "sampler": nucleaus_sample,
              "optimizer": optax.scale(0), #from colab version
              "tpu_size": 8}
              # "seq": 256,

#==== Functions =====#

def stopped(signum, frame):
    global stop_received
    stop_received = True

    logging.error("Stop signal received")


def upload_logs_to_bucket(start_time_date:str, tpu_nm:str):
    upload_blob(bkt_nm, str(error_log_path), f"logs_started_/{tpu_nm}/error.log")
    upload_blob(bkt_nm, str(info_log_path), f"logs_started_/{tpu_nm}/info.log")
   

def gracefully_exit(start_time_date:str, tpu_nm:str):
    logging.info("Gracefully exited")
    upload_logs_to_bucket(start_time_date, tpu_nm)
    tg.notify("exitted - {tpu_nm}")

    sys.exit(0)


# @dataclass
# class QueryDictWrapper:
#     start_idx: int
#     query_dicts: list


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Location of config file for setting up gpt-j")
    parser.add_argument("--startidx", type=int, default=None, help="start idx of the dicts to process for this tpu")
    parser.add_argument("--tpunm", type=str, default=None, help="name of tpu")
  
    args = parser.parse_args()
    return args


# Non-util funcs
def setup_gpt(setup_params):
    model_dir = setup_params["model_dir"]
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

""" If total_batch == 8, then this returns a list of 8 responses """
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

# Shallow copies good enough b/c it's a dict of primitive types
def make_copies(obj, k: int=8): 
    return [ copy(obj) for _ in range(k) ]
    # TO DO: Test that the copies are indep of each other!


def add_resp_to_qdict(qdict, response_str:str):
    qdict["response"] = extract_nm_fr_resp(response_str)
    return qdict
  

def save_bidx_qd_origidx(qd_save_dir, orig_qd_idx: int, batch_idx:int, ret_qd):
    qd_savefnm = f"{qd_save_dir}/{orig_qd_idx}_{batch_idx}.json"

    with open(bkt_pre+qd_savefnm, 'w') as f_out:
        ujson.dump(ret_qd, f_out)

""" Feed exactly one q_dict into GPT, get `batch_sz` responses back, augment, return """
def run_queries(batch_sz: int, ask_func, qd: Dict) -> Iterable[Dict]:  
    logger.debug(f"batch_sz is {batch_sz}")

    # 1. Replicate query dict `batch_sz` times, and feed `qd` into GPT to get a list of batch_sz responses
    # 2. Augment the batch_sz replicates with the responses
    # 3. Return list of replicates
    if batch_sz not in (1, 8):
        log.info("batch_sz not either 1 or 8")
        batch_sz = 8

    gpt_outs = ask_func(context=qd["prompt"], top_p=qd["top_p"], temp=qd["temp"])
    
    batch_qds = make_copies(qd, batch_sz)
    ret_qdicts = L(batch_qds).map_zipwith(add_resp_to_qdict, gpt_outs)

    # ret_qdicts = ( pseq(batch_qds)
    #                     .zip( pseq(gpt_outs) )
    #                     .smap(add_resp_to_qdict) )

    return ret_qdicts 


if __name__ == "__main__":
    tz = pytz.timezone('US/Eastern')
    start_time_date = datetime.now(tz).strftime("%d.%m.%y/%H.%M")

    # Init params
    args = parse_args()
    
    signal.signal(signal.SIGINT, stopped)
    signal.signal(signal.SIGTERM, stopped)

    ## Non-telegram logging:
    logging.config.dictConfig(init_log_config(args.tpunm))
    logger = logging.getLogger("root")
    logger.handlers[0] = RichHandler(markup=True)

    ## Telegram logging:
    config_tg_path = "configs/telegram.json"
    tg_params = ujson.load( open(bkt_pre+config_tg_path) )

    tg_hdlr = NotificationHandler('telegram', defaults=tg_params)
    tg_hdlr.setLevel(logging.ERROR)
    tg_hdlr.setFormatter( logging.Formatter('%(levelname)s - %(message)s - %(tpu_name)s') )
    logger.addHandler(tg_hdlr)

    logger = logging.LoggerAdapter(logger, {'tpu_name': str(args.tpunm)})

    # Set up params
    setup_params = std_params
    infer_config = ujson.load(open(args.config))
    setup_params.update(infer_config)

    run = wandb.init(project = "coref_gpt",
                    group = setup_params["group_id"],
                    job_type = "gptj_inference",
                    notes = "scaling up inference with batches",
                    config = infer_config)

    global bkt_nm 
    bkt_nm = setup_params["bucket"]
    input_qd_pkl_path, qd_save_dir = setup_params["input_qd_pkl"], setup_params["qd_save_dir"]
    
    start_idx = int(args.startidx)
    n_qdicts_to_infer = int(setup_params["n_qdicts_to_infer_per_tpu"])
    end_slice_idx = start_idx + n_qdicts_to_infer

    # Make seq hyperparam/config match batch_sz  
    infer_batch_sz = int(setup_params["per_replica_batch"])
    if infer_batch_sz not in (1, 8): 
        raise Exception("infer batch sz should be either 1 or 8")
    setup_params["seq"] = 2048 // infer_batch_sz

    # Set up model and model query function
    tokenizer, network, total_batch = setup_gpt(setup_params)
    ask = partial(ask_gpt, setup_params=setup_params, tokenizer=tokenizer, network=network, total_batch=total_batch)

    # Load query dicts
    input_qds = pickle.load( open( input_qd_pkl_path, "rb" ) )
    qds_to_infer = input_qds[start_idx:end_slice_idx]
    
    ## Log on wandb that I'm using pkl from before
    input_qdict_pkl = wandb.Artifact("akanv_in_qd_pkl_50k_ents", type="input_qdict_pkl", description="for 50k entities")
    input_qdict_pkl.add_reference(input_qd_pkl_path, name='akanv_in_qd_pkl_50k_ents')
    run.use_artifact(input_qdict_pkl)


    # Run queries, save as we go
    for qd_idx, qd in enumerate(tqdm(qds_to_infer)):
        logger.debug(f"Processing idx {qd_idx} of query dicts")
        
        if stop_received: gracefully_exit(start_time_date, args.tpunm)
        
        save_bidx_qd = partial(save_bidx_qd_origidx, qd_save_dir, qd_idx)

        try:
            ret_dicts = run_queries(infer_batch_sz, ask, orig_qd)
        except Exception as inst:
            logger.exception(f"Fatal error while trying to query GPT with qd_idx {qd_idx}")

        with get_context("spawn").Pool() as pool:
            pool.starmap(save_bidx_qd, enum(ret_dicts))
            # TO DO: Look into using async starmap instead


    logger.info("Done\n")
    tg.notify(f"DONE - {args.tpunm}")









    # pseq( ret_dicts.enumerate() ).smap(save_bidx_qd)     

    # json_str = ujson.dumps(ret_qd)
    # fout.write(json_str)






    # dest_qd_path = Path(orig_qd_dir)
    # if not dest_qd_path.parent.is_dir(): dest_qd_path.parent.mkdir() 
    # download_blob(bucket, orig_qd_dir,  orig_qd_dir)

    # all_query_dicts = pickle.load( open( dest_qd_path, "rb" ) )
    # qdicts_to_infer = all_query_dicts[start_idx: end_slice_idx]

    # # Query GPT and get updated query dicts

    # try:
    #     ret_qdicts = run_queries(infer_batch_sz, ask, qdicts_to_infer)
    # except Exception as inst:
    #     logger.exception("Fatal error while trying to process query_dicts")

    # Save updated query dicts



    # qdw = QueryDictWrapper(start_idx, ret_qdicts)
    # logger.debug(qdw)

    # qdw_savefnm = f"qdw_{start_idx}_to_{end_slice_idx}.p"
    # pickle.dump( qdw, open(qdw_savefnm, "wb") )
    # upload_blob(bucket, qdw_savefnm, f"{qd_save_dir}/"+qdw_savefnm)


# # stream from GCS
# for line in open('gs://my_bucket/my_file.txt'):
#     print(line)

# # stream content *into* GCS (write mode):
# with open('gs://my_bucket/my_file.txt', 'wb') as fout:
#     fout.write(b'hello world')