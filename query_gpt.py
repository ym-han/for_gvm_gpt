# Note re dependencies: jaxlib needs to be 0.1.67.
# when using a different version of jaxlib, error when running CausalTransformer: RuntimeError: Invalid argument: Argument does not match host shape or layout of computation parameter 0: want s32[]{:T(256)}, got s32[]

from utils_for_query import init_log_config, extract_nm_fr_resp, rm_white_space, download_blob, upload_blob, error_log_path, info_log_path

from datetime import datetime
import logging.config
from rich.logging import RichHandler
from notifiers.logging import NotificationHandler
from notifiers import get_notifier

import ujson
from pathlib import Path
import argparse
from copy import copy
from tqdm import tqdm
from google.cloud import storage
from smart_open import open
# import multiprocessing as mp
# from multiprocessing import get_context
# import parmap

from functools import partial
import itertools as itls
from fastcore.all import *
# from functional import pseq 

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

# ===== Logging, bucket related ===== #

bkt_nm = "coref_gpt"
bkt_pre = f"gs://{bkt_nm}/"
bkt_config = bkt_pre+"configs"

with open(bkt_config+"/wandb_key") as f:
	os.environ["WANDB_API_KEY"] = f.read()
import wandb

tg_params = ujson.load( open(bkt_config+"/telegram.json") )
tg = get_notifier('telegram')
def tg_notify(msg: str): tg.notify(message=msg, token=tg_params["token"], chat_id=tg_params["chat_id"])

# ===== Consts ====================== #

stop_received = False

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

def enum(lst): return L(L.range(lst), lst).zip()
# test_eq(
#     enum(["a", "b"]), 
#     L( [ (0, 'a'), (1, 'b') ] ) 
#     )


def stopped_base(logger, signum, frame):
	global stop_received
	stop_received = True

	logger.error("Stop signal received")


def upload_logs_to_bucket(start_time_date:str, tpu_nm:str):
	upload_blob(bkt_nm, str(error_log_path), f"logs/logs_started_{start_time_date}/{tpu_nm}/error.log")
	upload_blob(bkt_nm, str(info_log_path), f"logs/logs_started_{start_time_date}/{tpu_nm}/info.log")
   

def gracefully_exit_base(logger, start_time_date:str, tpu_nm:str):
	logger.info("Gracefully exited")
	upload_logs_to_bucket(start_time_date, tpu_nm)
	tg_notify(f"exitted - {tpu_nm}")

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
	parser.add_argument("--mock_setup", type=bool, default=False, help="whether to load network class or not")
	parser.add_argument("--step", type=int, default=False, help="number of (input / original) qdicts to infer")
  
	args = parser.parse_args()
	return args

def ask_func_for_testing(*args, **kwargs):
	return [str(i) for i in range(8)]

# Non-util funcs
def setup_gpt(setup_params, mock_setup=False):
	model_dir = setup_params["model_dir"]
	per_replica_batch = setup_params["per_replica_batch"]
	cores_per_replica = setup_params["cores_per_replica"]

	if mock_setup:
		tokenizer = transformers.GPT2TokenizerFast.from_pretrained('gpt2')
		total_batch = per_replica_batch * jax.device_count() // cores_per_replica
		return tokenizer, None, total_batch

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
def ask_gpt(setup_params:Dict=None, tokenizer=None, network=None, 
	total_batch=8, context=None, top_p=0.9, temp=0.9, gen_len=10, logger=None):

	logger.debug(f"top_p is {top_p};temp is {temp}\n")
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
  

""" Feed exactly one q_dict into GPT, get `batch_sz` responses back, augment, return """
def run_queries(batch_sz: int, ask_func, qd: Dict):  
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



# def save_bidx_qd_origidx(qd_save_dir, orig_qd_idx: int, batch_idx:int, ret_qd):
#     qd_savefnm = f"{qd_save_dir}/{orig_qd_idx}_{batch_idx}.json"

#     with open(bkt_pre+qd_savefnm, 'w') as f_out:
#         ujson.dump(ret_qd, f_out)


if __name__ == "__main__":
	tz = pytz.timezone('US/Eastern')
	start_time_date = datetime.now(tz).strftime("%d.%m.%y/%H.%M")
	short_starttd = datetime.now(tz).strftime("%d.%m/%H.%M")

	# Init params
	args = parse_args()
	
	# Set up params
	setup_params = std_params
	infer_config = ujson.load(open(args.config))
	setup_params.update(infer_config)

	# bkt_nm = setup_params["bucket"]
	input_qd_pkl_path, qd_save_dir = bkt_pre+setup_params["input_qd_pkl"], setup_params["qd_save_dir"]

	# Logging
	## Non-telegram logging:
	logging.config.dictConfig(init_log_config(args.tpunm))
	logger = logging.getLogger("root")
	logger.handlers[0] = RichHandler(markup=True)

	## Telegram logging:
	tg_hdlr = NotificationHandler('telegram', defaults=tg_params)
	tg_hdlr.setLevel(logging.ERROR)
	tg_hdlr.setFormatter( logging.Formatter('%(levelname)s - %(message)s - %(tpu_name)s') )
	logger.addHandler(tg_hdlr)

	logger = logging.LoggerAdapter(logger, {'tpu_name': str(args.tpunm)})

	## Pass logger as param to the relevant functions
	gracefully_exit = partial(gracefully_exit_base, logger)

	stopped = partial(stopped_base, logger)
	signal.signal(signal.SIGINT, stopped)
	signal.signal(signal.SIGTERM, stopped)


	start_idx, n_qdicts_to_infer = int(args.startidx), int(args.step)
	end_slice_idx = start_idx + n_qdicts_to_infer

	# Wandb
	run = wandb.init(project = "coref_gpt",
					 group = setup_params["group_id"],
					 job_type = "gptj_inference",
					 notes = "scaling up inference with batches",
					 name = f"50to100k_{short_starttd}_{str(args.tpunm)}_{str(args.startidx)}",
					 config = {**infer_config, 
							  "start_idx": args.startidx, 
							  "tpu_name": args.tpunm,
							  "config_file": args.config,
							  "n_qdicts_to_infer": n_qdicts_to_infer,
							  "input_qd_pkl": setup_params["input_qd_pkl"],
							  "qd_save_dir": setup_params["qd_save_dir"],
							  "tpu_size": setup_params["tpu_size"],
							  })
	
	# Make seq hyperparam/config match batch_sz  
	infer_batch_sz = int(setup_params["per_replica_batch"])
	if infer_batch_sz not in (1, 8): 
		raise Exception("infer batch sz should be either 1 or 8")
	setup_params["seq"] = 2048 // infer_batch_sz

	# Set up model and model query function
	if args.mock_setup:
		tokenizer, network, total_batch = setup_gpt(setup_params, mock_setup=True)
		ask = ask_func_for_testing
	else:
		tokenizer, network, total_batch = setup_gpt(setup_params)
		ask = partial(ask_gpt, setup_params=setup_params, tokenizer=tokenizer, network=network, total_batch=total_batch)

	def prompt_not_too_long(prompt): 
		return len(tokenizer.encode(prompt)) <= setup_params["seq"]


	# Load query dicts
	input_qds = pickle.load( open( input_qd_pkl_path, "rb" ) )

	qds_to_infer = ( (orig_idx, qd) for orig_idx, qd in 
						zip(range(start_idx, end_slice_idx), input_qds[start_idx:end_slice_idx]) 
						if prompt_not_too_long(qd["prompt"]) )

	## Log on wandb that I'm using pkl from before
	input_qdict_pkl = wandb.Artifact("akanvShort_in_qd_pkl_idx100k_to_200k_ents_from_edited_dump", type="input_qdict_pkl", description="idx 100k to 200_001 entities; no expn of uncert in qn prompts; based on EDITED kensho dump")
	input_qdict_pkl.add_reference(input_qd_pkl_path, name='akanvShort_in_qd_pkl_idx100k_to_200k_ents_from_edited_dump')
	run.use_artifact(input_qdict_pkl)


	# Run queries, save at regular chkpts (b/c tpu can be shutdown for maintenance etc)
	qds_skipped = []
	num_saved_pkls = 0
	qds_to_save = []

	for orig_idx, orig_qd in tqdm(qds_to_infer):
		logger.debug(f"Processing idx {orig_idx} of query dicts")
		
		# Chks
		if stop_received: gracefully_exit(start_time_date, args.tpunm)

		# len_prompt = len( tokenizer.encode(orig_qd["prompt"]) )
		# if len_prompt > setup_params["seq"]:
		#     logger.info(f"qd_idx {orig_idx}, entity {orig_qd['ent_nm']} too long; token length is {len_prompt} tokens")
		#     qds_skipped.append(orig_idx)
		#     continue
		
		# Query
		try:
			ret_dicts = run_queries(infer_batch_sz, ask, orig_qd)
		except Exception as inst:
			logger.exception(f"Fatal error while trying to query GPT with qd_idx {orig_idx}")

		qds_to_save.append( (orig_idx, list(ret_dicts)) )

		# Save if it's big enough, or if it's the last qdict
		max_qds_len = 45_000 # num here corr. to orig qd len
		if ( len(qds_to_save) >= max_qds_len) or (orig_idx == end_slice_idx - 1):
			path_suffix = f"{qd_save_dir}/from_orig_qd_{qds_to_save[0][0]}_to_{qds_to_save[-1][0]}.p"
			qd_lst_savefnm = bkt_pre + path_suffix
			num_saved_pkls += 1

			start_tm = time.time()
			pickle.dump( qds_to_save, open( qd_lst_savefnm, "wb" ) )
			logger.info(f"pkl {path_suffix} saved in {time.time() - start_tm}s")

			qds_to_save = []

	end_time = datetime.now(tz).strftime("%d.%m/%H.%M")
	if len(qds_skipped) > 0:
		logger.info(f"Done {end_time}; \nskipped {', '.join(qds_skipped)}\n")
	else:
		logger.info(f"Done {end_time}")

	tg_notify(f"DONE - {args.tpunm} - saved {num_saved_pkls} pkls")
	upload_logs_to_bucket(start_time_date, args.tpunm)
