# Some quick tests

from fastcore.all import *
import os
from query_gpt import *

batch_sz = 3 # in reality this will be 1 or 8

toy_qdict_1 = [{"prompt": "p1", "ent_nm": "nm1", "top_p": 0.8, "temp": 0.7}]
toy_qdict_2 = [{"prompt": "p1", "ent_nm": "nm1", "top_p": 0.8, "temp": 0.9},
              {"prompt": "p2", "ent_nm": "nm2", "top_p": 0.8, "temp": 0.9}]


def ask_func_for_testing(*args, **kwargs):
    return [str(i) for i in range(batch_sz)]


# extract_nm_fr_resp
test_eq(extract_nm_fr_resp(""), "no closing ]")
test_eq(extract_nm_fr_resp("Do not enquire from the centurion nodding"), "no closing ]")

test_eq(extract_nm_fr_resp("blah] 2222 !"), "blah")


# run_queries
stock_res = [str(i) for i in range(batch_sz)]

toy_ret_qds_1 = run_queries(batch_sz, ask_func_for_testing, toy_qdict_1)
test_eq(toy_ret_qds_1, 
    [{"prompt": "p1", "ent_nm": "nm1", "top_p": 0.8, "temp": 0.7, "response": "no closing ]"},
    {"prompt": "p1", "ent_nm": "nm1", "top_p": 0.8, "temp": 0.7, "response": "no closing ]"},
    {"prompt": "p1", "ent_nm": "nm1", "top_p": 0.8, "temp": 0.7, "response": "no closing ]"} ])

toy_ret_qds_2 = run_queries(batch_sz, ask_func_for_testing, toy_qdict_2)

test_eq( len(toy_ret_qds_2), 6)
test_eq( toy_ret_qds_2, 
        [{'prompt': 'p1',
          'ent_nm': 'nm1',
          'top_p': 0.8,
          'temp': 0.9,
          'response': 'no closing ]'},
         {'prompt': 'p1',
          'ent_nm': 'nm1',
          'top_p': 0.8,
          'temp': 0.9,
          'response': 'no closing ]'},
         {'prompt': 'p1',
          'ent_nm': 'nm1',
          'top_p': 0.8,
          'temp': 0.9,
          'response': 'no closing ]'},
         {'prompt': 'p2',
          'ent_nm': 'nm2',
          'top_p': 0.8,
          'temp': 0.9,
          'response': 'no closing ]'},
         {'prompt': 'p2',
          'ent_nm': 'nm2',
          'top_p': 0.8,
          'temp': 0.9,
          'response': 'no closing ]'},
         {'prompt': 'p2',
          'ent_nm': 'nm2',
          'top_p': 0.8,
          'temp': 0.9,
          'response': 'no closing ]'}] )


# ==== main logic
# os.system("python query_gpt.py --config configs/test_config.json --startidx 0")
# pickle.load( open("qdw_0.p", "rb"))

# === for debugging
# setup_params = json.load(open("configs/test_config.json"))
# bucket, orig_qd_path = setup_params["bucket"], setup_params["orig_qd_path"]
# qd_save_dir = setup_params["qd_save_dir"]

# # Set up model and model query function
# tokenizer, network, total_batch = setup_gpt(setup_params)
# ask = partial(ask_gpt, setup_params=setup_params, tokenizer=tokenizer, network=network, total_batch=total_batch)

# # Load query dicts
# dest_qd_path = pathlib.Path(orig_qd_path)
# if not dest_qd_path.parent.is_dir(): dest_qd_path.parent.mkdir() 
# download_blob(bucket, orig_qd_path,  orig_qd_path)

# start_idx = int(args.startidx)
# n_qdicts_to_infer = int(setup_params["n_qdicts_to_infer_per_tpu"])
# end_idx = start_idx + n_qdicts_to_infer

# all_query_dicts = pickle.load( open( dest_qd_path, "rb" ) )
# qdicts_to_infer = all_query_dicts[start_idx: end_idx]

# # Qeury GPT and get updated query dicts
# infer_batch_sz = int(setup_params["per_replica_batch"])
# ret_qdicts = run_queries(infer_batch_sz, ask, qdicts_to_infer)


# from timeit import default_timer as timer
# start = timer()
# ret_qdicts = run_queries(infer_batch_sz, ask, qdicts_to_infer)
# end=timer()
# time_for_two_input_qds = end-start
# time_per_input_qd = time_for_two_input_qds/2
#6.45 seconds