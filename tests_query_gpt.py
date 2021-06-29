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


# main logic
# os.system("python tests_query_gpt.py --config configs/TEST_config_gpt_j.json --startidx 0")