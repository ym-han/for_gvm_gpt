# Some quick tests

from fastcore.all import *
import os
import query_gpt

batch_sz = 3 # in reality this will be 1 or 8

toy_qdict_1 = [{"prompt": "p1", "ent_nm": "nm1"}]
toy_qdicts_2 = [{"prompt": "p1", "ent_nm": "nm1"},
              {"prompt": "p2", "ent_nm": "nm2"}]


def ask_func_for_testing(*args):
    return [str(i) for i in range(batch_sz)]


# extract_nm_fr_resp
test_eq(extract_nm_fr_resp(""), "no closing ]")
test_eq(extract_nm_fr_resp("Do not enquire from the centurion nodding"), "no closing ]")

test_eq(extract_nm_fr_resp("blah] 2222 !"), "blah")


# run_queries
stock_res = [str(i) for i in range(batch_sz)]

toy_ret_qds_1 = run_queries(batch_sz, ask_func_for_testing, toy_qdicts_1)
test_eq(toy_ret_qds_1, [ {"prompt": "p1", "ent_nm": "nm1", "response": stock_res},
                         {"prompt": "p1", "ent_nm": "nm1", "response": stock_res} ])

toy_ret_qds_2 = run_queries(batch_sz, ask_func_for_testing, toy_qdicts_2)
test_eq(toy_ret_qds_2, 
    [ {"prompt": "p1", "ent_nm": "nm1", 
    "response": stock_res} for _ in range(batch_sz*len(toy_qdicts_2))]
    )


# main logic
# os.system("python tests_query_gpt.py --config configs/TEST_config_gpt_j.json")