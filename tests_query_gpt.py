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


# Ideas / things to check:
# 1. Check the shape of logits, and understand the sampling code --- in particular, how many samples we're taking at each time step, and HOW we are choosing that sample.
# 2. Try vhanging seq to seq / batch_sz
# 3. Try increasing the number of samples at each step (but review how languagee gneration and how GPT works first or smtg)




# ========
setup_params = std_params
setup_params.update(json.load(open("configs/test_config.json")))
bucket, orig_qd_path = setup_params["bucket"], setup_params["orig_qd_path"]
qd_save_dir = setup_params["qd_save_dir"]
setup_params["cores_per_replica"] = 8
setup_params["per_replica_batch"] = 8 # batch sz; for testing
setup_params["seq"] = 256 # maybe THIS was the reason why it hadn't worked properly earlier?

# Set up model and model query function
tokenizer, network, total_batch = setup_gpt(setup_params)
ask = partial(ask_gpt, setup_params=setup_params, tokenizer=tokenizer, network=network, total_batch=total_batch, gen_len=5)
inf = partial(infer, setup_params=setup_params, tokenizer=tokenizer, network=network, total_batch=total_batch, gen_len=5)
# Load query dicts
dest_qd_path = pathlib.Path(orig_qd_path)
if not dest_qd_path.parent.is_dir(): dest_qd_path.parent.mkdir() 
download_blob(bucket, "test_querydicts/orig_qds_test_old.p",  orig_qd_path)

start_idx = 0
n_qdicts_to_infer = int(setup_params["n_qdicts_to_infer_per_tpu"])
end_idx = start_idx + n_qdicts_to_infer

all_query_dicts = pickle.load( open( dest_qd_path, "rb" ) )
qdicts_to_infer = all_query_dicts[start_idx: end_idx]

# Qeury GPT and get updated query dicts
infer_batch_sz = int(setup_params["per_replica_batch"])
ret_qdicts = run_queries(infer_batch_sz, ask, qdicts_to_infer)




def days_for_entities(n_ents, time_per_req = 0.3349, n_samples_per_ent=500, n_tpus=10):
    print(f"n_ents is {n_ents}, time_per_req is {time_per_req}, n_samples_per_ent is {n_samples_per_ent}, n_tpus is {n_tpus}")
    secs = (n_ents * time_per_req * n_samples_per_ent) / n_tpus 
    days = secs / (60 * 60 * 24)
    print(f"{days} days or {secs} secs\n")
    return days

def days_for_entities_v2_short_prompt(n_ents, n_samples_per_ent=500):
    return days_for_entities(n_ents, time_per_req=0.32865, n_samples_per_ent=n_samples_per_ent)

days_for_entities(30_000, time_per_req=0.2109194, n_tpus=100, n_samples_per_ent=500)
days_for_entities(30_000, time_per_req=0.2109252, n_tpus=10, n_samples_per_ent=300)


def days_for_entities_v3tpu_long_prompt(n_ents=40_000, n_tpus=10, n_samples_per_ent=500): 
    return days_for_entities(n_ents, time_per_req = 0.2133, n_tpus=n_tpus, n_samples_per_ent=n_samples_per_ent)

def days_for_entities_v3tpu_short_prompt(n_ents=40_000, n_tpus=10, n_samples_per_ent=500): 
    return days_for_entities(n_ents, time_per_req = 0.2107, n_tpus=n_tpus, n_samples_per_ent=n_samples_per_ent)

n_ents = 100_000
diff = days_for_entities(n_ents, n_tpus=10) - days_for_entities(n_ents, time_per_req=0.3260779, n_tpus=10) 
# difference between my current prompt and `test_prompt_even_shorter_no_md` = 12.3 days

days_for_entities(n_ents, time_per_req = 0.335, n_tpus=100)
#46.5
days_for_entities(n_ents, time_per_req = 0.328, n_tpus=100)
#45.6

# v2 TPUs:
days_for_entities(100_000, n_tpus=10, n_samples_per_ent=1_000)
# ~39



# V3 TPUs:

days_for_entities_v3tpu_short_prompt(n_ents=100_000, n_tpus=10, n_samples_per_ent=1_000)
#  24.4 days

days_for_entities_v3tpu_long_prompt(n_ents=100_000, n_tpus=10, n_samples_per_ent=1_000)
#24.7 days

from timeit import default_timer as timer
start = timer()
ret_qdicts = run_queries(infer_batch_sz, ask, qdicts_to_infer)
end=timer()
time_for_two_input_qds = end-start
time_per_input_qd = time_for_two_input_qds/2
# ==== BATCH of 8 ===
#6.014 seconds for 8 reqs (this is < 6.45 b/c gen_len = 5 instead of 10)
# In [24]: 6.014/8
# Out[24]: 0.75175

# another trial:
# In [32]: time_per_input_qd
# Out[32]: 5.9973688009995385
# In [33]: time_per_input_qd/8
# Out[33]: 0.7496711001249423


# ==== SINGLE REQ ===
# In [39]: time_per_input_qd
# Out[39]: 0.33491511249997075



def get_token_len(tokenizer, context):
    tokens = tokenizer.encode(context)
    return len(tokens)
token_len = partial(get_token_len, tokenizer)


from timeit import default_timer as timer

def avg_time_with_prompt_for_single_req(prompt, top_p=0.9, temp=1):
    def run_once():
        start = timer()
        test_rets = ask(context=prompt, top_p=top_p, temp=temp)
        end=timer()
        time = end-start
        return time

    times = [run_once() for i in range(10)]

    return np.mean(times)


def avg_time_with_prompt_for_batch(prompt, top_p=0.9, temp=1, batch_sz=8):
    def run_once():
        start = timer()
        test_rets = ask(context=prompt, top_p=top_p, temp=temp)
        end=timer()
        time_for_batch = end-start
        time_per_req = time_for_batch/batch_sz
        return time_per_req

    time_per_req_s = [run_once() for i in range(10)]

    return np.mean(time_per_req_s)

# largest prompt

test_prompt = """I'm talking to a brilliant AI that can give me alternative names and nicknames for people, groups, concepts, and entities. If it doesn't know what the thing is, or what alternative names for it are, it will respond with a [?]. It will give me colorful alternative names or nicknames like:
###
Description: U.S. basketball player.
Kobe Bryant aka [Black Mamba]
###
Description: federal city in and former capital of Russia.
St. Petersburg aka [Leningrad]
###
Description: Academic at a state school.
Sam XYZ aka [?]
because either this is not known, or alternative names can't be found for it.
###
Description: People who don't have valid immigration documentation for the country they live in.
Undocumented immigrants aka [Illegal aliens]
###
Description: Medicine for the stomach.
Diffaminsorin aka [?]
because either this is not known, or alternative names can't be found for it.
###
Description: U.S. singer, songwriter, record producer, and actress.
Beyoncé aka [Queen Bey]
###
Description: American politician.
Barack Obama aka ["""
test_prompt = rm_white_space(test_prompt)

avg_t_sing = avg_time_with_prompt_for_single_req(test_prompt, top_p=0.9, temp=1)
avg_t_sing

avg_t_sing_top_p1 = avg_time_with_prompt_for_single_req(test_prompt, top_p=1, temp=1)
avg_t_sing_top_p1



avg_t = avg_time_with_prompt_for_batch(test_prompt, top_p=0.9, temp=1)
avg_t

avg_t_p1 = avg_time_with_prompt_for_batch(test_prompt, top_p=1, temp=1)
avg_t_p1





# OK what if we use a smaller prompt? (but again, gen_len = 5)


test_prompt = """Here are some colorful alternative names and nicknames.
###
Description: federal city in and former capital of Russia.
St. Petersburg aka [Leningrad]
###
Description: People who don't have valid immigration documentation for the country they live in.
Undocumented immigrants aka [Illegal aliens]
###
Description: U.S. singer, songwriter, record producer, and actress.
Beyoncé aka [Queen Bey]
###
Description: American politician.
Barack Obama aka ["""
test_prompt = rm_white_space(test_prompt)

#tkn len: 108
token_len(test_prompt)


avg_t = avg_time_with_prompt_for_batch(test_prompt)
avg_t
# 0.7494 17

avg_t_single = avg_time_with_prompt_for_single_req(test_prompt)
avg_t_single
#v2:  0.32805316310004856
#v3: 0.2109



test_prompt_short_md = """Here are some colorful alternative names and nicknames.
###
Type: City
St. Petersburg aka [Leningrad]
###
Type: Group.
Undocumented immigrants aka [Illegal aliens]
###
Type: Celebrity.
Beyoncé aka [Queen Bey]
###
Type: Politician.
Barack Obama aka ["""
test_prompt_short_md = rm_white_space(test_prompt_short_md)

token_len(test_prompt_short_md)
#74

avg_t_short_md = avg_time_with_prompt_for_batch(test_prompt_short_md)
avg_t_short_md
# 0.7494 96695

avg_t_single = avg_time_with_prompt_for_single_req(test_prompt_short_md)
avg_t_single
#v2:  0.3261
#v3: 0.21095340

test_prompt_no_md = """Here are some colorful alternative names and nicknames.
St. Petersburg aka [Leningrad]
Undocumented immigrants aka [Illegal aliens]
Beyoncé aka [Queen Bey]
Barack Obama aka ["""
test_prompt_no_md = rm_white_space(test_prompt_no_md)

token_len(test_prompt_no_md)
#46

avg_t = avg_time_with_prompt_for_batch(test_prompt_no_md)
avg_t
# 0.7494 87703100067

avg_t_single = avg_time_with_prompt_for_single_req(test_prompt_no_md)
avg_t_single
#v2: 0.32801611039999445
#v3: 0.2107

test_prompt_even_shorter_no_md = """Here are some colorful alternative names and nicknames.
Undocumented immigrants aka [Illegal aliens]
Beyoncé aka [Queen Bey]
Barack Obama aka ["""
test_prompt_even_shorter_no_md = rm_white_space(test_prompt_even_shorter_no_md)

token_len(test_prompt_even_shorter_no_md)
#36

avg_t = avg_time_with_prompt_for_batch(test_prompt_even_shorter_no_md)
avg_t
# 0.7494 906

avg_t_single = avg_time_with_prompt_for_single_req(test_prompt_even_shorter_no_md)
avg_t_single
#v2: 0.3286482526000782
#v3: 0.2107


