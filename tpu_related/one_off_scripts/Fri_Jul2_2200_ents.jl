#= Ideas to make workflow smoother:
* Make it so that you can use just one script to do everything, starting from the ent subsetting to this
* Write a function for writing those config files, and for pulling from remotes
* could add smtg to chk if config file has alrdy been pushed to repo

=#
include("manage_tpus.jl")


tpu_home = "/home/ymh"

n_tpus = 5 # TO CHK
tpu_prefix = "nxt300" # TO CHK
tpu_nms = ["nxt300_$i" for i in 0:(n_tpus-1)]

# Testing screen
# run_const_cmd_str_on_tpus("screen -d -m sleep 160", tpu_nms)
# ok process seems to stay alive after disc'ing ssh, so let's do it

len_lst_qds = 263760
qds_Δ = len_lst_qds // n_tpus #52752
qds_idxes = offset_start_idxes(offset=0, n_tpus=n_tpus, step=Int64(qds_Δ))

config_fnm = "config_400_to_2600_step52752.json" # TO CHK
query_cmd_maker(idx) = query_cmd_from_jl_idx(idx, qds_idxes, config_fnm)

@test query_cmd_maker(1) == "cd /home/ymh; source env/bin/activate; cd for_gvm_gpt; screen -d -m python3 query_gpt.py --config configs/config_400_to_2600_step52752.json --startidx 0"
@test query_cmd_maker(2) == "cd /home/ymh; source env/bin/activate; cd for_gvm_gpt; screen -d -m python3 query_gpt.py --config configs/config_400_to_2600_step52752.json --startidx 52752"


# 

git_pull_on_tpus(tpu_nms)
# to do: could add smtg to chk if config file has alrdy been pushed to repo
map_idx_cmd_over_tpus(query_cmd_maker, tpu_prefix, n_tpus)

