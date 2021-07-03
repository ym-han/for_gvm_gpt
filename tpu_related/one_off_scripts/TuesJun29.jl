# https://docs.julialang.org/en/v1/manual/asynchronous-programming/


# Key constants

n_tpus = 7

# Functions

function make_cmd(tpu_name, cmd_str)
  cmd = `gcloud alpha compute tpus tpu-vm ssh $tpu_name --command=$cmd_str`
end

function run_on_tpu(tpu_name, cmd_str) 
  cmd = make_cmd(tpu_name, cmd_str)
  read(cmd, String)
end

make_str(lst) = join(lst, "; ")  

# 0. Create the TPUs (see the python scripts for this)

# 1. Set up the TPUs appropriately
vm_setup_instructs = ["git clone https://github.com/ym-han/for_gvm_gpt.git",
                     "cd for_gvm_gpt/setup_vm/",
                     "chmod +x init_vm.sh",
                     "sh init_vm.sh"]
setup_instr_str = join(vm_setup_instructs, "; ")  


tpu_nms = ["top160_$i" for i in 0:(n_tpus-1)]

setup_tpu(tpu) = run_on_tpu(tpu, setup_instr_str)

@time begin
  setup_logs = asyncmap(setup_tpu, tpu_nms)
end

setup_logs


# hmm, seem to have run into installation issues, so let's sudo uninstall some of the stuff...
uninstall_instructs = ["sudo pip3 uninstall smart-open --yes",
                      "sudo pip3 uninstall transformers --yes",
                      "sudo pip3 uninstall Twisted --yes",
                      "sudo pip3 uninstall jax --yes",
                      "sudo pip3 uninstall jaxlib --yes",
                      "sudo pip3 uninstall torch --yes",
                      "sudo pip3 uninstall torch-xla --yes",
                      "sudo pip3 uninstall wandb --yes",
                      "sudo pip3 uninstall dm-haiku --yes"]
uninstall_ints =  join(uninstall_instructs, "; ")

@time begin
   asyncmap(tpu->run_on_tpu(tpu, "cd for*; git stash; git pull"), tpu_nms)
end


@time begin
   asyncmap(tpu->run_on_tpu(tpu, uninstall_instructs), tpu_nms)
end

# then reinstall
more_instructs = ["cd $tpu_home",
                  "source env/bin/activate",
                  "pip3 install -r mesh-transformer-jax/requirements.txt",
                  "pip3 install mesh-transformer-jax/ jax==0.2.12",
                  "pip3 uninstall jaxlib --yes",
                  "pip3 install jaxlib==0.1.67"]

@time begin
   asyncmap(tpu->run_on_tpu(tpu, make_str(more_instructs)), tpu_nms)
end

# 2. Query gpt

tpu_home = "/home/ymh"

start_idxes = range(0, step=500, length=n_tpus)
offset_start_idxes(offset) = range(offset, step=500, length=n_tpus)
round_2_start_idxes = offset_start_idxes(start_idxes[7])

makeshift_idxes = 1:7
function tpunm_from_makeshift_idx(idx)
# where idx is 1, 2, ... , 7
  py_idx = idx - 1
  tpu_nm = "top160_$py_idx" 
end

function query_cmd_from_makeshift_idx(idx, start_idxes)
  start_i = start_idxes[idx]
  py_cmds = ["cd $tpu_home", 
            "source env/bin/activate",
            "cd for_gvm_gpt",
            "python3 query_gpt.py --config configs/config_top100_step500.json --startidx $start_i"]
  make_str(py_cmds)
end

function query_gpt_on_makeshift_idx_round2(idx)
  tpu_nm = tpunm_from_makeshift_idx(idx)
  cmd_str = query_cmd_from_makeshift_idx(idx, round_2_start_idxes)
  cmd = make_cmd(tpu_nm, cmd_str)
  read(cmd, String)
end

# OK I didn't have time for round 2; let's go do it afer I wake up. and let's increase the step size if indeed it was working properly

@time begin
   asyncmap(query_gpt_on_makeshift_idx_round2, makeshift_idxes)
end

# ok looks like it somehow went faster than I had expected
# 3403.853799 seconds
#  (3403)/60 min ≈ 57 min
# let's say about 6.6 seconds per input qdict for each tpu; this is a very conservative estimate, since it does take a while for the model to start up

num_qdicts_100_ents = 12_000
# for 7 tpus, we'd need each tpu to go through num_qdicts_100_ents/7 qdicts = 1714.3 qdicts
# conserv est: going thru 1714 qdicts will take a tp about 188 min, or about 3h

# what if we increased the sampel size to 40 from 32?
# then instead of 12_000 qdicts for 100 ents, we'd have 12_000/4 * 5 = 15_000 qdicts
# about 4hours, with 7 tpus
# and if we made it 200 ents, 8 hours


# I should bump step_size up to 10k: that would be 1 hr 20 min

#= Notes
Re asyncmap:

function qtest(i)
  print("Hi!")
  sleep(3)
  "done!"
end

@time begin
  logs = asyncmap(qtest, 1:10)
end
=#