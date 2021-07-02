#= Screen: 
  https://orcaman.blogspot.com/2013/08/google-compute-engine-keeping-your.html 
  https://stackoverflow.com/questions/48221807/google-cloud-instance-terminate-after-close-browser
=#  

# https://docs.julialang.org/en/v1/manual/asynchronous-programming/
using Test

# Key constants

n_tpus = 7

# Functions

function calc_time(n_tpus, n_input_qds)
  startup_time = 5 * 60
  time_per_input_qd = 6.45 #seconds
  secs_required = startup_time + time_per_input_qd * n_input_qds / n_tpus

  mins_required = secs_required/60
  hours_required = mins_required/60

  println("this will take at least $mins_required min or $hours_required hours")
  return mins_required
end


function make_cmd(tpu_name, cmd_str)
  cmd = `gcloud alpha compute tpus tpu-vm ssh $tpu_name --command=$cmd_str`
end

function run_on_tpu(tpu_name, cmd_str) 
  cmd = make_cmd(tpu_name, cmd_str)
  println("calling gcloud with: $cmd")

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
n_tpus = 5
tpu_nms = ["nxt300_$i" for i in 0:(n_tpus-1)]

offset_start_idxes(; offset=0, step=500, n_tpus=n_tpus) = range(offset, step=step, length=n_tpus)
@test offset_start_idxes(n_tpus=5) == 0:500:2000
@test offset_start_idxes(n_tpus=7) == 0:500:3000
@test offset_start_idxes(offset=3, n_tpus=2) == 3:500:503
  

function tpunm_from_makeshift_idx(idx, tpu_prefix)
  # where idx is 1, 2, ... , 7
  py_idx = idx - 1
  tpu_nm = "$tpu_prefix" * "_$py_idx" 
end

@test tpunm_from_makeshift_idx(1, "tpu") == "tpu_0"
@test tpunm_from_makeshift_idx(2, "tpu") == "tpu_1"

# screen -d -m
function make_query_cmd_from_makeshift_idx(idx, idx_range, config_fnm)
  start_i = idx_range[idx]
  py_cmds = ["cd $tpu_home", 
            "source env/bin/activate",
            "cd for_gvm_gpt",
            "nohup python3 query_gpt.py --config configs/$config_fnm --startidx $start_i"]
  make_str(py_cmds)
end

# where cmd_maker is ftn from idx to cmd_string
function run_query_cmd(idx, cmd_maker)
  tpu_nm = tpunm_from_makeshift_idx(idx, tpu_prefix)
  cmd_str = cmd_maker(idx)

  run_on_tpu(tpu_nm, cmd_str)
end

function map_over_tpus(cmd_maker, n_tpus)
  makeshift_jl_idxes = 1:n_tpus
  
  @time begin
     asyncmap(i -> run_query_cmd(i, cmd_maker), makeshift_jl_idxes)
  end
end


# OK I didn't have time for round 2; let's go do it afer I wake up. and let's increase the step size if indeed it was working properly

# Thurs night finishing up the top 100 collection with our 5 tpus
round_2_step = Integer((12_000 - 7*500) / n_tpus)
round_2_qds_idxes = offset_start_idxes(offset=3_000, n_tpus=5, step=Integer(round_2_step))

round2_query_cmd_maker(idx) = make_query_cmd_from_makeshift_idx(idx, round_2_qds_idxes, "config_rest_of_top100_step1700.json")

@test round2_query_cmd_maker(1) == "cd /home/ymh; source env/bin/activate; cd for_gvm_gpt; nohup python3 query_gpt.py --config configs/config_rest_of_top100_step1700.json --startidx 3000"
@test round2_query_cmd_maker(2) == "cd /home/ymh; source env/bin/activate; cd for_gvm_gpt; nohup python3 query_gpt.py --config configs/config_rest_of_top100_step1700.json --startidx 4700"

map_over_tpus(round2_query_cmd_maker, 5)


# running the 5 non-preemptible on the 300 after top 100:



# 1. Set up the TPUs appropriately
n_tpus = 5
tpu_nms = ["nxt300_$i" for i in 0:(n_tpus-1)]
makeshift_jl_idxes = 1:n_tpus

vm_setup_instructs = ["git clone https://github.com/ym-han/for_gvm_gpt.git",
                     "cd for_gvm_gpt/setup_vm/",
                     "chmod +x init_vm.sh",
                     "sh init_vm.sh"]
setup_instr_str = join(vm_setup_instructs, "; ")  


setup_tpu(tpu) = run_on_tpu(tpu, setup_instr_str)

@time begin
  setup_logs = asyncmap(setup_tpu, tpu_nms)
end


@time begin
  setup_logs = asyncmap(tpu->run_on_tpu(tpu, "killall screen"), tpu_nms)
end

@time begin
  lses = asyncmap(tpu->run_on_tpu(tpu, make_str(["cd for*/configs", "ls"])), tpu_nms)
end

# 2. Query gpt
tpu_prefix = "nxt300"

makeshift_jl_idxes = 1:n_tpus

@test tpunm_from_makeshift_idx(1, tpu_prefix) == "nxt300_0"
@test tpunm_from_makeshift_idx(2, tpu_prefix) == "nxt300_1"

nxt300ents_idx_range = offset_start_idxes(offset=0, step=7200)

cmd_maker(idx) = make_query_cmd_from_makeshift_idx(idx, nxt300ents_idx_range, "config_next300_after_top100.json")

@test cmd_maker(1) == "cd /home/ymh; source env/bin/activate; cd for_gvm_gpt; nohup python3 query_gpt.py --config configs/config_next300_after_top100.json --startidx 0"
@test cmd_maker(2) == "cd /home/ymh; source env/bin/activate; cd for_gvm_gpt; nohup python3 query_gpt.py --config configs/config_next300_after_top100.json --startidx 7200"


@time begin
   asyncmap(i -> run_query_cmd(i, cmd_maker), makeshift_jl_idxes)
end


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
