# Cleaner version of code for running embarrasingly parallel jobs on TPUs
# TO DO: Add setup code (tho not sure if doing this is worth it)

# This assumes 0-based idxing for tpu names

using Test

const zone_central = "us-central1-f"
const zone_europe = "europe-west4-a"

# Util functions
function calc_time(n_tpus, n_input_qds)
  startup_time = 5 * 60
  time_per_input_qd = 6.45 #seconds
  secs_required = startup_time + time_per_input_qd * n_input_qds / n_tpus

  mins_required = secs_required/60
  hours_required = mins_required/60

  println("this will take at least $mins_required min or $hours_required hours")
  return mins_required
end

# https://cloud.google.com/sdk/gcloud/reference/alpha/compute/ssh
function make_cmd(tpu_name::String, cmd_str::String; zone::String=zone_central, test_mode=false)
  if test_mode
    cmd = `gcloud alpha compute tpus tpu-vm ssh $tpu_name --zone=$zone --dry-run --command=$cmd_str`    
  else
    cmd = `gcloud alpha compute tpus tpu-vm ssh $tpu_name --zone=$zone --command=$cmd_str` 
  end
end

function run_on_tpu(tpu_name, cmd_str::String; zone::String=zone_central, test_mode::Bool=false) 
  cmd = make_cmd(tpu_name, cmd_str; zone=zone, test_mode=test_mode)
  println("calling gcloud with: $cmd")

  read(cmd, String)
end

make_str(lst) = join(lst, "; ")  

""" Note that the cmd str for this function is the same across all the tpus.
In particular, the cmd str here does not depend on tpu idx. """
function run_const_cmd_str_on_tpus(cmd_str, tpu_nms; zone::String=zone_central, test_mode::Bool=false)
  println("running the following on tpus: $cmd_str")

  @time begin
    asyncmap(tpu->run_on_tpu(tpu, cmd_str, zone=zone, test_mode=test_mode), tpu_nms)
  end
end

function tpunm_from_jl_idx(idx::Int64, tpu_prefix::String)
  # where idx is 1, 2, ... , 7
  py_idx = idx - 1
  tpu_nm = "$tpu_prefix" * "_$py_idx" 
end
@test tpunm_from_jl_idx(1, "tpu") == "tpu_0"
@test tpunm_from_jl_idx(2, "tpu") == "tpu_1"

# a cleaner way to do this would be to have this be a special case of 
# a idx_dep_cmd_from_jl_idx(idx, py_cmds) function
function cmd_str_from_jl_idx(idx, qdict_idx_range, config_fnm::String)
  start_i = qdict_idx_range[idx]
  py_cmds = ["cd $tpu_home", 
          "source env/bin/activate",
          "cd for_gvm_gpt",
          "screen -d -m python3 query_gpt.py --config configs/$config_fnm --startidx $start_i"]

  make_str(py_cmds)
end
#screen -d -m
#nohup

# where cmd_maker is ftn from jl idx to cmd string
function run_query_cmd(tpu_nm_idx::Tuple, cmd_maker; zone::String=zone_central, test_mode::Bool=false)
  idx, tpu_nm = tpu_nm_idx
  cmd_str = cmd_maker(idx)

  run_on_tpu(tpu_nm, cmd_str, zone=zone, test_mode=test_mode)
end 

git_pull_on_tpus(tpu_nms; zone::String=zone_central, test_mode::Bool=false) = run_const_cmd_str_on_tpus("cd for*; git stash; git pull", tpu_nms, zone=zone, test_mode=test_mode)

function map_idxed_cmd_over_tpus(cmd_maker, tpu_nms; zone::String=zone_central, test_mode::Bool=false)
  tpu_nms_idxes = zip(1:length(tpu_nms), tpu_nms)
  
  @time begin
     asyncmap(nm_idx -> run_query_cmd(nm_idx, cmd_maker, zone=zone, test_mode=test_mode), tpu_nms_idxes)
  end
end

offset_start_idxes(; offset=0, step=500, n_tpus=n_tpus) = range(offset, step=step, length=n_tpus)

@test offset_start_idxes(n_tpus=5) == 0:500:2000
@test offset_start_idxes(n_tpus=7) == 0:500:3000
@test offset_start_idxes(offset=3, n_tpus=2) == 3:500:503


# Querying gpt

## Things that will need to be changed for different inputs

function example_of_querying_gpt()
  tpu_home = "/home/ymh"

  n_tpus = 5
  tpu_prefix = "tpu"
  tpu_nms = [tpu_prefix * "_" * string(i) for i in 0:n_tpus-1]
  # we're assuming 0-based idxing for tpu_nms

  round_2_step = Integer((12_000 - 7*500) / n_tpus)
  round_2_qds_idxes = offset_start_idxes(offset=3_000, n_tpus=5, step=Integer(round_2_step))
  round2_query_cmd_maker(idx) = cmd_str_from_jl_idx(idx, round_2_qds_idxes, "config_rest_of_top100_step1700.json")

  # @test round2_query_cmd_maker(1) == "cd /home/ymh; source env/bin/activate; cd for_gvm_gpt; nohup python3 query_gpt.py --config configs/config_rest_of_top100_step1700.json --startidx 3000"
  # @test round2_query_cmd_maker(2) == "cd /home/ymh; source env/bin/activate; cd for_gvm_gpt; nohup python3 query_gpt.py --config configs/config_rest_of_top100_step1700.json --startidx 4700"

  map_idxed_cmd_over_tpus(round2_query_cmd_maker, tpu_nms) #optional args: zone, test_mode
end


