# Cleaner version of code for running embarrasingly parallel jobs on TPUs
# TO DO: Add setup code (tho not sure if doing this is worth it)

using Test

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


function make_cmd(tpu_name, cmd_str)
  cmd = `gcloud alpha compute tpus tpu-vm ssh $tpu_name --command=$cmd_str`
end

function run_on_tpu(tpu_name, cmd_str) 
  cmd = make_cmd(tpu_name, cmd_str)
  println("calling gcloud with: $cmd")

  read(cmd, String)
end

make_str(lst) = join(lst, "; ")  

""" Note that the cmd str for this function is the same across all the tpus.
In particular, the cmd str here does not depend on tpu idx. """
function run_const_cmd_str_on_tpus(cmd_str, tpu_nms)
  println("running the following on tpus: $cmd_str")

  @time begin
   asyncmap(tpu->run_on_tpu(tpu, cmd_str), tpu_nms)
end

function tpunm_from_tpuidx(idx, tpu_prefix)
  # where idx is 1, 2, ... , 7
  py_idx = idx - 1
  tpu_nm = "$tpu_prefix" * "_$py_idx" 
end

@test tpunm_from_tpuidx(1, "tpu") == "tpu_0"
@test tpunm_from_tpuidx(2, "tpu") == "tpu_1"

# a cleaner way to do this would be to have this be a special case of 
# a idx_dep_cmd_from_tpuidx(idx, py_cmds) function
function query_cmd_from_tpuidx(idx, qdict_idx_range, config_fnm, py_cmds)
  start_i = qdict_idx_range[idx]
  py_cmds = ["cd $tpu_home", 
          "source env/bin/activate",
          "cd for_gvm_gpt",
          "nohup python3 query_gpt.py --config configs/$config_fnm --startidx $start_i"]

  make_str(py_cmds)
end

# where cmd_maker is ftn from idx to cmd_string
function run_query_cmd(idx, cmd_maker)
  tpu_nm = tpunm_from_tpuidx(idx, tpu_prefix)
  cmd_str = cmd_maker(idx)

  run_on_tpu(tpu_nm, cmd_str)
end

function map_over_tpus(cmd_maker, n_tpus)
  makeshift_jl_idxes = 1:n_tpus
  
  @time begin
     asyncmap(i -> run_query_cmd(i, cmd_maker), makeshift_jl_idxes)
  end
end

offset_start_idxes(; offset=0, step=500, n_tpus=n_tpus) = range(offset, step=step, length=n_tpus)

@test offset_start_idxes(n_tpus=5) == 0:500:2000
@test offset_start_idxes(n_tpus=7) == 0:500:3000
@test offset_start_idxes(offset=3, n_tpus=2) == 3:500:503


# Querying gpt

## Things that will need to be changed
const tpu_home = "/home/ymh"

n_tpus = 5
tpu_prefix = "tpu"

round2_query_cmd_maker(idx) = query_cmd_from_tpuidx(idx, round_2_qds_idxes, "config_rest_of_top100_step1700.json")

@test round2_query_cmd_maker(1) == "cd /home/ymh; source env/bin/activate; cd for_gvm_gpt; nohup python3 query_gpt.py --config configs/config_rest_of_top100_step1700.json --startidx 3000"
@test round2_query_cmd_maker(2) == "cd /home/ymh; source env/bin/activate; cd for_gvm_gpt; nohup python3 query_gpt.py --config configs/config_rest_of_top100_step1700.json --startidx 4700"

map_over_tpus(round2_query_cmd_maker, n_tpus)






