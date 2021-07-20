import functools
# import multiprocessing
# import subprocess

from tpu_utils import get_connection, create_tpu, wait_til, check_tpu, delete_tpu

n_tpus = 5
#
# for i in tpu_nms:
#     delete_tpu(f"swarm-jax-test-{i}", "europe-west4-a")
#
# exit()

zone = "us-central1-f" 
# "europe-west4-a"
tpu_prefix = "nxt300"
tpu_nms = [f"{tpu_prefix}_{i}" for i in range(n_tpus)]

for tpu in tpu_nms:
    create_tpu(f"{tpu}", zone, "v2-8", preemptible=False)

for tpu in tpu_nms:
    assert wait_til(f"{tpu}", zone, {'state': 'READY', 'health': 'HEALTHY'})

conns = []
for tpu in tpu_nms:
    conns += get_connection(f"{tpu}", zone)

with multiprocessing.Pool(processes=n_tpus) as p:
    p.map(functools.partial(start_ray, address=address), conns)


# begin scrawtch work

vm_setup_instructs = ["git clone https://github.com/ym-han/for_gvm_gpt.git",
                     "cd for_gvm_gpt/setup_vm/",
                     "chmod +x init_vm.sh",
                     "sh init_vm.sh"]
setup_instr_str = "; ".join(vm_setup_instructs)  

instruct_template = """gcloud alpha compute tpus tpu-vm ssh {tpu_name} --command='""" + f"""{setup_instr_str}'"""  


for tpu in tpu_nms:
    gcloud_instruct_str = instruct_template.format(tpu_name=tpu)
    print(f"gcloud_instruct_str is {gcloud_instruct_str}\n")
    subprocess.run(gcloud_instruct_str, shell=True, check=True, capture_output=True)




def start_ray(conn, address):
    conn.sudo('rm -rf *.py')
    conn.sudo('rm -rf swarm_jax')

    for i in glob.glob("*.py"):
        print(i)
        conn.put(i, "")

    conn.run("mkdir swarm_jax -p")

    for i in glob.glob("swarm_jax/*.py"):
        print(i)
        conn.put(i, "swarm_jax/")

    conn.sudo('python3 setup.py install')

    conn.put("scripts/init_ray.sh", "/tmp/ray-tpu.sh")
    print(conn.sudo('chmod +x /tmp/ray-tpu.sh'))
    print(conn.sudo('/tmp/ray-tpu.sh'))
    try:
        print(conn.run('ray stop -f'))
    except:
        pass
        print(conn.run(f"ray start --address={address} --load-code-from-local --resources='" + '{"tpu": 1}\''))


model = SwarmCharTransformerBig
swarm = Swarm(model, optimizer, 2 ** 16, train_dataset.get_samples, prec)
swarm.run(100000, "runs/512_30L", "ckpt/512_30L")

ray.shutdown()