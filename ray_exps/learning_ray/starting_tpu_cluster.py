# TESTING THINGS OUT

import ray
from fabric import Connection
import functools
import multiprocessing


zone_central = "us-central1-f"
n_tpus = 2

# for i in range(n_tpus):
#     delete_tpu(f"swarm-jax-test-{i}", "europe-west4-a")
#
# exit()

head_info = ray.init(dashboard_host="0.0.0.0")
address = head_info['redis_address']


tpu_nms = [f"us-p-{i}" for i in range(2)]

conns = []
for tpu_nm in tpu_nms:
    conns += get_connection(tpu_nm, zone_central)

# In [10]: conns
# Out[10]: [<Connection host=10.128.0.24>, <Connection host=10.128.0.22>]


with multiprocessing.Pool(processes=n_tpus) as p:
    p.map(functools.partial(start_ray, address=address), conns)


#

train_dataset = TextLoader("data/enwik9", batchsize=(8, 8), sample_size=1024, length=90000000)

optimizer = optax.chain(
    optax.clip_by_global_norm(0.25),
    optax.adam(2e-4, b1=0.9, b2=0.99, eps=1e-5))

prec = NetworkPrecision(fwd_act="float32", rev_act="float32", grad="float32")

model = SwarmCharTransformerBig
swarm = Swarm(model, optimizer, 2 ** 16, train_dataset.get_samples, prec)
swarm.run(100000, "runs/512_30L", "ckpt/512_30L")

ray.shutdown()
