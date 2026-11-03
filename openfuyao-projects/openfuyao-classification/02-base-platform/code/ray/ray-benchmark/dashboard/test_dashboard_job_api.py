import os
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from ray.cluster_utils import Cluster


@pytest.fixture
def ray_cluster_with_running_jobs(request):
    NODE_NUMS = 3
    running_job_num = request.param["running_job_num"]

    cluster = Cluster(
        initialize_head=True,
        head_node_args={
            "num_cpus": 100,
            "dashboard_host": "0.0.0.0",
        },
    )

    for _ in range(NODE_NUMS - 1):
        cluster.add_node(num_cpus=100)

    def submit_job():
        # TODO: how to remove the hard code here?
        current_dir = os.path.dirname(os.path.abspath(__file__))
        job_script_absolute_path = current_dir + "/job_example.py"

        os.system(f"ray job submit --no-wait -- python {job_script_absolute_path}")

        # submit_url = f'http://{dashboard_address}/api/jobs'

        # submit_info = {
        #     "entrypoint": f"python {job_script_path}",
        #     "runtime_env": {}
        # }
        # requests.post(submit_url, json=submit_info)

    with ThreadPoolExecutor(max_workers=192) as executor:
        for _ in range(running_job_num):
            executor.submit(submit_job)

    yield cluster

    cluster.shutdown()


@pytest.mark.benchmark(group="dashboard", warmup=False)
@pytest.mark.parametrize(
    "ray_cluster_with_running_jobs",
    [
        {"running_job_num": 10},
        {"running_job_num": 100},
        # {"running_job_nums": 500},
    ],
    indirect=True,
)
def test_dashboard_get_all_job_api_delay(ray_cluster_with_running_jobs, benchmark):
    head_node = ray_cluster_with_running_jobs.head_node
    dashboard_address = head_node.address_info["webui_url"]

    def call_get_job_api():
        requests.get(f"http://{dashboard_address}/api/jobs/")
        # assert len(json.loads(resp.text)) == running_job_num
        # print(resp.text)

    benchmark(call_get_job_api)


@pytest.mark.benchmark(group="dashboard", warmup=False)
@pytest.mark.parametrize(
    "ray_cluster_with_running_jobs",
    [
        {"running_job_num": 10},
        {"running_job_num": 100},
        # {"running_job_nums": 500},
    ],
    indirect=True,
)
def test_dashboard_ray_submit_job_delay(ray_cluster_with_running_jobs, benchmark):
    def submit_job():
        # TODO: how to remove the hard code here?
        current_dir = os.path.dirname(os.path.abspath(__file__))
        job_script_absolute_path = current_dir + "/job_example.py"

        os.system(f"ray job submit --no-wait -- python {job_script_absolute_path}")

    # 但是函数 `call_post_job_api` 不是幂等的，靠后的 call 集群中已有的 job 会更多一些
    benchmark(submit_job)
