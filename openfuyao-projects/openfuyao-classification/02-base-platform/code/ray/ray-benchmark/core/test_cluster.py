import os
import pytest


@pytest.mark.benchmark(group="core_cluster")
def test_ray_cluster_start(benchmark):
    def run():
        os.system("ray start --head")
        os.system("ray stop")

    benchmark(run)
