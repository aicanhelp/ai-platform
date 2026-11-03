import ray
import pytest


class Actor:
    def __call__(self, batch):
        return batch


@pytest.mark.benchmark(group="data")
@pytest.mark.parametrize("dataset_size", [100, 1000])
def test_data_3_stage_map(benchmark, dataset_size):
    ray.init()

    ds = ray.data.from_items(range(dataset_size)).repartition(dataset_size)
    ds = ds.map_batches(Actor, concurrency=1, batch_size=5, num_cpus=1)
    ds = ds.map_batches(Actor, concurrency=1, batch_size=5, num_cpus=1)
    ds = ds.map_batches(Actor, concurrency=1, batch_size=5, num_cpus=1)

    benchmark(ds.materialize)
    # benchmark.extra_info['throughput'] = dataset_size / benchmark.
    ray.shutdown()
