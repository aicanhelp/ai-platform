import pytest
import ray


@ray.remote
class Actor:
    def call(self):
        return 1


@pytest.mark.benchmark(group="core_actor")
def test_actor_call(benchmark):
    ray.init()
    actor = Actor.remote()
    # ensure actor creation is finished.
    ray.get(actor.call.remote())

    def call_sync():
        ray.get(actor.call.remote())

    benchmark(call_sync)

    ray.shutdown()


@pytest.mark.benchmark(group="core_actor")
def test_actor_call_create(benchmark):
    ray.init()

    def call_sync():
        actor = Actor.remote()
        ray.get(actor.call.remote())

    benchmark(call_sync)

    ray.shutdown()
