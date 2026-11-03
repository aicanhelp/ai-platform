import time
import ray

DEFAULT_SLEEP_TIME = 1000
DEFAULT_ACTOR_NUMS = 10

ray.init()


@ray.remote
class Test:
    def call(self):
        time.sleep(DEFAULT_SLEEP_TIME)
        return "Done"


actors = [Test.remote() for _ in range(DEFAULT_ACTOR_NUMS)]
actor_refs = [actor.call.remote() for actor in actors]

results = ray.get(actor_refs)
