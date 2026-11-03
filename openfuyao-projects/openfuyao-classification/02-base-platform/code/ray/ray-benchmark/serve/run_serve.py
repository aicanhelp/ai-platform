import ray
from ray import serve
from starlette.requests import Request


num_replicas = 2

ray.init()


@serve.deployment(num_replicas=num_replicas)
class DeploymentExample:
    async def __call__(self, request: Request):
        return "hi"


app = DeploymentExample.bind()
serve.run(app)
