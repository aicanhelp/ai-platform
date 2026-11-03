import logging
import time

import aiohttp.web
import ray.dashboard.optional_utils as dashboard_optional_utils
import ray.dashboard.utils as dashboard_utils
from ray.autoscaler.v2.schema import Stats
from ray.autoscaler.v2.sdk import ClusterStatusParser
from ray.dashboard.modules.accelerator.accelerator_consts import accelerators
from ray.dashboard.subprocesses.module import SubprocessModule
from ray.dashboard.subprocesses.routes import SubprocessRouteTable as routes

logger = logging.getLogger(__name__)


class ResourcesHead(SubprocessModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @routes.get("/accelerator/get_accelerators")
    async def get_accelerators(self, req) -> aiohttp.web.Response:
        req_time = time.time()
        cluster_status = await self.gcs_client.async_get_cluster_status()
        reply_time = time.time()
        cluster_status = ClusterStatusParser.from_get_cluster_status_reply(
            cluster_status,
            stats=Stats(
                gcs_request_time_s=reply_time - req_time, request_ts_s=req_time
            ),
        )
        result = []
        for usage in cluster_status.cluster_resource_usage:
            if usage.resource_name in accelerators:
                result.append(accelerators[usage.resource_name])
        return dashboard_optional_utils.rest_response(
            status_code=dashboard_utils.HTTPStatusCode.OK,
            success=True,
            message="",
            result=result,
        )

    async def run(self):
        await super().run()
