import asyncio
import copy
import dataclasses
import enum
import json
import logging
import os
import re
import requests
import time
import traceback
from collections import defaultdict
from datetime import datetime
from random import choice
from typing import AsyncIterator, Dict, List, Optional, Tuple

import aiohttp.web
import ray.dashboard.consts as dashboard_consts
from aiohttp.client import ClientResponse
from aiohttp.web import Request, Response, StreamResponse

import ray
from ray import NodeID
from ray._common.utils import get_or_create_event_loop, load_class
from ray._common.pydantic_compat import BaseModel, Extra, Field, validator
from ray._common.utils import (
    hex_to_binary,
    binary_to_hex,
)
from ray._private.gcs_utils import GcsChannel
from ray._private.ray_constants import (
    KV_NAMESPACE_DASHBOARD,
    env_bool,
    RAY_INTERNAL_NAMESPACE_PREFIX,
)
from ray.dashboard.consts import (
    GCS_RPC_TIMEOUT_SECONDS,
    DASHBOARD_AGENT_ADDR_NODE_ID_PREFIX,
    RAY_CLUSTER_ACTIVITY_HOOK,
    TRY_TO_GET_AGENT_INFO_INTERVAL_SECONDS,
    WAIT_AVAILABLE_AGENT_TIMEOUT,
)
from ray._private.runtime_env.packaging import (
    package_exists,
    pin_runtime_env_uri,
    upload_package_to_gcs,
)
from ray.core.generated.gcs_service_pb2 import DeleteJobInfoRequest
from ray.core.generated.gcs_service_pb2_grpc import JobInfoGcsServiceStub
from ray.core.generated.common_pb2 import TaskStatus
from ray.dashboard.modules.job.common import (
    JOB_ID_METADATA_KEY,
    JobDeleteResponse,
    JobInfoStorageClient,
    JobLogsResponse,
    JobStopResponse,
    JobSubmitRequest,
    JobSubmitResponse,
    http_uri_components_to_uri,
)
from ray.dashboard.modules.job.pydantic_models import (
    JobDetails,
    JobType,
    InstResUsage,
    JobResUsage,
)
from ray.dashboard.modules.job.utils import (
    find_job_by_ids,
    get_job_driver_list,
    get_head_node_id,
    parse_and_validate_request,
)
from ray.dashboard.modules.version import CURRENT_VERSION, VersionResponse
from ray.dashboard.subprocesses.module import SubprocessModule
from ray.dashboard.subprocesses.routes import SubprocessRouteTable as routes
from ray.dashboard.subprocesses.utils import ResponseType

from ray import JobID
from ray.dashboard.resource_profile import get_all_classes_profile_results
from ray.dashboard.utils import ray_address_to_api_server_url
from ray.data._internal.stats import STATS_ACTOR_NAME
from ray.util.state.common import protobuf_message_to_dict
from ray.util.state.state_manager import StateDataSourceClient

from ray.core.generated.gcs_pb2 import ActorTableData

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Feature flag controlling whether critical Ray Job control operations are performed
# exclusively by the Job Agent running on the Head node (or randomly sampled Worker one)
#
# NOTE: This flag serves as a temporary kill-switch and should be eventually cleaned up
RAY_JOB_AGENT_USE_HEAD_NODE_ONLY = env_bool("RAY_JOB_AGENT_USE_HEAD_NODE_ONLY", True)


class RayActivityStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    ERROR = "ERROR"


class RayActivityResponse(BaseModel, extra=Extra.allow):
    """
    Pydantic model used to inform if a particular Ray component can be considered
    active, and metadata about observation.
    """

    is_active: RayActivityStatus = Field(
        ...,
        description=(
            "Whether the corresponding Ray component is considered active or inactive, "
            "or if there was an error while collecting this observation."
        ),
    )
    reason: Optional[str] = Field(
        None, description="Reason if Ray component is considered active or errored."
    )
    timestamp: float = Field(
        ...,
        description=(
            "Timestamp of when this observation about the Ray component was made. "
            "This is in the format of seconds since unix epoch."
        ),
    )
    last_activity_at: Optional[float] = Field(
        None,
        description=(
            "Timestamp when last actvity of this Ray component finished in format of "
            "seconds since unix epoch. This field does not need to be populated "
            "for Ray components where it is not meaningful."
        ),
    )

    @validator("reason", always=True)
    def reason_required(cls, v, values, **kwargs):
        if "is_active" in values and values["is_active"] != RayActivityStatus.INACTIVE:
            if v is None:
                raise ValueError(
                    'Reason is required if is_active is "active" or "error"'
                )
        return v


class JobAgentSubmissionClient:
    """A local client for submitting and interacting with jobs on a specific node
    in the remote cluster.
    Submits requests over HTTP to the job agent on the specific node using the REST API.
    """

    def __init__(
        self,
        dashboard_agent_address: str,
    ):
        self._agent_address = dashboard_agent_address
        self._session = aiohttp.ClientSession()

    async def _raise_error(self, resp: ClientResponse):
        status = resp.status
        error_text = await resp.text()
        raise RuntimeError(f"Request failed with status code {status}: {error_text}.")

    async def submit_job_internal(self, req: JobSubmitRequest) -> JobSubmitResponse:
        logger.debug(f"Submitting job with submission_id={req.submission_id}.")

        async with self._session.post(
            f"{self._agent_address}/api/job_agent/jobs/", json=dataclasses.asdict(req)
        ) as resp:
            if resp.status == 200:
                result_json = await resp.json()
                return JobSubmitResponse(**result_json)
            else:
                await self._raise_error(resp)

    async def stop_job_internal(self, job_id: str) -> JobStopResponse:
        logger.debug(f"Stopping job with job_id={job_id}.")

        async with self._session.post(
            f"{self._agent_address}/api/job_agent/jobs/{job_id}/stop"
        ) as resp:
            if resp.status == 200:
                result_json = await resp.json()
                return JobStopResponse(**result_json)
            else:
                await self._raise_error(resp)

    async def delete_job_internal(self, job_id: str) -> JobDeleteResponse:
        logger.debug(f"Deleting job with job_id={job_id}.")

        async with self._session.delete(
            f"{self._agent_address}/api/job_agent/jobs/{job_id}"
        ) as resp:
            if resp.status == 200:
                result_json = await resp.json()
                return JobDeleteResponse(**result_json)
            else:
                await self._raise_error(resp)

    async def get_job_logs_internal(self, job_id: str) -> JobLogsResponse:
        async with self._session.get(
            f"{self._agent_address}/api/job_agent/jobs/{job_id}/logs"
        ) as resp:
            if resp.status == 200:
                result_json = await resp.json()
                return JobLogsResponse(**result_json)
            else:
                await self._raise_error(resp)

    async def delete_job_logs_internal(
        self, job_or_submission_id: str, job_type: str, driver_pid=None
    ) -> str:
        url = (
            f"{self._agent_address}/api/job_agent/jobs/{job_or_submission_id}/logs"
            f"?job_type={job_type}"
        )
        if driver_pid:
            url += f"&driver_pid={driver_pid}"
        async with self._session.delete(url=url) as resp:
            if resp.status == aiohttp.web.HTTPOk.status_code:
                return "SUCCESS"
            elif resp.status == aiohttp.web.HTTPNotFound.status_code:
                return "NOT_FOUND"
            else:
                await self._raise_error(resp)

    async def tail_job_logs(self, job_id: str) -> AsyncIterator[str]:
        """Get an iterator that follows the logs of a job."""
        ws = await self._session.ws_connect(
            f"{self._agent_address}/api/job_agent/jobs/{job_id}/logs/tail"
        )

        while True:
            msg = await ws.receive()

            if msg.type == aiohttp.WSMsgType.TEXT:
                yield msg.data
            elif msg.type == aiohttp.WSMsgType.CLOSED:
                break
            elif msg.type == aiohttp.WSMsgType.ERROR:
                pass

    async def add_profile_job(self, submission_id: str) -> str:
        async with self._session.post(
            f"{self._agent_address}/api/job_agent/jobs/{submission_id}/profile"
        ) as resp:
            if resp.status == 200:
                result_text = await resp.text()
                return result_text
            else:
                return "Failed"

    async def close(self, ignore_error=True):
        try:
            await self._session.close()
        except Exception:
            if not ignore_error:
                raise


class JobHead(SubprocessModule):
    """Runs on the head node of a Ray cluster and handles Ray Jobs APIs.

    NOTE(architkulkarni): Please keep this class in sync with the OpenAPI spec at
    `doc/source/cluster/running-applications/job-submission/openapi.yml`.
    We currently do not automatically check that the OpenAPI
    spec is in sync with the implementation. If any changes are made to the
    paths in the @route decorators or in the Responses returned by the
    methods (or any nested fields in the Responses), you will need to find the
    corresponding field of the OpenAPI yaml file and update it manually. Also,
    bump the version number in the yaml file and in this class's `get_version`.
    """

    # Time that we sleep while tailing logs while waiting for
    # the supervisor actor to start. We don't know which node
    # to read the logs from until then.
    WAIT_FOR_SUPERVISOR_ACTOR_INTERVAL_S = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._job_info_client = None

        # To make sure that the internal KV is initialized by getting the lazy property
        assert self.gcs_client is not None
        assert ray.experimental.internal_kv._internal_kv_initialized()

        # It contains all `JobAgentSubmissionClient` that
        # `JobHead` has ever used, and will not be deleted
        # from it unless `JobAgentSubmissionClient` is no
        # longer available (the corresponding agent process is dead)
        # {node_id: JobAgentSubmissionClient}
        self._agents: Dict[NodeID, JobAgentSubmissionClient] = dict()
        self._gcs_channel = GcsChannel(gcs_address=self.gcs_address, aio=True)
        self._gcs_channel.connect()
        self._gcs_job_info_stub = JobInfoGcsServiceStub(self._gcs_channel.channel())
        self._job_id_patt = re.compile(r"^[\da-f]{8}$")
        self._state_client = StateDataSourceClient(
            gcs_channel=self._gcs_channel.channel(), gcs_client=self._gcs_client
        )
        self._profile_job_num = 0
        self._job_classes_profile_results = {}

    async def get_target_agent(
        self, timeout_s: float = WAIT_AVAILABLE_AGENT_TIMEOUT
    ) -> JobAgentSubmissionClient:
        """
        Get a `JobAgentSubmissionClient`, which is a client for interacting with jobs
        via an agent process.

        Args:
            timeout_s: The timeout for the operation.

        Returns:
            A `JobAgentSubmissionClient` for interacting with jobs via an agent process.

        Raises:
            TimeoutError: If the operation times out.
        """
        if RAY_JOB_AGENT_USE_HEAD_NODE_ONLY:
            return await self._get_head_node_agent(timeout_s)

        return await self._pick_random_agent(timeout_s)

    async def _pick_random_agent(
        self, timeout_s: float
    ) -> Optional[JobAgentSubmissionClient]:
        """
        Try to disperse as much as possible to select one of
        the `CANDIDATE_AGENT_NUMBER` agents to solve requests.
        the agents will not pop from `self._agents` unless
        it's dead. Saved in `self._agents` is the agent that was
        used before.
        Strategy:
            1. if the number of `self._agents` has reached
               `CANDIDATE_AGENT_NUMBER`, randomly select one agent from
               `self._agents`.
            2. if not, randomly select one agent from all available agents,
               it is possible that the selected one already exists in
               `self._agents`.

        If there's no agent available at all, or there's exception, it will retry every
        `TRY_TO_GET_AGENT_INFO_INTERVAL_SECONDS` seconds indefinitely.

        Args:
            timeout_s: The timeout for the operation.

        Returns:
            A `JobAgentSubmissionClient` for interacting with jobs via an agent process.

        Raises:
            TimeoutError: If the operation times out.
        """
        start_time_s = time.time()
        last_exception = None
        while time.time() < start_time_s + timeout_s:
            try:
                return await self._pick_random_agent_once()
            except Exception as e:
                last_exception = e
                logger.exception(
                    f"Failed to pick a random agent, retrying in {TRY_TO_GET_AGENT_INFO_INTERVAL_SECONDS} seconds..."
                )
                await asyncio.sleep(TRY_TO_GET_AGENT_INFO_INTERVAL_SECONDS)
        raise TimeoutError(
            f"Failed to pick a random agent within {timeout_s} seconds. The last exception is {last_exception}"
        )

    async def _pick_random_agent_once(self) -> JobAgentSubmissionClient:
        """
        Query the internal kv for all agent infos, and pick agents randomly. May raise
        exception if there's no agent available at all or there's network error.
        """
        # NOTE: Following call will block until there's at least 1 agent info
        #       being populated from GCS
        agent_node_ids = await self._fetch_all_agent_node_ids()

        # delete dead agents.
        for dead_node in set(self._agents) - set(agent_node_ids):
            client = self._agents.pop(dead_node)
            await client.close()

        if len(self._agents) >= dashboard_consts.CANDIDATE_AGENT_NUMBER:
            node_id = choice(list(self._agents))
            return self._agents[node_id]
        else:
            # Randomly select one from among all agents, it is possible that
            # the selected one already exists in `self._agents`
            node_id = choice(list(agent_node_ids))

            if node_id not in self._agents:
                # Fetch agent info from InternalKV, and create a new
                # JobAgentSubmissionClient. May raise if the node_id is removed in
                # InternalKV after the _fetch_all_agent_node_ids, though unlikely.
                ip, http_port, _ = await self._fetch_agent_info(node_id)
                agent_http_address = f"http://{ip}:{http_port}"
                self._agents[node_id] = JobAgentSubmissionClient(agent_http_address)

            return self._agents[node_id]

    async def _get_head_node_agent_once(self) -> JobAgentSubmissionClient:
        head_node_id_hex = await get_head_node_id(self.gcs_client)

        if not head_node_id_hex:
            raise Exception("Head node id has not yet been persisted in GCS")

        head_node_id = NodeID.from_hex(head_node_id_hex)

        if head_node_id not in self._agents:
            ip, http_port, _ = await self._fetch_agent_info(head_node_id)
            agent_http_address = f"http://{ip}:{http_port}"
            self._agents[head_node_id] = JobAgentSubmissionClient(agent_http_address)

        return self._agents[head_node_id]

    async def _get_head_node_agent(self, timeout_s: float) -> JobAgentSubmissionClient:
        """Retrieves HTTP client for `JobAgent` running on the Head node. If the head
        node does not have an agent, it will retry every
        `TRY_TO_GET_AGENT_INFO_INTERVAL_SECONDS` seconds indefinitely.

        Args:
            timeout_s: The timeout for the operation.

        Returns:
            A `JobAgentSubmissionClient` for interacting with jobs via the head node's agent process.

        Raises:
            TimeoutError: If the operation times out.
        """
        timeout_point = time.time() + timeout_s
        exception = None
        while time.time() < timeout_point:
            try:
                return await self._get_head_node_agent_once()
            except Exception as e:
                exception = e
                logger.exception(
                    f"Failed to get head node agent, retrying in {TRY_TO_GET_AGENT_INFO_INTERVAL_SECONDS} seconds..."
                )
                await asyncio.sleep(TRY_TO_GET_AGENT_INFO_INTERVAL_SECONDS)
        raise TimeoutError(
            f"Failed to get head node agent within {timeout_s} seconds. The last exception is {exception}"
        )

    async def _fetch_all_agent_node_ids(self) -> List[NodeID]:
        """
        Fetches all NodeIDs with agent infos in the cluster.

        May raise exception if there's no agent available at all or there's network error.
        Returns: List[NodeID]
        """
        keys = await self.gcs_client.async_internal_kv_keys(
            f"{DASHBOARD_AGENT_ADDR_NODE_ID_PREFIX}".encode(),
            namespace=KV_NAMESPACE_DASHBOARD,
            timeout=GCS_RPC_TIMEOUT_SECONDS,
        )
        if not keys:
            # No agent keys found, retry
            raise Exception("No agents found in InternalKV.")
        return [
            NodeID.from_hex(key[len(DASHBOARD_AGENT_ADDR_NODE_ID_PREFIX) :].decode())
            for key in keys
        ]

    async def _fetch_agent_info(self, target_node_id: NodeID) -> Tuple[str, int, int]:
        """
        Fetches agent info by the Node ID. May raise exception if there's network error or the
        agent info is not found.

        Returns: (ip, http_port, grpc_port)
        """
        key = f"{DASHBOARD_AGENT_ADDR_NODE_ID_PREFIX}{target_node_id.hex()}"
        value = await self.gcs_client.async_internal_kv_get(
            key,
            namespace=KV_NAMESPACE_DASHBOARD,
            timeout=GCS_RPC_TIMEOUT_SECONDS,
        )
        if not value:
            raise KeyError(
                f"Agent info not found in internal KV for node {target_node_id}. "
                "It's possible that the agent didn't launch successfully due to "
                "port conflicts or other issues. Please check `dashboard_agent.log` "
                "for more details."
            )
        return json.loads(value.decode())

    @routes.get("/api/version")
    async def get_version(self, req: Request) -> Response:
        # NOTE(edoakes): CURRENT_VERSION should be bumped and checked on the
        # client when we have backwards-incompatible changes.
        resp = VersionResponse(
            version=CURRENT_VERSION,
            ray_version=ray.__version__,
            ray_commit=ray.__commit__,
            package_time=ray.__package_time__,
            session_name=self.session_name,
        )
        return Response(
            text=json.dumps(dataclasses.asdict(resp)),
            content_type="application/json",
            status=aiohttp.web.HTTPOk.status_code,
        )

    @routes.get("/api/packages/{protocol}/{package_name}")
    async def get_package(self, req: Request) -> Response:
        package_uri = http_uri_components_to_uri(
            protocol=req.match_info["protocol"],
            package_name=req.match_info["package_name"],
        )

        logger.debug(f"Adding temporary reference to package {package_uri}.")
        try:
            pin_runtime_env_uri(package_uri)
        except Exception:
            return Response(
                text=traceback.format_exc(),
                status=aiohttp.web.HTTPInternalServerError.status_code,
            )

        if not package_exists(package_uri):
            return Response(
                text=f"Package {package_uri} does not exist",
                status=aiohttp.web.HTTPNotFound.status_code,
            )

        return Response()

    @routes.put("/api/packages/{protocol}/{package_name}")
    async def upload_package(self, req: Request):
        package_uri = http_uri_components_to_uri(
            protocol=req.match_info["protocol"],
            package_name=req.match_info["package_name"],
        )
        logger.info(f"Uploading package {package_uri} to the GCS.")
        try:
            data = await req.read()
            await get_or_create_event_loop().run_in_executor(
                None,
                upload_package_to_gcs,
                package_uri,
                data,
            )
        except Exception:
            return Response(
                text=traceback.format_exc(),
                status=aiohttp.web.HTTPInternalServerError.status_code,
            )

        return Response(status=aiohttp.web.HTTPOk.status_code)

    @routes.post("/api/jobs/")
    async def submit_job(self, req: Request) -> Response:
        result = await parse_and_validate_request(req, JobSubmitRequest)
        # Request parsing failed, returned with Response object.
        if isinstance(result, Response):
            return result
        else:
            submit_request: JobSubmitRequest = result

        try:
            job_agent_client = await self.get_target_agent()
            resp = await job_agent_client.submit_job_internal(submit_request)
        except asyncio.TimeoutError:
            return Response(
                text="No available agent to submit job, please try again later.",
                status=aiohttp.web.HTTPInternalServerError.status_code,
            )
        except (TypeError, ValueError):
            return Response(
                text=traceback.format_exc(),
                status=aiohttp.web.HTTPBadRequest.status_code,
            )
        except Exception:
            return Response(
                text=traceback.format_exc(),
                status=aiohttp.web.HTTPInternalServerError.status_code,
            )

        return Response(
            text=json.dumps(dataclasses.asdict(resp)),
            content_type="application/json",
            status=aiohttp.web.HTTPOk.status_code,
        )

    @routes.post("/api/jobs/{job_or_submission_id}/stop")
    async def stop_job(self, req: Request) -> Response:
        job_or_submission_id = req.match_info["job_or_submission_id"]
        if self._job_id_patt.match(job_or_submission_id):
            # 若是能在 job_table 里找到，则说明JobType是DRIVER，此API无法处理。
            if await self.gcs_client.async_get_job_info(job_or_submission_id):
                return Response(
                    text="Can only stop submission type jobs",
                    status=aiohttp.web.HTTPBadRequest.status_code,
                )

        if await self._job_info_client.get_info(job_or_submission_id):
            try:
                job_agent_client = await asyncio.wait_for(
                    self.get_target_agent(),
                    timeout=WAIT_AVAILABLE_AGENT_TIMEOUT,
                )
                resp = await job_agent_client.stop_job_internal(job_or_submission_id)
            except Exception:
                return Response(
                    text=traceback.format_exc(),
                    status=aiohttp.web.HTTPInternalServerError.status_code,
                )
            return Response(
                text=json.dumps(dataclasses.asdict(resp)),
                content_type="application/json",
            )
        else:
            return Response(
                text=f"Job {job_or_submission_id} does not exist",
                status=aiohttp.web.HTTPNotFound.status_code,
            )

    async def delete_from_job_table(self, job_id: str):
        match = re.match(r"^[\da-f]{8}$", job_id)
        if not match:
            return False
        request = DeleteJobInfoRequest()
        request.job_id = JobID(hex_to_binary(job_id)).binary()
        reply = await self._gcs_job_info_stub.DeleteJobInfo(request)
        if reply.status.code == 0:
            return True
        elif reply.status.code == 17:
            return False
        elif reply.status.code == 4 and reply.status.message == "RUNNING":
            raise RuntimeError(
                f"Attempted to delete job '{job_id}', "
                f"but it is in a non-terminal state RUNNING."
            )
        else:
            raise Exception(
                f"Delete Job Failed. Code={reply.status.code} "
                f"Message={reply.status.message}"
            )

    @routes.delete("/api/jobs/{job_or_submission_id}")
    async def delete_job(self, req: Request) -> Response:
        job_or_submission_id = req.match_info["job_or_submission_id"]
        try:
            deleted = await self.delete_from_job_table(job_or_submission_id)
            if deleted:
                return Response(
                    text=json.dumps({"deleted": True}), content_type="application/json"
                )
            submission_id = job_or_submission_id
            job = await self._job_info_client.get_info(submission_id)
            if job:
                if job.status is None or not job.status.is_terminal():
                    raise RuntimeError(
                        f"Attempted to delete job '{submission_id}', "
                        f"but it is in a non-terminal state {job.status}."
                    )
                job_infos = await self._gcs_client.async_get_all_job_info(
                    job_or_submission_id=submission_id
                )
                for job_id, job_item in job_infos.items():
                    await self.delete_from_job_table(job_id.hex())
                await self._job_info_client.delete_info(submission_id)
                return Response(
                    text=json.dumps({"deleted": True}), content_type="application/json"
                )
            else:
                return Response(
                    text=f"Job {submission_id} does not exit",
                    status=aiohttp.web.HTTPNotFound.status_code,
                )
        except Exception:
            return Response(
                text=traceback.format_exc(),
                status=aiohttp.web.HTTPInternalServerError.status_code,
            )

    @routes.get("/api/jobs/{job_or_submission_id}")
    async def get_job_info(self, req: Request) -> Response:
        job_or_submission_id = req.match_info["job_or_submission_id"]
        job = await find_job_by_ids(
            self.gcs_client,
            self._job_info_client,
            job_or_submission_id,
        )
        if not job:
            return Response(
                text=f"Job {job_or_submission_id} does not exist",
                status=aiohttp.web.HTTPNotFound.status_code,
            )

        job.resource_usage = JobResUsage()
        job.footprint = dict()
        if job.type is JobType.DRIVER and job.status.is_terminal():
            await self.get_resource_stat(job.job_id, job.resource_usage, job.footprint)
        elif job.type is JobType.SUBMISSION and job.status.is_terminal():
            job_infos = await self._gcs_client.async_get_all_job_info()
            for job_id, job_item in job_infos.items():
                if job_item.config.ray_namespace.startswith(
                    RAY_INTERNAL_NAMESPACE_PREFIX
                ):
                    continue
                metadata = job_item.config.metadata
                job_submission_id = metadata.get(JOB_ID_METADATA_KEY)
                if job_submission_id and job_submission_id == job.submission_id:
                    await self.get_resource_stat(
                        job_item.job_id.hex(), job.resource_usage, job.footprint
                    )

        return Response(
            text=json.dumps(job.dict()),
            content_type="application/json",
        )

    @routes.post("/api/jobs/{job_id}/{submission_id}/profile")
    async def add_profile_job(self, req: Request) -> Response:
        job_id = req.match_info["job_id"]
        self._profile_job_num += 1
        self._job_classes_profile_results[job_id] = None

        submission_id = req.match_info["submission_id"]
        for _, job_agent_client in self._agents.items():
            resp = await job_agent_client.add_profile_job(submission_id)
            if resp == "Failed":
                return Response(
                    text="Add job profile failed",
                    status=aiohttp.web.HTTPBadRequest.status_code,
                )
        return Response(
            text="Add profile job successfully",
            status=aiohttp.web.HTTPOk.status_code,
        )

    async def get_resource_stat(self, job_id, resource_usage, footprint):
        def add_item_to_res_usage(
            resource_usage, footprint, name, type, dur, node_ip, CPU=0, **kwargs
        ):
            if name not in resource_usage.reprs:
                resource_usage.reprs[name] = InstResUsage(name=name, type=type)
            resource_usage.total.instNum += 1
            resource_usage.total.instTime += dur
            resource_usage.total.cpuTime += CPU * dur

            resource_usage.reprs[name].instNum += 1
            resource_usage.reprs[name].instTime += dur
            resource_usage.reprs[name].cpuTime += CPU * dur

            for key, val in kwargs.items():
                resource_usage.total.accelerators[key] += val * dur
                resource_usage.reprs[name].accelerators[key] += val * dur
            if node_ip not in footprint:
                footprint[node_ip] = defaultdict(int)
            footprint[node_ip][name] = 1

        node_physical_stats = requests.get(
            f"http://localhost:{self.http_port}/ds/node_physical_stats"
        ).json()["data"]["detail"]
        profile_data = {}
        profile = False
        if (
            job_id in self._job_classes_profile_results
            and self._job_classes_profile_results[job_id] is None
        ):
            profile = True

        actor_reply = await self._state_client.get_all_actor_info(
            limit=5000, filters=[("job_id", "=", job_id), ("state", "=", "DEAD")]
        )
        for actor in actor_reply.actor_table_data:
            actor_name = actor.repr_name or actor.name or actor.class_name
            start_time_ms = actor.start_time
            end_time_ms = actor.end_time if actor.end_time != 0 else time.time() * 1000
            duration = (end_time_ms - start_time_ms) / 1000
            add_item_to_res_usage(
                resource_usage,
                footprint,
                name=actor_name,
                type="ACTOR",
                dur=duration,
                node_ip=actor.address.ip_address,
                **actor.required_resources,
            )
            if profile:
                class_profile_data = profile_data.setdefault(actor_name, [])
                node_id = binary_to_hex(actor.address.raylet_id)
                actor_profile_data = dict(
                    node_physical_stats[node_id]
                    .get("profile_data", {})
                    .get(actor.pid, {})
                )
                if len(actor_profile_data) > 0:
                    actor_profile_data["requiredResources"] = {
                        **actor.required_resources
                    }
                    class_profile_data.append(actor_profile_data)
            await asyncio.sleep(0)

        task_reply = await self._state_client.get_all_task_info(
            limit=10000, filters=[("job_id", "=", job_id), ("type", "=", "NORMAL_TASK")]
        )
        for task in task_reply.events_by_task:
            task_state = task.state_updates.state_ts_ns
            task_info = task.task_info
            node_ip = task.profile_events.node_ip_address
            if (TaskStatus.FINISHED not in task_state) and (
                TaskStatus.FAILED not in task_state
            ):
                continue
            start_time_ms = task_state[TaskStatus.RUNNING] / 1e6
            end_time_ms = time.time() * 1000
            if TaskStatus.FINISHED in task_state:
                end_time_ms = task_state[TaskStatus.FINISHED] / 1e6
            if TaskStatus.FAILED in task_state:
                end_time_ms = task_state[TaskStatus.FAILED] / 1e6
            duration = (end_time_ms - start_time_ms) / 1000
            add_item_to_res_usage(
                resource_usage,
                footprint,
                name=task_info.name,
                type="NORMAL_TASK",
                dur=duration,
                node_ip=node_ip,
                **task_info.required_resources,
            )
            if profile:
                class_profile_data = profile_data.setdefault(task_info.name, [])
                node_id = binary_to_hex(task.state_updates.node_id)
                task_profile_data = dict(
                    node_physical_stats[node_id]
                    .get("profile_data", {})
                    .get(task.state_updates.worker_pid, {})
                )
                if len(task_profile_data) > 0:
                    task_profile_data["requiredResources"] = {
                        **task_info.required_resources
                    }
                    class_profile_data.append(task_profile_data)
            await asyncio.sleep(0)
        if profile and len(profile_data) > 0:
            classes_profile_results = get_all_classes_profile_results(profile_data)
            self._profile_job_num -= 1
            self._job_classes_profile_results[job_id] = copy.deepcopy(
                classes_profile_results
            )
            if self._profile_job_num <= 0:
                self._profile_job_num = 0
                api_server_url = ray_address_to_api_server_url("auto")
                request_url = api_server_url + "/profile/delete"
                _ = await aiohttp.ClientSession().post(request_url)
            for class_name, profile_result in classes_profile_results.items():
                resource_usage.reprs[class_name].profileResult = profile_result
        elif self._job_classes_profile_results.get(job_id, None):
            classes_profile_results = self._job_classes_profile_results[job_id]
            for class_name, profile_result in classes_profile_results.items():
                resource_usage.reprs[class_name].profileResult = profile_result

    @routes.get("/api/jobs/footprint")
    async def get_jobs_footprint(self, req: Request) -> Response:
        entrypoint_prefix = req.query.get("entrypoint_prefix")
        if entrypoint_prefix is None or entrypoint_prefix == "":
            return Response(text="", status=aiohttp.web.HTTPBadRequest.status_code)

        job_id_filters = []
        job_reply = await self._gcs_client.async_get_all_job_info()
        for job_id, job_item in job_reply.items():
            if job_item.entrypoint is not None and job_item.entrypoint.startswith(
                entrypoint_prefix
            ):
                job_id_filters.append(("job_id", "=", job_id.hex()))

        def update_footprint(footprint, node_ip, job_id, inst_name, cost_time):
            if node_ip not in footprint:
                footprint[node_ip] = {}
            if job_id not in footprint[node_ip]:
                footprint[node_ip][job_id] = defaultdict(float)
            footprint[node_ip][job_id][inst_name] += 1

        if len(job_id_filters) == 0:
            return Response(text=json.dumps({}), content_type="application/json")
        footprint = {}
        actor_reply = await self._state_client.get_all_actor_info(
            limit=5000, filters=job_id_filters
        )
        for actor in actor_reply.actor_table_data:
            if (
                actor.state == ActorTableData.ActorState.PENDING_CREATION
                or actor.start_time == 0
            ):
                continue
            if actor.is_detached and actor.name in {
                STATS_ACTOR_NAME,
                "AutoscalingRequester",
            }:
                continue
            node_ip = actor.address.ip_address
            job_id = actor.job_id.hex()
            actor_name = actor.repr_name or actor.name or actor.class_name
            end_time_ms = actor.end_time if actor.end_time != 0 else time.time() * 1000
            cost_time = (end_time_ms - actor.start_time) / 1000
            update_footprint(footprint, node_ip, job_id, actor_name, cost_time)

        task_reply = await self._state_client.get_all_task_info(
            limit=10000, filters=[("type", "=", "NORMAL_TASK")] + job_id_filters
        )
        for task in task_reply.events_by_task:
            if TaskStatus.RUNNING not in task.state_updates.state_ts_ns:
                continue
            node_ip = task.profile_events.node_ip_address
            job_id = task.task_info.job_id.hex()
            task_state = task.state_updates.state_ts_ns
            task_name = task.task_info.name
            start_time_ms = task_state[TaskStatus.RUNNING] / 1e6
            end_time_ms = time.time() * 1000
            if TaskStatus.FINISHED in task_state:
                end_time_ms = task_state[TaskStatus.FINISHED] / 1e6
            if TaskStatus.FAILED in task_state:
                end_time_ms = task_state[TaskStatus.FAILED] / 1e6
            cost_time = end_time_ms - start_time_ms
            update_footprint(footprint, node_ip, job_id, task_name, cost_time)
        return Response(text=json.dumps(footprint), content_type="application/json")

    # TODO(rickyx): This endpoint's logic is also mirrored in state API's endpoint.
    # We should eventually unify the backend logic (and keep the logic in sync before
    # that).
    @routes.get("/api/jobs/")
    async def list_jobs(self, req: Request) -> Response:
        (driver_jobs, submission_job_drivers), submission_jobs = await asyncio.gather(
            get_job_driver_list(self.gcs_client),
            self._job_info_client.get_all_jobs(),
        )

        submission_jobs = [
            JobDetails(
                **dataclasses.asdict(job),
                submission_id=submission_id,
                job_id=submission_job_drivers.get(submission_id).id
                if submission_id in submission_job_drivers
                else None,
                driver_info=submission_job_drivers.get(submission_id),
                type=JobType.SUBMISSION,
            )
            for submission_id, job in submission_jobs.items()
        ]
        return Response(
            text=json.dumps(
                [
                    *[submission_job.dict() for submission_job in submission_jobs],
                    *[job_info.dict() for job_info in driver_jobs.values()],
                ]
            ),
            content_type="application/json",
        )

    @routes.get("/api/jobs/{job_or_submission_id}/logs")
    async def get_job_logs(self, req: Request) -> Response:
        job_or_submission_id = req.match_info["job_or_submission_id"]
        if self._job_id_patt.match(job_or_submission_id):
            # 若是能在 job_table 里找到，则说明JobType是DRIVER，此API无法处理。
            if await self.gcs_client.async_get_job_info(job_or_submission_id):
                return Response(
                    text="Can only get logs of submission type jobs",
                    status=aiohttp.web.HTTPBadRequest.status_code,
                )

        job = await self._job_info_client.get_info(job_or_submission_id)
        if job:
            try:
                job_agent_client = self.get_job_driver_agent_client(job)
                payload = (
                    await job_agent_client.get_job_logs_internal(job_or_submission_id)
                    if job_agent_client
                    else JobLogsResponse("")
                )
                return Response(
                    text=json.dumps(dataclasses.asdict(payload)),
                    content_type="application/json",
                )
            except Exception:
                return Response(
                    text=traceback.format_exc(),
                    status=aiohttp.web.HTTPInternalServerError.status_code,
                )
        else:
            return Response(
                text=f"Job {job_or_submission_id} does not exist",
                status=aiohttp.web.HTTPNotFound.status_code,
            )

        if job.type is not JobType.SUBMISSION:
            return Response(
                text="Can only get logs of submission type jobs",
                status=aiohttp.web.HTTPBadRequest.status_code,
            )

        try:
            node_id = NodeID.from_hex(job.driver_node_id)
            node_dict = self.gcs_client.async_get_all_node_info(node_id=node_id)
            node_info = protobuf_message_to_dict(
                node_dict[node_id], fields_to_decode=[]
            )
            if node_info["state"] == "DEAD":
                raise RuntimeError(
                    "Request failed with status code 404: The node hosting the driver is dead."
                )

            job_agent_client = self.get_job_driver_agent_client(job)
            payload = (
                await job_agent_client.get_job_logs_internal(job.submission_id)
                if job_agent_client
                else JobLogsResponse("")
            )
            return Response(
                text=json.dumps(dataclasses.asdict(payload)),
                content_type="application/json",
            )
        except Exception:
            return Response(
                text=traceback.format_exc(),
                status=aiohttp.web.HTTPInternalServerError.status_code,
            )

    async def delete_job_log_file(
        self,
        job_type: str,
        job_or_submission_id: str,
        driver_node_id: str,
        driver_pid: int = None,
    ):
        if driver_node_id not in self._agents:
            driver_agent_http_address = await self._gcs_aio_client.internal_kv_get(
                f"DASHBOARD_AGENT_HTTP_ADDRESS:{driver_node_id}".encode(),
                namespace=KV_NAMESPACE_DASHBOARD,
            )
            if driver_agent_http_address is None:
                raise Exception("Failed to obtain the HTTP address of the driver_agent")
            driver_agent_http_address = "http://" + driver_agent_http_address.decode()
            self._agents[driver_node_id] = JobAgentSubmissionClient(
                driver_agent_http_address
            )
        job_agent_client = self._agents[driver_node_id]
        return await job_agent_client.delete_job_logs_internal(
            job_or_submission_id, job_type, driver_pid
        )

    @routes.delete("/api/jobs/{job_or_submission_id}/logs")
    async def delete_job_logs(self, req: Request) -> Response:
        job_or_submission_id = req.match_info["job_or_submission_id"]
        try:
            job_info = await self._gcs_aio_client.get_job_info(job_or_submission_id)
            if job_info:
                if job_info.is_dead:
                    driver_node_id = job_info.driver_address.raylet_id.hex()
                    status = await self.delete_job_log_file(
                        "DRIVER",
                        job_or_submission_id,
                        driver_node_id,
                        job_info.driver_pid,
                    )
                else:
                    status = "JOB_IS_RUNNING"
                return Response(
                    text=json.dumps({"status": status}), content_type="application/json"
                )

            job_info = await self._job_info_client.get_info(job_or_submission_id)
            if job_info:
                if job_info.status is None or not job_info.status.is_terminal():
                    status = "JOB_IS_RUNNING"
                else:
                    job_info_dict = await self._gcs_aio_client.get_all_job_info(
                        job_or_submission_id=job_or_submission_id
                    )
                    for job_id, job_item in job_info_dict.items():
                        driver_node_id = job_item.driver_address.raylet_id.hex()
                        await self.delete_job_log_file(
                            "DRIVER", job_id.hex(), driver_node_id, job_item.driver_pid
                        )
                    status = await self.delete_job_log_file(
                        "SUBMISSION", job_or_submission_id, job_info.driver_node_id
                    )
                return Response(
                    text=json.dumps({"status": status}), content_type="application/json"
                )
            return Response(
                text=f"Job {job_or_submission_id} does not exist",
                status=aiohttp.web.HTTPNotFound.status_code,
            )
        except Exception:
            return Response(
                text=traceback.format_exc(),
                status=aiohttp.web.HTTPInternalServerError.status_code,
            )

    @routes.get(
        "/api/jobs/{job_or_submission_id}/logs/tail", resp_type=ResponseType.WEBSOCKET
    )
    async def tail_job_logs(self, req: Request) -> StreamResponse:
        job_or_submission_id = req.match_info["job_or_submission_id"]
        if self._job_id_patt.match(job_or_submission_id):
            # 若是能在 job_table 里找到，则说明JobType是DRIVER，此API无法处理。
            if await self.gcs_client.async_get_job_info(job_or_submission_id):
                return Response(
                    text="Can only get logs of submission type jobs",
                    status=aiohttp.web.HTTPBadRequest.status_code,
                )

        if await self._job_info_client.get_info(job_or_submission_id):
            submission_id = job_or_submission_id
            ws = aiohttp.web.WebSocketResponse()
            await ws.prepare(req)

            driver_agent_http_address = None
            while driver_agent_http_address is None:
                job = await self._job_info_client.get_info(submission_id)
                driver_agent_http_address = job.driver_agent_http_address
                status = job.status
                if status.is_terminal() and driver_agent_http_address is None:
                    # Job exited before supervisor actor started.
                    return ws

                await asyncio.sleep(self.WAIT_FOR_SUPERVISOR_ACTOR_INTERVAL_S)

            job_agent_client = self.get_job_driver_agent_client(job)

            async for lines in job_agent_client.tail_job_logs(job_or_submission_id):
                await ws.send_str(lines)

            return ws
        else:
            return Response(
                text=f"Job {job_or_submission_id} does not exist",
                status=aiohttp.web.HTTPNotFound.status_code,
            )

    def get_job_driver_agent_client(
        self, job: JobDetails
    ) -> Optional[JobAgentSubmissionClient]:
        if job.driver_agent_http_address is None:
            return None

        driver_node_id = job.driver_node_id
        if driver_node_id not in self._agents:
            self._agents[driver_node_id] = JobAgentSubmissionClient(
                job.driver_agent_http_address
            )

        return self._agents[driver_node_id]

    @routes.get("/api/component_activities")
    async def get_component_activities(
        self, req: aiohttp.web.Request
    ) -> aiohttp.web.Response:
        timeout = req.query.get("timeout", None)
        if timeout and timeout.isdigit():
            timeout = int(timeout)
        else:
            timeout = 30

        # Get activity information for driver
        driver_activity_info = await self._get_job_activity_info(timeout=timeout)
        resp = {"driver": dict(driver_activity_info)}

        if RAY_CLUSTER_ACTIVITY_HOOK in os.environ:
            try:
                cluster_activity_callable = load_class(
                    os.environ[RAY_CLUSTER_ACTIVITY_HOOK]
                )
                external_activity_output = cluster_activity_callable()
                assert isinstance(external_activity_output, dict), (
                    f"Output of hook {os.environ[RAY_CLUSTER_ACTIVITY_HOOK]} "
                    "should be Dict[str, RayActivityResponse]. Got "
                    f"output: {external_activity_output}"
                )
                for component_type in external_activity_output:
                    try:
                        component_activity_output = external_activity_output[
                            component_type
                        ]
                        # Parse and validate output to type RayActivityResponse
                        component_activity_output = RayActivityResponse(
                            **dict(component_activity_output)
                        )
                        resp[component_type] = dict(component_activity_output)
                    except Exception as e:
                        logger.exception(
                            f"Failed to get activity status of {component_type} "
                            f"from user hook {os.environ[RAY_CLUSTER_ACTIVITY_HOOK]}."
                        )
                        resp[component_type] = {
                            "is_active": RayActivityStatus.ERROR,
                            "reason": repr(e),
                            "timestamp": datetime.now().timestamp(),
                        }
            except Exception as e:
                logger.exception(
                    "Failed to get activity status from user "
                    f"hook {os.environ[RAY_CLUSTER_ACTIVITY_HOOK]}."
                )
                resp["external_component"] = {
                    "is_active": RayActivityStatus.ERROR,
                    "reason": repr(e),
                    "timestamp": datetime.now().timestamp(),
                }

        return aiohttp.web.Response(
            text=json.dumps(resp),
            content_type="application/json",
            status=aiohttp.web.HTTPOk.status_code,
        )

    async def _get_job_activity_info(self, timeout: int) -> RayActivityResponse:
        # Returns if there is Ray activity from drivers (job).
        # Drivers in namespaces that start with _ray_internal_ are not
        # considered activity.
        # This includes the _ray_internal_dashboard job that gets automatically
        # created with every cluster
        try:
            reply = await self.gcs_client.async_get_all_job_info(
                skip_submission_job_info_field=True,
                skip_is_running_tasks_field=True,
                timeout=timeout,
            )

            num_active_drivers = 0
            latest_job_end_time = 0
            for job_table_entry in reply.values():
                is_dead = bool(job_table_entry.is_dead)
                in_internal_namespace = job_table_entry.config.ray_namespace.startswith(
                    "_ray_internal_"
                )
                latest_job_end_time = (
                    max(latest_job_end_time, job_table_entry.end_time)
                    if job_table_entry.end_time
                    else latest_job_end_time
                )
                if not is_dead and not in_internal_namespace:
                    num_active_drivers += 1

            current_timestamp = datetime.now().timestamp()
            # Latest job end time must be before or equal to the current timestamp.
            # Job end times may be provided in epoch milliseconds. Check if this
            # is true, and convert to seconds
            if latest_job_end_time > current_timestamp:
                latest_job_end_time = latest_job_end_time / 1000
                assert current_timestamp >= latest_job_end_time, (
                    f"Most recent job end time {latest_job_end_time} must be "
                    f"before or equal to the current timestamp {current_timestamp}"
                )

            is_active = (
                RayActivityStatus.ACTIVE
                if num_active_drivers > 0
                else RayActivityStatus.INACTIVE
            )
            return RayActivityResponse(
                is_active=is_active,
                reason=f"Number of active drivers: {num_active_drivers}"
                if num_active_drivers
                else None,
                timestamp=current_timestamp,
                # If latest_job_end_time == 0, no jobs have finished yet so don't
                # populate last_activity_at
                last_activity_at=latest_job_end_time if latest_job_end_time else None,
            )
        except Exception as e:
            logger.exception("Failed to get activity status of Ray drivers.")
            return RayActivityResponse(
                is_active=RayActivityStatus.ERROR,
                reason=repr(e),
                timestamp=datetime.now().timestamp(),
            )

    async def run(self):
        await super().run()
        if not self._job_info_client:
            self._job_info_client = JobInfoStorageClient(self.gcs_client)
