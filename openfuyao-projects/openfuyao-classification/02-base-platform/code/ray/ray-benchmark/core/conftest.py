import binascii

import grpc
from google.protobuf.json_format import ParseDict
import pytest

import ray
from ray.core.generated import gcs_pb2, gcs_service_pb2, gcs_service_pb2_grpc

from ray._private.gcs_utils import GcsChannel


from google.protobuf import json_format


@pytest.fixture()
def cluster_with_jobs(job_num=100):
    address_info = ray.init(include_dashboard=False)
    gcs_address = address_info.address_info.get("gcs_address")
    channel = grpc.insecure_channel(gcs_address)
    job_info_stub = gcs_service_pb2_grpc.JobInfoGcsServiceStub(channel)

    job_dict = {
        "jobId": "AQAAAA==",
        "isDead": True,
        "timestamp": "1740647162531",
        "driverIpAddress": "172.17.0.2",
        "driverPid": "139226",
        "config": {
            "runtimeEnvInfo": {
                "serializedRuntimeEnv": "{}",
                "runtimeEnvConfig": {"setupTimeoutSeconds": 600, "eagerInstall": True},
            },
            "rayNamespace": "12f0c4c8-10a1-4fd2-9d92-5ec537093d93",
            "defaultActorLifetime": "NON_DETACHED",
            "pyDriverSysPath": ["/root", "/root"],
        },
        "startTime": "1740647159514",
        "endTime": "1740647162531",
        "entrypoint": "python task.py",
        "driverAddress": {
            "rayletId": "eWimVgTHatqTf01chxMvbeEF0Z8Z0XLYe0SRQg==",
            "ipAddress": "172.17.0.2",
            "port": 10002,
            "workerId": "AQAAAP///////////////////////////////w==",
        },
    }
    job_msg = ParseDict(job_dict, gcs_pb2.JobTableData())

    for idx in range(job_num):
        job_msg.job_id = binascii.unhexlify(f"{idx:08X}")
        request = gcs_service_pb2.AddJobRequest(data=job_msg)
        job_info_stub.AddJob(request, timeout=30)

    # +1 because job_agent will submit a test job by default
    yield gcs_address, job_num + 1

    ray.shutdown()


@pytest.fixture()
def cluster_with_tasks(task_num=10000):
    address_info = ray.init(include_dashboard=False)

    gcs_address = address_info.address_info.get("gcs_address")
    gcs_channel = GcsChannel(gcs_address=gcs_address, aio=False)
    gcs_channel.connect()

    task_events_dict = {
        "taskId": "yO9FzNARJXH///////////////8BAAAA",
        "taskInfo": {
            "name": "call",
            "funcOrClassName": "call",
            "jobId": "AQAAAA==",
            "taskId": "yO9FzNARJXH///////////////8BAAAA",
            "parentTaskId": "//////////////////////////8BAAAA",
            "requiredResources": {"CPU": 3.0},
            "runtimeEnvInfo": {
                "serializedRuntimeEnv": "{}",
                "runtimeEnvConfig": {"setupTimeoutSeconds": 600, "eagerInstall": True},
            },
        },
        "stateUpdates": {
            "nodeId": "qPoA8XnHqyvTwaN0xpAJ0z1EejZMUvI/goJ0Bg==",
            "workerId": "MxBh6pjsIW/6AQpXco28v4UedVEnDIr1Dvxd+g==",
            "taskLogInfo": {
                "stdoutFile": "/tmp/ray/session_2025-02-27_14-41-31_517001_272/logs/worker-331061ea98ec216ffa010a57728dbcbf851e7551270c8af50efc5dfa-ffffffff-21602.out",
                "stderrFile": "/tmp/ray/session_2025-02-27_14-41-31_517001_272/logs/worker-331061ea98ec216ffa010a57728dbcbf851e7551270c8af50efc5dfa-ffffffff-21602.err",
                "stdoutStart": 33,
                "stdoutEnd": 73,
                "stderrStart": 33,
                "stderrEnd": 33,
            },
            "workerPid": 21602,
            "stateTsNs": {
                "2": "1740640298120085841",
                "9": "1740640300367977050",
                "6": "1740640298123256273",
                "1": "1740640298120058411",
                "5": "1740640298122169529",
            },
        },
        "profileEvents": {
            "componentType": "worker",
            "componentId": "MxBh6pjsIW/6AQpXco28v4UedVEnDIr1Dvxd+g==",
            "nodeIpAddress": "172.17.0.4",
            "events": [
                {
                    "startTime": "1740640298127300540",
                    "endTime": "1740640298127305530",
                    "extraData": "{}",
                    "eventName": "task:deserialize_arguments",
                },
                {
                    "startTime": "1740640298127381260",
                    "endTime": "1740640300136629750",
                    "extraData": "{}",
                    "eventName": "task:execute",
                },
                {
                    "startTime": "1740640300136652210",
                    "endTime": "1740640300365949043",
                    "extraData": "{}",
                    "eventName": "task:store_outputs",
                },
                {
                    "startTime": "1740640298127258910",
                    "endTime": "1740640300365991233",
                    "extraData": '{"name": "__main__.call", "task_id": "c8ef45ccd0112571ffffffffffffffffffffffff01000000"}',
                    "eventName": "task::call",
                },
            ],
        },
        "jobId": "AQAAAA==",
    }

    task_info_stub = gcs_service_pb2_grpc.TaskInfoGcsServiceStub(gcs_channel.channel())

    task_events_msg = json_format.ParseDict(task_events_dict, gcs_pb2.TaskEvents())

    for idx in range(task_num):
        job_id = binascii.unhexlify(f"{idx:08X}")
        task_id = binascii.unhexlify(
            f"d8ef45ccd0112571ffffffffffffffffffffffff{idx:08X}"
        )
        parent_task_id = binascii.unhexlify(
            f"ffffffffffffffffffffffffffffffffffffffff{idx:08X}"
        )
        task_events_msg.job_id = job_id
        task_events_msg.task_id = task_id
        task_events_msg.task_info.job_id = job_id
        task_events_msg.task_info.task_id = task_id
        task_events_msg.task_info.parent_task_id = parent_task_id

        task_event_data = gcs_pb2.TaskEventData()
        task_event_data.events_by_task.append(task_events_msg)
        task_event_data.job_id = job_id

        request = gcs_service_pb2.AddTaskEventDataRequest(data=task_event_data)
        task_info_stub.AddTaskEventData(request, timeout=30)

    yield gcs_address, task_num

    ray.shutdown()
