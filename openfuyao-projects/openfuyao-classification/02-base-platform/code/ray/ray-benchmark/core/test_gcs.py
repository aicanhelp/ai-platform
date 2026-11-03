import pytest
from ray.core.generated import gcs_service_pb2, gcs_service_pb2_grpc
from ray._raylet import GcsClient
from ray._private.gcs_utils import GcsChannel

# @pytest.mark.parametrize("cluster_with_jobs", [10, 100, 1000], indirect=True)
@pytest.mark.benchmark(group="core_gcs")
def test_get_all_job_info(cluster_with_jobs, benchmark):
    gcs_address, submitted_job_num = cluster_with_jobs
    gcs_client = GcsClient(address=gcs_address)
    job_reply = benchmark(gcs_client.get_all_job_info)

    assert len(job_reply) == submitted_job_num


@pytest.mark.benchmark(group="core_gcs")
def test_get_all_task(cluster_with_tasks, benchmark):
    gcs_address, submitted_task_num = cluster_with_tasks
    # gcs_client = GcsClient(address=gcs_address)
    gcs_channel = GcsChannel(gcs_address=gcs_address, aio=False)
    gcs_channel.connect()

    gcs_task_info_stub = gcs_service_pb2_grpc.TaskInfoGcsServiceStub(
        gcs_channel.channel()
    )
    request = gcs_service_pb2.GetTaskEventsRequest(
        limit=submitted_task_num,
    )
    task_reply = benchmark(gcs_task_info_stub.GetTaskEvents, request, timeout=30)

    assert len(task_reply.events_by_task) == submitted_task_num
