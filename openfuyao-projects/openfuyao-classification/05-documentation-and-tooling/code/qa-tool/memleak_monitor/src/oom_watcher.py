import logging
import os
from datetime import datetime, timezone

import requests
from kubernetes import client, config, watch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)
AGGREGATE_URL = os.getenv("AGGREGATE_URL", "http://aggregate.jitto.svc:5000/oom")
config.load_incluster_config()


def _is_container_oom_killed(container_status) -> bool:
    terminated = getattr(container_status.state, "terminated", None)
    last_terminated = getattr(container_status.last_state, "terminated", None)
    return (
        (terminated is not None and terminated.reason == "OOMKilled")
        or (last_terminated is not None and last_terminated.reason == "OOMKilled")
    )


def is_oom_killed(pod):
    if not pod.status.container_statuses:
        return False
    for container_status in pod.status.container_statuses:
        if (not container_status.ready) and _is_container_oom_killed(container_status):
            return True
    return False


def extract_oom_info(pod):
    oom_containers = []
    for container_status in pod.status.container_statuses:
        if _is_container_oom_killed(container_status):
            container_info = {
                "timestamp": int(datetime.now(timezone.utc).timestamp()),
                "node": pod.spec.node_name or "unknown",
                "namespace": pod.metadata.namespace,
                "pod": pod.metadata.name,
                "container": container_status.name
            }
            oom_containers.append(container_info)
    return oom_containers


def send_oom_event(event_data):
    try:
        response = requests.post(
            AGGREGATE_URL,
            json=event_data,
            headers={'Content-Type': 'application/json'},
        )
        logger.info(f"OOM event sent, url={AGGREGATE_URL}, status={response.status_code}")
    except requests.RequestException:
        logger.exception("Error sending OOM event")


if __name__ == "__main__":
    v1 = client.CoreV1Api()
    w = watch.Watch()
    for event in w.stream(v1.list_namespaced_pod, ''):
        logger.info((
            "Kubernetes event received, "
            f"event_type={event.get('type')}, "
            f"namespace={event['object'].metadata.namespace}, "
            f"pod={event['object'].metadata.name}"
        ))
        pod = event.get('object')
        event_type = event.get('type')
        
        if event_type == "MODIFIED" and pod:
            if is_oom_killed(pod):
                logger.warning((
                    "Detected OOMKilled pod, "
                    f"namespace={pod.metadata.namespace}, "
                    f"pod={pod.metadata.name}"
                ))
                oom_events = extract_oom_info(pod)
                for oom_event in oom_events:
                    send_oom_event(oom_event)
