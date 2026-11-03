import logging

from kubernetes import client
import yaml


class ResourceMonitor:
    def __init__(
        self,
        feature_category: str,
        step_logger: logging.Logger,
        resource_category_path,
    ):
        self._step_logger = step_logger
        self._feature_category = feature_category
        self._resource_category_path = resource_category_path
        self._core_v1_api = client.CoreV1Api()
        self._pod_list: list[str] = self._load_pod_list()
        self._node_list: list[str] = self._get_node_names()

    def _get_node_names(self) -> list[str]:
        try:
            return [
                node.metadata.name
                for node in self._core_v1_api.list_node().items
            ]
        except Exception as e:
            self._step_logger.error(f"获取节点列表失败: {e}")
            return []

    def _load_pod_list(self) -> list[str]:
        """
        加载 pod_categorization.yaml，获取指定 feature_category 下的 pod 列表。
        """
        self._step_logger.info(f"加载 [{self._feature_category}] 的监控 Pod 列表。")
        with open(self._resource_category_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if not data:
            return []

        pod_category = next((c for c in data if c['category'] == self._feature_category))
        if not pod_category:
            return []

        distribution_category = next((c for c in data if c['category'] == "distribution"))
        pod_category['rules'].extend(distribution_category['rules'])

        pod_dict = []
        for rule in pod_category['rules']:
            namespace = rule['namespace']
            for selector in rule['labelSelectors']:
                pod_list = self._core_v1_api.list_namespaced_pod(namespace, label_selector=selector)
                for pod in pod_list.items:
                    if pod.status.phase == "Running":
                        name = pod.metadata.name
                        pod_dict.append(f"{namespace}_{name}")

        return pod_dict

    def monitor_pod_resources(self):
        """
        获取并更新监控 pod 的 CPU 和内存资源占用。
        """
        self._step_logger.info(f"进行 {self._feature_category} 资源监控数据采集。")

        resource_results = []

        for node in self._node_list:
            try:
                summary: dict = self._core_v1_api.api_client.call_api(
                    f"/api/v1/nodes/{node}/proxy/stats/summary",
                    "GET",
                    response_type="object",
                    auth_settings=["BearerToken"],
                    _preload_content=True
                )[0]

                pods = summary.get("pods", [])
                for pod in pods:
                    pod_ref = pod.get("podRef", {})
                    namespace = pod_ref.get("namespace")
                    name = pod_ref.get("name")

                    if not namespace or not name:
                        continue

                    pod_key = f"{namespace}_{name}"
                    if pod_key not in self._pod_list:
                        continue

                    cpu = pod.get("cpu", {}).get("usageNanoCores")
                    memory = pod.get("memory", {}).get("usageBytes")

                    resource_results.append({
                        "pod": pod_key,
                        "cpu": cpu,
                        "memory": memory
                    })

            except Exception as e:
                self._step_logger.error(
                    f"资源数据采集时，获取 node {node} summary 失败: {e}"
                )
        return resource_results
