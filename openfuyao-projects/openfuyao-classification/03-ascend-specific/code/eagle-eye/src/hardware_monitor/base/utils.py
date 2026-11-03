# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""
This module provides utilities to initialize metric fields, collect errors,
and enrich results with container information.
"""

from typing import List, Any

from hardware_monitor.collector.container_info.containerd.collector_for_container_info import \
    (DevicesInfo,
     CONTAINERS_DEVICES_CACHE_KEY)


def init_metric_fields(fields: List[str]) -> dict[str, Any]:
    """
    Initializes a dictionary with the provided field names, setting each field's value to "NA".
    Additionally, it adds an "errors" key with an empty list as its value.
    """
    result = {field: "NA" for field in fields}
    result["errors"] = []
    return result


def add_error(result: dict[str, Any],
              module: str,
              code: int,
              message: str) -> None:
    """
    Adds an error entry to the errors list in the provided result dictionary.
    """
    result["errors"].append({
        "module": module,
        "error_code": code,
        "error_message": message
    })


def enrich_result_with_container_info(result: dict[str, Any], phy_id: int, cache: Any) -> dict[str, Any]:
    """
    Enriches a metric result dict with namespace, pod_name, container_name from the
    container info cache.
    """
    raw = cache.get(CONTAINERS_DEVICES_CACHE_KEY) if cache else None
    if not raw:
        return result

    containers_devices_infos = raw[0] if isinstance(raw, list) else raw
    if not isinstance(containers_devices_infos, dict):
        return result

    device_to_info = {}
    for info in containers_devices_infos.values():
        if isinstance(info, DevicesInfo) and info.device_ids:
            for dev_id in info.device_ids:
                device_to_info[dev_id] = info

    container_info = device_to_info.get(phy_id)

    if container_info:
        result["namespace"] = container_info.namespace or ""
        result["pod_name"] = container_info.pod_name or ""
        result["container_name"] = container_info.container_name or ""
    else:
        result["namespace"] = ""
        result["pod_name"] = ""
        result["container_name"] = ""

    return result

