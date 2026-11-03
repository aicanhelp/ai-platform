import math

import numpy as np

from ray.dashboard.modules.job.pydantic_models import ProfileResult


def get_all_classes_profile_results(profile_data):
    classes_profile_data = {}
    for class_name, class_profile_data in profile_data.items():
        for actor_profile_data in class_profile_data:
            if class_name not in classes_profile_data:
                classes_profile_data[class_name] = {**actor_profile_data}
                classes_profile_data[class_name]["aicoreUsage"] = [
                    *actor_profile_data["aicoreUsage"]
                ]
                classes_profile_data[class_name]["npuMemoryUsageRate"] = [
                    *actor_profile_data["npuMemoryUsageRate"]
                ]
                classes_profile_data[class_name]["cpuUsage"] = [
                    *actor_profile_data["cpuUsage"]
                ]
                classes_profile_data[class_name]["memUsage"] = [
                    *actor_profile_data["memUsage"]
                ]
            else:
                classes_profile_data[class_name]["aicoreUsage"].extend(
                    actor_profile_data.get("aicoreUsage", [])
                )
                classes_profile_data[class_name]["npuMemoryUsageRate"].extend(
                    actor_profile_data.get("npuMemoryUsageRate", [])
                )
                classes_profile_data[class_name]["cpuUsage"].extend(
                    actor_profile_data.get("cpuUsage", [])
                )
                classes_profile_data[class_name]["memUsage"].extend(
                    actor_profile_data.get("memUsage", [])
                )

    classes_profile_results = {}
    for class_name, data in classes_profile_data.items():
        result = ProfileResult()
        result.childProcessNum = data["childProcessNum"]
        result.npuChildProcessNum = data["npuChildProcessNum"]

        if len(data["requiredResources"]) == 0:
            data["requiredResources"]["CPU"] = 1
        npu_needed = data["requiredResources"].get("NPU", 0)
        if npu_needed > 0 and npu_needed <= 1:
            if npu_needed < 1:
                data["requiredResources"]["NPU"] = round((npu_needed - 0.9) * 10, 2)
            else:
                data["requiredResources"]["NPU"] = round(npu_needed, 2)
        result.originalRequiredResources = data["requiredResources"]

        npu_profile = {
            "aiCore": {"average_usage": 0, "max_usage": 0, "Q3": 0},
            "HBM": {"average_usage": 0, "max_usage": 0, "Q3": 0},
        }
        aicore_usage = [x for x in data.get("aicoreUsage", []) if x != 0]
        if len(aicore_usage) > 0:
            npu_profile["aiCore"]["average_usage"] = (
                math.ceil(sum(aicore_usage) / len(aicore_usage) * 100) / 100
            )
            npu_profile["aiCore"]["max_usage"] = (
                math.ceil(max(aicore_usage) * 100) / 100
            )
            data_array = np.array(aicore_usage)
            npu_profile["aiCore"]["Q3"] = (
                math.ceil(np.percentile(data_array, 75) * 100) / 100
            )

        HBM_usage = [x for x in data.get("npuMemoryUsageRate", []) if x != 0]
        if len(HBM_usage) > 0:
            npu_profile["HBM"]["average_usage"] = (
                math.ceil(sum(HBM_usage) / len(HBM_usage) * 100) / 100
            )
            npu_profile["HBM"]["max_usage"] = math.ceil(max(HBM_usage) * 100) / 100
            data_array = np.array(HBM_usage)
            npu_profile["HBM"]["Q3"] = (
                math.ceil(np.percentile(data_array, 75) * 100) / 100
            )

        cpu_profile = {"average_usage": 0, "max_usage": 0, "Q3": 0}
        import platform

        cpu_profile["type"] = platform.machine()
        cpu_usage = [x for x in data.get("cpuUsage", []) if x != 0]
        if len(cpu_usage) > 0:
            cpu_profile["average_usage"] = (
                math.ceil(sum(cpu_usage) / len(cpu_usage) * 100) / 100
            )
            cpu_profile["max_usage"] = math.ceil(max(cpu_usage) * 100) / 100
            data_array = np.array(cpu_usage)
            cpu_profile["Q3"] = math.ceil(np.percentile(data_array, 75) * 100) / 100

        mem_profile = {"average_usage": 0, "max_usage": 0, "Q3": 0}
        mem_usage = [x for x in data.get("memUsage", []) if x != 0]
        if len(mem_usage) > 0:
            mem_profile["average_usage"] = (
                math.ceil(sum(mem_usage) / len(mem_usage) / 1024 / 1024 * 100) / 100
            )
            mem_profile["max_usage"] = (
                math.ceil(max(mem_usage) / 1024 / 1024 * 100) / 100
            )
            data_array = np.array(mem_usage)
            mem_profile["Q3"] = (
                math.ceil(np.percentile(data_array, 75) / 1024 / 1024 * 100) / 100
            )

        result.npuProfile = npu_profile
        result.cpuProfile = cpu_profile
        result.memoryProfile = mem_profile

        classes_profile_results[class_name] = result

    return classes_profile_results
