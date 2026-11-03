from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import threading
import time
import traceback

from kubernetes import client
import numpy as np

from performance.runner.constants import (
    KUBECONFIG_PATH,
    RESOURCE_MONITOR_INTERVAL,
    RESOURCE_MONITOR_FAILED_THRESHOLD,
)
from performance.runner.resource_monitor.resource_monitor import ResourceMonitor
from performance.runner.session import SessionContext, POD_RESOURCES_FILE


def _should_continue_resource_monitor(
    start_monotonic: float,
    stop_thread: threading.Thread | None,
    duration_seconds: float | None,
) -> bool:
    if stop_thread is not None and not stop_thread.is_alive():
        return False
    if duration_seconds is not None and duration_seconds > 0:
        if time.monotonic() - start_monotonic >= duration_seconds:
            return False
    if stop_thread is not None:
        return stop_thread.is_alive()
    return True


def _resource_monitor_stop_reason(
    start_monotonic: float,
    stop_thread: threading.Thread | None,
    duration_seconds: float | None,
) -> str:
    if stop_thread is not None and not stop_thread.is_alive():
        return "网络监控线程已结束"
    if duration_seconds is not None and duration_seconds > 0:
        if time.monotonic() - start_monotonic >= duration_seconds:
            return f"已达到设定运行时长 {duration_seconds}s"
    return "监控循环结束"


def exec_resource_monitor(
    ctx: SessionContext,
    feature_category: str,
    resource_category_path: Path,
    *,
    stop_thread: threading.Thread | None = None,
    duration_seconds: float | None = None,
):
    """
    执行资源监控主流程。

    停止条件（满足其一即停止）：
    - stop_thread 已结束（与网络监控联动时使用）
    - duration_seconds > 0 且已达设定秒数
    - duration_seconds <= 0 且无 stop_thread 时无限运行，直至进程中断或失败超阈值
    """
    rotate_step_log = (
        stop_thread is None
        and duration_seconds is not None
        and duration_seconds <= 0
    )
    step_logger = ctx.step_logger("resource", feature_category, rotate=rotate_step_log)
    pod_writer = ctx.resource_pod_writer(feature_category)

    try:
        step_logger.info(f"为 [{feature_category}] 开启资源监控器。")
        monitor = ResourceMonitor(feature_category, step_logger, resource_category_path)
    except Exception as e:
        step_logger.error(f"资源监控开启失败: {e} \n {traceback.format_exc()}")
        return

    succeed_round, fail_round = 0, 0
    start_monotonic = time.monotonic()

    if stop_thread is not None and duration_seconds is None:
        step_logger.info(f"[{feature_category}] 资源监控将持续运行，直到网络监控线程结束。")
    elif duration_seconds is not None and duration_seconds > 0:
        step_logger.info(f"[{feature_category}] 资源监控将独立运行 {duration_seconds}s。")
    else:
        step_logger.info(f"[{feature_category}] 资源监控将独立无限运行，直至手动中断或失败超阈值。")

    try:
        while _should_continue_resource_monitor(start_monotonic, stop_thread, duration_seconds):
            try:
                pods = monitor.monitor_pod_resources()
                succeed_round += 1
                pod_writer.write({
                    'run_id': ctx.paths.run_id,
                    'feature': feature_category,
                    'round': succeed_round,
                    'timestamp': time.time(),
                    'pods': pods,
                })
                step_logger.info(f"[{feature_category}] 第 {succeed_round} 轮资源监控数据采集成功完成。")
            except Exception as e:
                step_logger.error(f"资源监控失败: {e} \n {traceback.format_exc()}")
                fail_round += 1
                if RESOURCE_MONITOR_FAILED_THRESHOLD and fail_round > RESOURCE_MONITOR_FAILED_THRESHOLD:
                    step_logger.error(f"[{feature_category}] 资源监控失败次数超过阈值，停止监控。")
                    break
            time.sleep(RESOURCE_MONITOR_INTERVAL)
    except KeyboardInterrupt:
        step_logger.info(f"[{feature_category}] 收到中断信号，资源监控停止。")

    stop_reason = _resource_monitor_stop_reason(start_monotonic, stop_thread, duration_seconds)
    step_logger.info(
        f"[{feature_category}] 资源监控停止（{stop_reason}）。共完成 {succeed_round} 轮采样。"
    )


def generate_resource_csv(
    ctx: SessionContext,
    result_csv_path: Path,
):
    def calc_stats(values: list[float], res_type: str, round_digit: int = 3) -> dict[str, float]:
        arr = np.array(values, dtype=float)
        ratio = 1e6 if res_type == "cpu" else 1024 * 1024 * 1024
        return {
            f"{res_type}_avg": round(float(np.mean(arr)) / ratio, round_digit),
            f"{res_type}_p25": round(float(np.percentile(arr, 25)) / ratio, round_digit),
            f"{res_type}_p75": round(float(np.percentile(arr, 75)) / ratio, round_digit),
            f"{res_type}_p90": round(float(np.percentile(arr, 90)) / ratio, round_digit),
            f"{res_type}_p99": round(float(np.percentile(arr, 99)) / ratio, round_digit),
        }

    fieldnames = ['namespace', 'pod', 'sample'] + \
        ['cpu_avg', 'cpu_p25', 'cpu_p75', 'cpu_p90', 'cpu_p99'] + \
        ['memory_avg', 'memory_p25', 'memory_p75', 'memory_p90', 'memory_p99']
    csv_data = []

    pod_cpu = defaultdict(list)
    pod_memory = defaultdict(list)

    resource_monitor_dir = ctx.paths.resource_monitor
    if not resource_monitor_dir.exists():
        return

    for feature in os.listdir(resource_monitor_dir):
        feature_dir = resource_monitor_dir / feature
        if not feature_dir.is_dir():
            continue
        pod_file = feature_dir / POD_RESOURCES_FILE
        if not pod_file.exists():
            continue

        with pod_file.open('r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                for pod_res in record.get('pods', []):
                    pod_cpu[pod_res["pod"]].append(pod_res["cpu"])
                    pod_memory[pod_res["pod"]].append(pod_res["memory"])

    for pod_key in pod_cpu:
        csv_data.append(
            pod_key.split("_") + [len(pod_cpu[pod_key])] +
            list(calc_stats(pod_cpu[pod_key], 'cpu').values()) +
            list(calc_stats(pod_memory[pod_key], 'memory').values())
        )

    csv_data.sort(key=lambda row: (row[0], row[1]))

    result_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(csv_data)
