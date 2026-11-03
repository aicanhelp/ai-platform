# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

#!/usr/bin/env python3
"""
Continuously stream vLLM decode/prefill pod logs, collect KV cache transfer cost,
and compute percentile distributions when idle timeout is reached.

When a pod restarts, the script automatically:
  - Saves previous container logs (with errors) to local files
  - Reconnects to the new container and continues collecting

Usage:
    python3 collect_kv_transfer_metrics.py [-n NAMESPACE] [-t TIMEOUT] [-o OUTPUT_DIR]

Options:
    -n, --namespace   Kubernetes namespace  (default: ai-inference)
    -t, --timeout     Seconds with no new cost entry before auto-exit (default: 10)
    -o, --output-dir  Directory to save crash logs (default: ./decode_crash_logs)
"""

import subprocess
import re
import sys
import os
import math
import threading
import time
import argparse
from collections import defaultdict
from datetime import datetime

PERCENTILES = [10, 25, 50, 75, 90, 95, 99]
COST_PATTERN = re.compile(r"cost:\s*(\d+)\s*us")

STAT_LINE_MARKER = "GPU KV cache usage"
STAT_FIELDS = {
    'prompt_throughput':     re.compile(r"Avg prompt throughput:\s*([\d.]+)"),
    'generation_throughput': re.compile(r"Avg generation throughput:\s*([\d.]+)"),
    'running':               re.compile(r"Running:\s*(\d+)"),
    'waiting':               re.compile(r"Waiting:\s*(\d+)"),
    'gpu_kv_usage':          re.compile(r"GPU KV cache usage:\s*([\d.]+)"),
    'prefix_cache_hit_rate': re.compile(r"Prefix cache hit rate:\s*([\d.]+)"),
    'preemptions':           re.compile(r"Preemptions:\s*(\d+)"),
    'ext_prefix_cache_hit_rate': re.compile(r"External prefix cache hit rate:\s*([\d.]+)"),
}

RECONNECT_INTERVAL = 3


def get_target_pods(namespace):
    """Return pods whose name contains 'decode' or 'prefill'."""
    cmd = ["kubectl", "get", "pods", "-n", namespace,
           "--no-headers", "-o", "custom-columns=NAME:.metadata.name"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error getting pods: {result.stderr}")
        sys.exit(1)

    pods = []
    for line in result.stdout.strip().split("\n"):
        name = line.strip()
        if name and ("decode" in name or "prefill" in name):
            pods.append(name)
    return sorted(pods)


def save_previous_logs(pod_name, namespace, output_dir, restart_events, lock):
    """Fetch and save the previous (crashed) container's logs."""
    cmd = ["kubectl", "logs", pod_name, "-n", namespace, "--previous"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return

    prev_logs = result.stdout
    if not prev_logs.strip():
        return

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{pod_name}_crash_{ts}.log")
    with open(filename, "w") as f:
        f.write(prev_logs)

    tail_lines = prev_logs.strip().split("\n")
    last_n = tail_lines[-min(30, len(tail_lines)):]

    with lock:
        restart_events.append({
            "pod": pod_name,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "log_file": filename,
            "log_lines": len(tail_lines),
            "tail": last_n,
        })

    return filename


def stream_pod_logs(pod_name, namespace, costs_dict, stat_logs, lock,
                    last_seen, stop_event, output_dir, restart_events):
    """Stream logs with auto-reconnect on pod restart."""
    while not stop_event.is_set():
        cmd = ["kubectl", "logs", "-f", pod_name, "-n", namespace]
        proc = None
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
            )
            while not stop_event.is_set():
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    continue
                m = COST_PATTERN.search(line)
                if m:
                    with lock:
                        costs_dict[pod_name].append(int(m.group(1)))
                        last_seen[0] = time.time()
                if STAT_LINE_MARKER in line:
                    entry = {}
                    for field, pat in STAT_FIELDS.items():
                        m2 = pat.search(line)
                        if m2:
                            entry[field] = float(m2.group(1))
                    if entry:
                        with lock:
                            stat_logs[pod_name].append(entry)
        except Exception:
            pass
        finally:
            if proc:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

        if stop_event.is_set():
            break

        # Stream ended unexpectedly — pod likely restarted
        with lock:
            msg = f"\n  [RESTART] {pod_name} log stream ended, saving previous container logs..."
            sys.stdout.write(msg)
            sys.stdout.flush()

        try:
            save_previous_logs(pod_name, namespace, output_dir, restart_events, lock)
        except Exception:
            pass

        # Wait before reconnecting
        for _ in range(RECONNECT_INTERVAL * 10):
            if stop_event.is_set():
                return
            time.sleep(0.1)

        with lock:
            sys.stdout.write(f"\n  [RECONNECT] {pod_name} reconnecting...\n")
            sys.stdout.flush()


def percentile(sorted_data, p):
    """Linear interpolation percentile (same as numpy default)."""
    n = len(sorted_data)
    if n == 0:
        return 0
    k = (p / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def format_us(value_us):
    if value_us >= 1_000_000:
        return f"{value_us / 1_000_000:.3f} s"
    elif value_us >= 1_000:
        return f"{value_us / 1_000:.2f} ms"
    else:
        return f"{value_us} us"


def print_distribution(label, costs):
    costs_sorted = sorted(costs)
    total = len(costs_sorted)
    avg = sum(costs_sorted) / total

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"  Total cost entries : {total}")
    print(f"  Min                : {format_us(costs_sorted[0]):>12}  ({costs_sorted[0]} us)")
    print(f"  Max                : {format_us(costs_sorted[-1]):>12}  ({costs_sorted[-1]} us)")
    print(f"  Avg                : {format_us(int(avg)):>12}  ({int(avg)} us)")
    print()
    print(f"  {'Percentile':<12} {'Value (us)':>12} {'Readable':>14}")
    print(f"  {'-' * 40}")
    for p in PERCENTILES:
        val = int(round(percentile(costs_sorted, p)))
        print(f"  P{p:<10} {val:>12}   {format_us(val):>12}")


def print_engine_stats(label, entries):
    """Print engine stat summary (LoggingStatLogger metrics) for a pod."""
    if not entries:
        return

    print(f"\n{'=' * 70}")
    print(f"  ENGINE STATS: {label}")
    print(f"{'=' * 70}")
    print(f"  Stat log entries : {len(entries)}")

    display_fields = [
        ('prefix_cache_hit_rate',      'Prefix cache hit %'),
        ('ext_prefix_cache_hit_rate',  'Ext prefix cache hit %'),
        ('gpu_kv_usage',               'GPU KV cache %'),
        ('running',                    'Running'),
        ('waiting',                    'Waiting'),
        ('preemptions',                'Preemptions'),
        ('prompt_throughput',          'Prompt tput (tok/s)'),
        ('generation_throughput',      'Gen tput (tok/s)'),
    ]

    print(f"\n  {'Metric':<22} {'Min':>10} {'Max':>10} {'Avg':>10} {'Last':>10}")
    print(f"  {'-' * 62}")

    for field_key, field_label in display_fields:
        values = [e[field_key] for e in entries if field_key in e]
        if not values:
            continue
        min_v = min(values)
        max_v = max(values)
        avg_v = sum(values) / len(values)
        last_v = values[-1]
        print(f"  {field_label:<22} {min_v:>10.2f} {max_v:>10.2f} {avg_v:>10.2f} {last_v:>10.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Stream vLLM decode pod logs and compute KV cache transfer cost distribution.")
    parser.add_argument("-n", "--namespace", default="ai-inference",
                        help="Kubernetes namespace (default: ai-inference)")
    parser.add_argument("-t", "--timeout", type=int, default=10,
                        help="Seconds with no new cost before auto-exit (default: 10)")
    parser.add_argument("-o", "--output-dir", default="./decode_crash_logs",
                        help="Directory to save crash logs (default: ./decode_crash_logs)")
    args = parser.parse_args()

    print(f"Namespace  : {args.namespace}")
    print(f"Timeout    : {args.timeout}s (auto-exit after no new cost)")
    print(f"Crash logs : {os.path.abspath(args.output_dir)}")
    print(f"Fetching decode / prefill pods ...")

    pods = get_target_pods(args.namespace)
    if not pods:
        print("No decode/prefill pods found!")
        sys.exit(1)

    decode_pods = [p for p in pods if "decode" in p]
    prefill_pods = [p for p in pods if "prefill" in p]
    print(f"Found {len(pods)} target pod(s)  "
          f"(decode={len(decode_pods)}, prefill={len(prefill_pods)}):")
    for p in pods:
        tag = "decode" if "decode" in p else "prefill"
        print(f"  - [{tag}] {p}")

    costs_dict = defaultdict(list)
    stat_logs = defaultdict(list)
    restart_events = []
    lock = threading.Lock()
    last_seen = [time.time()]
    stop_event = threading.Event()

    threads = []
    for pod in pods:
        t = threading.Thread(
            target=stream_pod_logs,
            args=(pod, args.namespace, costs_dict, stat_logs, lock, last_seen,
                  stop_event, args.output_dir, restart_events),
            daemon=True,
        )
        t.start()
        threads.append(t)

    print(f"\nStreaming logs ... (Ctrl+C to stop early)\n")

    start_time = time.time()
    try:
        while True:
            time.sleep(1)
            with lock:
                total = sum(len(v) for v in costs_dict.values())
                stat_total = sum(len(v) for v in stat_logs.values())
                per_pod_cost = {k: len(v) for k, v in costs_dict.items()}
                idle = time.time() - last_seen[0]
                restarts = len(restart_events)

            elapsed = time.time() - start_time

            def _short(name):
                for tag in ('-decode', '-prefill'):
                    if tag in name:
                        return name.split(tag)[0] + tag
                return name

            pod_summary = "  ".join(
                f"{_short(k)}:{v}" for k, v in sorted(per_pod_cost.items())
            ) if per_pod_cost else "waiting..."

            restart_tag = f"  restarts={restarts}" if restarts else ""
            sys.stdout.write(
                f"\r  [{elapsed:.0f}s] cost={total} stats={stat_total}  "
                f"idle={idle:.0f}s/{args.timeout}s{restart_tag}  |  {pod_summary}    "
            )
            sys.stdout.flush()

            if total > 0 and idle >= args.timeout:
                print(f"\n\n  No new cost for {args.timeout}s — stopping collection.")
                break
    except KeyboardInterrupt:
        print(f"\n\n  Interrupted by user — stopping collection.")

    stop_event.set()
    for t in threads:
        t.join(timeout=3)

    # --- Print results ---
    with lock:
        final_costs = dict(costs_dict)
        final_stats = dict(stat_logs)
        final_restarts = list(restart_events)

    has_cost = any(final_costs.values())
    has_stats = any(final_stats.values())

    if not has_cost and not has_stats:
        print("\n  No data collected.")
        sys.exit(0)

    # --- Cost distribution (decode pods) ---
    if has_cost:
        for pod in decode_pods:
            if final_costs.get(pod):
                print_distribution(f"Pod: {pod}", final_costs[pod])

        all_costs = []
        for p in decode_pods:
            all_costs.extend(final_costs.get(p, []))
        if len(decode_pods) > 1 and all_costs:
            print_distribution("ALL DECODE PODS AGGREGATED", all_costs)
    else:
        print("\n  No cost data collected from decode pods.")

    # --- Engine stats (all pods) ---
    for pod in pods:
        if final_stats.get(pod):
            print_engine_stats(pod, final_stats[pod])

    # --- Print restart / crash summary ---
    if final_restarts:
        print(f"\n{'=' * 70}")
        print(f"  POD RESTART EVENTS ({len(final_restarts)} total)")
        print(f"{'=' * 70}")
        for i, ev in enumerate(final_restarts, 1):
            print(f"\n  [{i}] Pod  : {ev['pod']}")
            print(f"      Time : {ev['time']}")
            print(f"      File : {ev['log_file']}")
            print(f"      Previous container log lines: {ev['log_lines']}")
            print(f"      --- Last lines before crash ---")
            for line in ev["tail"]:
                print(f"      | {line}")
    else:
        print(f"\n  No pod restarts detected during collection.")

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == "__main__":
    main()
