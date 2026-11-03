/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package hardware

const (
	benchmarkPyContent = `import torch
import torch.distributed as dist
import torch_npu
import os

def main():
    dist.init_process_group(backend='hccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.npu.set_device(rank)

    # Record device mapping information
    actual_device = torch.npu.current_device()
    physical_devices = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', 'NOT_SET')

    # Write to file for later parsing
    with open(f'/tmp/rank_{rank}_device.txt', 'w') as f:
        f.write(f"rank={rank}\n")
        f.write(f"actual_device={actual_device}\n")
        f.write(f"physical_devices={physical_devices}\n")

    print(f"[Device Mapping] Rank {rank}: actual_device={actual_device}, physical={physical_devices}")
    print(f"Rank {rank}/{world_size} starting on device npu:{rank}")

    # Warmup
    for _ in range(3):
        t = torch.randn(4096, 4096, device=f'npu:{rank}')
        torch.matmul(t, t)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

    # Benchmark
    for step in range(50):
        tensor = torch.randn(4096, 4096, device=f'npu:{rank}')
        result = torch.matmul(tensor, tensor)
        dist.all_reduce(result, op=dist.ReduceOp.SUM)

        if rank == 0 and step % 10 == 0:
            print(f"Step {step} completed")

    dist.destroy_process_group()
    print(f"Rank {rank} finished")

if __name__ == "__main__":
    main()
`

	fixRankPyContent = `import json
import glob
import re
import sys
import os

prof_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/prof_data"

# Step 1: Build mapping from physical device ID to rank ID
# by reading /tmp/rank_X_device.txt files
device_to_rank = {}
for rank_file in glob.glob("/tmp/rank_*_device.txt"):
    try:
        with open(rank_file, 'r') as f:
            content = f.read()

        # Parse rank and actual_device from file
        rank_match = re.search(r'rank=(\d+)', content)
        device_match = re.search(r'actual_device=(\d+)', content)

        if not rank_match or not device_match:
            print(f"Error: {rank_file} missing rank or actual_device field", file=sys.stderr)
            sys.exit(1)

        rank = int(rank_match.group(1))
        actual_device = int(device_match.group(1))

        # Get physical devices list
        phys_match = re.search(r'physical_devices=([0-9,]+)', content)
        if not phys_match:
            print(f"Error: {rank_file} missing physical_devices field", file=sys.stderr)
            sys.exit(1)

        phys_list = [int(x) for x in phys_match.group(1).split(',')]
        if actual_device >= len(phys_list):
            print(
                f"Error: {rank_file} actual_device={actual_device} "
                f"out of range (phys_list has {len(phys_list)} devices)",
                file=sys.stderr
            )
            sys.exit(1)

        physical_device = phys_list[actual_device]
        device_to_rank[physical_device] = rank
        print(f"Mapping: physical device {physical_device} -> rank {rank}")

    except Exception as e:
        print(f"Error reading {rank_file}: {e}", file=sys.stderr)
        sys.exit(1)

if not device_to_rank:
    print("Error: No device-to-rank mapping found", file=sys.stderr)
    sys.exit(1)

# Step 2: Fix rank_id in all info.json.X files
info_files = glob.glob(f"{prof_dir}/PROF_*/device_*/info.json.*")
info_files = [f for f in info_files if not f.endswith('.done')]

if not info_files:
    print(f"Error: No info.json files found in {prof_dir}", file=sys.stderr)
    sys.exit(1)

fixed_count = 0
for info_path in sorted(info_files):
    # Extract physical device_id from filename: info.json.5 -> 5
    match = re.search(r'info\.json\.(\d+)$', info_path)
    if not match:
        continue

    physical_device_id = int(match.group(1))

    # Get rank ID from mapping, or fallback to physical device ID
    rank_id = device_to_rank.get(physical_device_id, physical_device_id)

    try:
        with open(info_path, 'r') as f:
            data = json.load(f)

        old_rank = data.get('rank_id', -1)
        data['rank_id'] = rank_id

        with open(info_path, 'w') as f:
            json.dump(data, f, separators=(',', ':'))

        print(f"Fixed {info_path}: rank_id {old_rank} -> {rank_id} (physical device {physical_device_id})")
        fixed_count += 1
    except Exception as e:
        print(f"Error fixing {info_path}: {e}", file=sys.stderr)
        sys.exit(1)

if fixed_count == 0:
    print("Error: Failed to fix any files", file=sys.stderr)
    sys.exit(1)

print(f"Successfully fixed {fixed_count} files")
sys.exit(0)
`

	msprofPodYAMLTemplate = `apiVersion: v1
kind: Pod
metadata:
  name: %s
  namespace: %s
spec:
  nodeName: %s
  restartPolicy: Never
  containers:
  - name: msprof
    image: %s
    command: ["sleep", "infinity"]
    resources:
      requests:
        %s: "%d"
      limits:
        %s: "%d"
    securityContext:
      privileged: true
    volumeMounts:
    - name: dcmi
      mountPath: /usr/local/dcmi
    - name: npu-smi
      mountPath: /usr/local/bin/npu-smi
    - name: lib64
      mountPath: /usr/local/Ascend/driver/lib64
    - name: version
      mountPath: /usr/local/Ascend/driver/version.info
    - name: ascend-install-info
      mountPath: /etc/ascend_install.info
    - name: hccn-conf
      mountPath: /etc/hccn.conf
  volumes:
  - name: dcmi
    hostPath:
      path: /usr/local/dcmi
  - name: npu-smi
    hostPath:
      path: /usr/local/bin/npu-smi
      type: File
  - name: lib64
    hostPath:
      path: /usr/local/Ascend/driver/lib64
  - name: version
    hostPath:
      path: /usr/local/Ascend/driver/version.info
      type: File
  - name: ascend-install-info
    hostPath:
      path: /etc/ascend_install.info
      type: File
  - name: hccn-conf
    hostPath:
      path: /etc/hccn.conf
      type: File
`

	msprofExecScriptTemplate = `
set -e
set -o pipefail

echo "=== Setting up environment ==="
if [ ! -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    echo "Error: set_env.sh not found" >&2
    exit 1
fi
source /usr/local/Ascend/ascend-toolkit/set_env.sh

echo "=== Installing torch and torch-npu ==="
pip3 install torch torch-npu

echo "=== Collecting profiling data ==="
TORCHRUN=$(which torchrun 2>/dev/null)
if [ -z "$TORCHRUN" ]; then
    echo "Error: torchrun not found, please check torch installation" >&2
    exit 1
fi
msprof --application="$TORCHRUN --nproc_per_node=%d /tmp/benchmark.py" --output=/tmp/prof_data --hccl=on

echo "=== Fixing rank_id ==="
python3 /tmp/fix_rank.py /tmp/prof_data

echo "=== Locating msprof.py ==="
MSPROF_PY=$(find /usr/local/Ascend -name "msprof.py" -type f 2>/dev/null | head -1)
if [ -z "$MSPROF_PY" ]; then
    echo "Error: msprof.py not found" >&2
    exit 1
fi
echo "Found msprof.py at: $MSPROF_PY"

echo "=== Importing cluster data ==="
python3 $MSPROF_PY import -dir=/tmp/prof_data --cluster

echo "=== Exporting summary ==="
python3 $MSPROF_PY export summary -dir=/tmp/prof_data 2>&1
`
)
