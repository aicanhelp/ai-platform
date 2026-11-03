# vNPU

## Overview

The vNPU project focuses on building NPU compute slicing and dynamic scheduling capabilities on top of the container platform for Ascend NPUs. It supports multi-container device sharing while providing memory and AICore resource isolation to improve resource utilization. By offering diverse scheduling strategies, diverse compute slicing modes, and observability capabilities, it delivers an integrated, easy-to-use Ascend NPU compute access and management solution.

## Key Features

- Diverse compute slicing modes
  - Soft slicing: User-space virtualization based on CANN Runtime API interception, offering finer and more flexible slicing precision
    - Maximum 1-to-20 slicing, with AICore slicing granularity as small as 1% and memory slicing granularity as small as 1Gi
    - AICore and memory isolation. AICore uses time-sharing multiplexing with non-strict isolation; compute isolation fluctuation is approximately less than 10% of a full card. Memory is strictly isolated
    - Supports limiting only memory without limiting AICore; each container can be configured independently
    - Multiple AICore scheduling strategies, supporting fixed quota mode, elastic mode, and preemptive mode
    - Performance overhead less than 5%
  - Hard slicing: Virtualization based on Ascend HDK, splitting an NPU into multiple vNPUs mounted to containers for use
- Diverse cluster scheduling strategies
  - NPU resource scheduling plugin extended from volcano, supporting resource sharing scheduling
  - Supports binpack scheduling
  - Supports full-card scheduling; on the same node, some containers use full cards while others share a single card
  - [DRA-based scheduling](https://gitcode.com/openFuyao/npu-dra-plugin)
- Observability
  - Supports collecting vNPU-related monitoring metrics and reporting them to Prometheus

## Constraints and Limitations

**Soft slicing constraints and limitations:**

1. Currently, a container can apply for at most one vNPU.
2. Currently, a container can only have one process using a vNPU.
3. Only supports slicing and limiting AICore and memory capacity; does not support slicing control of AI CPU, VPC, VDEC, JEPGD, and other cores.
4. Business containers must not be privileged containers.

## Quick Start

### Code Structure

- volcano-xpu-plugin: volcano plugin that implements NPU virtualization resource scheduling.
- xpu-device-plugin: device plugin that implements NPU resource reporting and allocation.
- client_update: responsible for deploying the soft slicing interception library.
- xpu-exporter: responsible for reporting vNPU metrics.
- ci: build and compilation scripts.
- charts: files related to vNPU component deployment.

### Build & Deployment

Please refer to the [User Guide](https://gitcode.com/openFuyao/vNPU/blob/master/docs/user-guide.md).

### Pod Using NPU

A business Pod only needs to declare the required vNPU resources without modifying the business code. The vNPU scheduler and device plugin automatically schedule the Pod to an appropriate node and NPU device and mount the device.

For example, a Pod using soft slicing declares 1 vNPU, 50% AICore, and 8Gi memory:

```yaml
spec:
    schedulerName: volcano
    containers:
        resources:
          limits:
            huawei.com/vnpu-number: 1  
            huawei.com/vnpu-cores: 50  
            huawei.com/vnpu-memory.1Gi: 8 
          requests:
            huawei.com/vnpu-number: 1
            huawei.com/vnpu-cores: 50
            huawei.com/vnpu-memory.1Gi: 8
```

Using hard slicing, requesting 50% AICore:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: testpod
  annotations:
    huawei.com/vnpu-mode: "hard"
    huawei.com/vnpu-hard-aicpu-level: "low"
spec:
    schedulerName: volcano
    containers:
        resources:
          limits:
            huawei.com/vnpu-number: 1  
            huawei.com/vnpu-cores: 50  # AICore percentage
          requests:
            huawei.com/vnpu-number: 1
            huawei.com/vnpu-cores: 50
```

A Pod declaring the use of two full cards:

```yaml
spec:
    schedulerName: volcano
    containers:
        resources:
          limits:
            huawei.com/vnpu-number: 2 
          requests:
            huawei.com/vnpu-number: 2
```

For more usage methods, please refer to the [User Guide](https://gitcode.com/openFuyao/vNPU/blob/master/docs/user-guide.md).

## FAQ

**Supported NPU Models**

| Chip Model | Soft Slicing | Hard Slicing | HDK Version Compatibility | CANN Version Compatibility (Soft Slicing) |
| ---------- | ------------ | ------------ | ------------------------- | ------------------------------------------ |
| 310P3      | Supported    | Supported    | 25.5.0 and above          | 8.5.0, 9.1.0                              |
| 910B4      | Supported    | Supported    | 25.5.0 and above          | 8.5.0, 9.1.0                              |
| 910C       | Not supported| Not supported| NA                        | NA                                         |

Note: The chip models and software version combinations listed above have been thoroughly verified. Software and hardware not within this range are unverified and may have compatibility issues.

**Supported volcano Versions**

Currently, volcano 1.15.0 is supported. Newer versions are pending planning.

## Contribution and Support

### Contribution Guide

Developers are welcome to participate in project optimization and iteration. To contribute code, please submit through the following methods:

- Visit the GitCode repository: https://gitcode.com/openFuyao/vNPU.git.
- Submit a Pull Request with a detailed feature description or issue fix description.
- Submit an Issue to report bugs encountered during use or feature requests.

### Support Channels

If you encounter issues during deployment or use, please open an Issue on the GitCode project page. We will respond promptly and provide technical support. Thank you for your use and feedback, which helps the vNPU project continue to improve!
