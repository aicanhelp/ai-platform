# sig-ub-enable

UB Enable SIG aims to build an open, efficient AI cloud-native enablement system oriented to UB supernodes. This SIG focuses on three major areas: **K8s resource scheduling enhancement based on supernodes**, **application scenario acceleration**, and **developer toolchain**. For key capabilities such as high-speed communication and resource pooling of UB supernodes, we explore new technical paradigms in resource utilization and business performance improvement, promoting community technology evolution in this field.

## Work Scope

- **K8s resource scheduling enhancement based on supernodes**: Build modular, high-performance, K8s ecosystem-compatible cloud-native components, enabling container workloads to use UB supernode capabilities.
- **Application scenario acceleration based on supernodes**: For key application scenarios, explore innovative architectures and optimization paradigms based on UB supernodes, bringing significant improvements in performance and efficiency.
- **Supernode-specific development toolchain**: Provide UB supernode developer full lifecycle enablement services, achieving UB affinity development/compilation/testing/performance analysis on developer machines.

## Technology Collaboration

### Collaboration with other openFuyao SIG capabilities

- **Resource scheduling enablement**: Collaborate with container orchestration engine SIG to extend K8s scheduling component capabilities, sense UB supernode shared topology relationships, and perform scheduling selection for related Pods.
- **Application scenario acceleration based on supernodes**: Collaborate with AI inference SIG, big data SIG, and container orchestration engine SIG to explore UB application acceleration innovation paradigms.
- **Supernode-specific development toolchain**: Collaborate with infrastructure SIG to continuously optimize developer toolchain and software engineering capabilities for UB supernode enablement scenarios.

#### Collaboration with industry open-source technology solutions

- **K8s resource scheduling enhancement based on supernodes**: Kubernetes native diversified resource management and scheduling solutions.
- **Application scenario acceleration based on supernodes**: Collaborate with CXL, UALink bus, and various scenario application upstream communities.
- **Supernode-specific development toolchain**: Build UB supernode simulation capabilities based on open-source project QEMU; add UB support based on Lmbench memory testing project, etc.

## 2026 Planning

### 1. K8s resource scheduling enhancement based on supernodes

- **KT1: Container cluster release UB high bandwidth/low latency capabilities**
	- (2026 Q1) UB CNI plugin (Villus 2.0): Based on URMA language communication and UB network hardware, build UB network card and DPU network card CNI plugin, connecting container network with UB network, achieving low latency high-performance communication.
    - (2026 Q2) UB CSI plugin: Through flexible UB peer-to-peer architecture SSI direct connection, build SSI storage CSI plugin, container storage dynamic allocation and mounting.

- **KT2: Enhance K8s scheduling based on UB, improve resource utilization**
	- (2026 Q1) Container live migration: Based on UB large bandwidth/memory pooling and kernel state CRI capabilities, superimpose K8s topology-aware scheduling, Pod migration interruption time less than 0.5s.
    - (2026 Q2) Topology-aware scheduling: Pool resource quantity and metrics collection, topology-aware scheduling, bin packing optimization and other affinity scheduling algorithms based on load, balancing system resources.
      - (2026 Q2) Remote-Fork: Based on UB shared memory implemented Remote-Fork fast container startup, achieving hundred-millisecond level container cold startup, improving system availability, elasticity efficiency and resource utilization.

### 2. Application scenario acceleration based on supernodes

- **KT1: UB memory pooling and communication improve distributed KVCache performance, bringing AI inference TTFT optimization**
   - (2026 Q1) KVCache mooncake high availability, improve stability; optimize cache mechanism based on UB memory pooling, TTFT reduced by 20%.
   - (2026 Q1) Provide high-performance metric subscription-publish data flow, key metric latency < 100ms, improve observability/decision/implementation end-to-end efficiency.
   - (2026 Q2) Mainstream inference framework integration and compatibility, integrate into mooncake Transfer Engine shared memory architecture (to be planned).
   - (2026 Q2) Distribution acceleration: Weight, image, data hardware enablement acceleration.

- **KT2: UB memory pooling technology improve big data Spark performance**
    - Build Spark Shuffle, HDFS RW performance improvement, compute task end-to-end performance improvement xx%+, compute resource saving about xx% (in technical evaluation).

- **KT3: UB distributed pooling architecture, bringing cloud-native database multi-master architecture (similar to: Alibaba PolarDB @ CXL)**.

- **KT4: UB large capacity KVCache and RPC, improve E2E advertising recommendation latency**.

- **KT5: UB memory semantics bring low latency high-speed communication, replacing traditional communication networks**.

- **Other application scenarios**
    - Supernode large memory virtual machine and memory oversubscription.
    - Point-to-point data direct connection, distributed locks.
    - ...

### 3. Supernode-specific development toolchain

- **2026 Q1**: Support personal local UB application development, compilation, experience UB simulation service.
   - (2026.01) Provide personal developer UB application development environment quick installation solution.
      + Provide UB development image (openEuler + UB development package).
      + Provide VSCode + UB development image remote development solution.
      + Support UB application local compilation (ARM->ARM, X86->ARM).
      + Support UB application cloud compilation and code checking (openFuyao community pipeline).

   - (2026.03) Provide cloud UB simulation service (Beta), support personal developers remote access to simulation service, import UB application binary, online functional testing and verification.
- **2026 Q2**: Support UB simulation service, release UB DT framework.
   - Provide cloud UB simulation service capability, support personal developers one-click application for UB simulation service, perform UB application testing.
   - Provide lightweight UB DT framework (Beta), support high-frequency UB API interface Mock capability.
- **2026 Q3**: UB simulation service experience optimization, support UB performance analysis, AI heuristic programming experience.
   - Provide UB simulation group network enhancement capability, support personal developers flexible group networking, flexible node scaling.
   - Provide UB DT framework enhancement capability, support general scenario one-click Mock.
   - Provide UB performance engineering capability (Beta), support UB application performance testing and analysis.
   - Provide AI heuristic programming based on CodeArts (Beta), provide UB API recommendation and automatic generation capability.
- **2026 Q4**: AI heuristic programming capability enhancement, release UB one-stop developer toolchain.
   - Release AI heuristic programming based on CodeArts, improve UB API recommendation accuracy, support best practice paradigm code generation, expand API recommendation scenarios.

## Current Progress (251129)

### 1. openFuyao ub-enable-sig operation

- Completed SIG group operation work organization and idea discussion.
- Completed community code repository creation and pipeline configuration: ub-enable-sig, ub-enable-quickstart.
- Supernode bus documentation system (currently completed Lingxi UB bus technical material listing, CXL/NVLink and others to be listed).
   - Technical white paper and specification reference.
   - Quick development guide, including environment configuration and demo code examples.
   - Scenario-based code examples, simplified based on Spark/Flink/Socket 3 scenarios.
- 26 year planning: in progress, TC discussion to be held in December.

### 2. K8s resource scheduling enhancement based on supernodes

- K8s manage UB memory borrowing, network, storage and installation deployment design and development completed.
- K8s UB enhanced scheduling solution:
   - Completed scheduling container and bin packing optimization 2 scenario designs, under development.
   - Completed live migration design (2s -> 0.5s), function under development.
   - Currently in incubation, community open-source release rhythm under confirmation (strive for 1230, latest 0330).

### 3. Application scenario acceleration based on supernodes (AI inference and big data)

- **AI inference UB enhancement**
   - Mooncake enhancement based on memory sharing design and development completed, Mooncake Store TE based on shared memory transformation design completed, and community TC review completed.
   - Currently based on community co-construction, completed partial testing, depends on Lingxi UB development package, community open-source release rhythm under confirmation (expected 0330).
- **Big data UB enhancement**
   - Uniffle software integration HCOM V1 code development and simulation verification completed, HCOM V2 Java API development code completed, testing pending.
   - Flink Shuffle feature in HCOM RDMA, StateBackend, CheckPoint scenarios design, development and verification completed, expected to bring 15% performance improvement.
   - Currently in incubation, community open-source release rhythm under confirmation (expected 0330).

### 4. Supernode-specific development toolchain

- UB development process打通: Based on UB Mooncake and UB K8s open-source scenario penetration, including openEuler base build image, openFuyao CI/CD pipeline.
- UB development package dependencies: UB S Comm open-sourced in OpenEuler community, currently ubs-comm/ubs-engine etc. have uploaded code, others to be submitted by end of month (depending on situation, there may be some delay), UB development package to be released with openEuler 24.03 LTS SP3 in 1230.
- UB simulation service and laboratory: Promoting Lingxi UB simulation service opening.

## Progress (2026.4.10)

### ub-enable sig open-sourced two UB enablement projects: ubs-openstack-enable, ubs-k8s-enable

| Project | Project Introduction | Local Development Build | Unit Testing | Configuration Description | API Interface | Scenario Examples | Pipeline Build | Installation Deployment Guide |
|--|--|--|--|--|--|--|--|--|--|
|ubs-openstack-enable  |  ready | ready | ready | ready | ready | ready | ready | ready | 
|ubs-k8s-enable|  ready | ready | ready | ready | ready | ready | ready | ready | 

### Next Steps Plan

Target ubs-openstack-enable and ubs-k8s-enable two project developer journeys, from code download -> compilation -> development -> unit testing -> compilation packaging -> functional testing -> code submission, achieving UB project out-of-the-box usability goal.
