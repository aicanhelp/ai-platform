# Agent Sandbox Checkpoint and Restore

## Feature Introduction

Checkpoint and Restore are sandbox snapshot and recovery capabilities provided by OpenKruise Agents. With Checkpoint, you can capture a complete snapshot of a running Sandbox including memory and filesystem, and then use Restore to quickly launch a new Sandbox instance from an existing snapshot, enabling preservation and reuse of business environments.

### Application Scenarios

- **AI Agent Session Recovery**: An Agent creates a snapshot after reaching an intermediate state, and can later restore from the snapshot to continue execution, avoiding repeated initialization and Token consumption.
- **Environment Template Reuse**: Create a snapshot from a Sandbox environment with installed dependencies and configurations, then quickly clone multiple Sandboxes with the same environment.
- **Development and Debugging**: Create snapshots at key nodes, allowing quick rollback to a known state when issues arise.
- **Elastic Scaling**: Combined with SandboxSet warm-up pools, quickly scale out Sandbox instances with business state based on snapshots.

### Capability Scope

- Create full snapshots (memory + filesystem) of running Sandboxes.
- Support multiple storage backends (Local, S3, MinIO, CSI, NFS) for persistent snapshot archiving.
- Quickly restore new Sandbox instances from snapshots; restored Sandboxes contain the complete memory and filesystem state at the time of the snapshot.
- Support automatic snapshot expiration (TTL) and manual deletion (automatic remote archive cleanup).
- Trigger Checkpoint and Restore via E2B SDK compatible API.

### Highlight Features

- **Declarative Management**: Declaratively manage snapshot lifecycle through Kubernetes CRD (Checkpoint), seamlessly integrated with the K8s ecosystem.
- **VM-level Snapshot**: Based on Kata Containers VM-level isolation, snapshots contain the complete virtual machine state; after restoration, processes, network connections, etc. can maintain their original state.
- **Multiple Storage Backends**: Support five storage backends (Local, S3, MinIO, CSI, NFS) to meet different persistence requirements.
- **Data Integrity Verification**: Snapshot artifacts include SHA256 checksums, automatically verified during restoration to ensure data has not been tampered with.
- **SandboxTemplate Linkage**: When creating a Checkpoint, a SandboxTemplate is automatically generated, eliminating the need to manually rebuild Pod configurations during Restore.
- **E2B SDK Compatible**: Checkpoint ID can be used directly through the E2B SDK to trigger restoration, reducing integration costs.

### Basic Concepts

**Table 1** Basic Concepts

| Concept | Description |
|---------|-------------|
| Checkpoint | The CRD carrier for snapshot operations, defining the target Sandbox, persistence content, TTL, and other parameters, tracking the snapshot lifecycle (Pending→Creating→Succeeded/Failed). |
| Restore | The process of restoring a new Sandbox from an existing Checkpoint, triggered by setting the `agents.kruise.io/restore-from` annotation on a Sandbox. |
| SandboxTemplate | The Pod template associated with the snapshot, automatically generated when creating a Checkpoint, saving the source Sandbox's PodSpec, VolumeClaimTemplates, etc., used to construct a new Sandbox during Restore. |
| sandbox-node-agent | The node-level DaemonSet component responsible for communicating with containerd/kata-shim to execute snapshot creation, archive upload, snapshot download, and other operations. |
| ArchiveLocation | The URI identifier of the snapshot archive in the storage backend (e.g., `local:///var/lib/checkpoints/...`, `s3://bucket/...`). |

### Implementation Principles

#### Checkpoint Creation Flow

The Checkpoint Controller communicates with the sandbox-node-agent on the node via gRPC to complete snapshot creation.

```
User creates Checkpoint CR
       │
       ▼
Controller: Parse source Sandbox and Pod
       │
       ▼
Controller: Detect runtime type (only kata-qemu supported)
       │
       ▼
Controller: Preflight check node-agent availability
       │
       ▼
Controller: gRPC call node-agent CreateCheckpoint
       │
       ▼
node-agent: containerd→kata-shim creates snapshot archive
       │
       ▼
node-agent: Upload archive to storage backend + write metadata.json
       │
       ▼
Controller: Poll GetCheckpointStatus until COMPLETED
       │
       ▼
Checkpoint Phase→Succeeded, ArchiveLocation populated
```

#### Restore Recovery Flow

```
User creates Sandbox (with restore-from annotation)
       │
       ▼
Sandbox Controller: Detect restore-from annotation
       │
       ▼
Controller: Get Checkpoint CR, validate status (Succeeded+ArchiveLocation+RuntimeType)
       │
       ▼
Controller: Build Restore Pod, inject archive location, runtime type, etc. annotations
       │
       ▼
kata-shim: Read Pod annotations at startup, gRPC call node-agent DownloadSnapshot
       │
       ▼
node-agent: Download archive from storage backend, verify SHA256, decompress to /var/lib/kata-restore/<pod-uid>/
       │
       ▼
kata-shim: Restore VM from snapshot, Pod enters Running
       │
       ▼
Controller: Sandbox Phase→Running, asynchronously clean up node local snapshot files
```

### Relationship with Related Features

**Table 2** Relationship with Related Features

| Feature | Relationship |
|---------|--------------|
| Sandbox | The target object of Checkpoint; the product of Restore is a new Sandbox. |
| SandboxTemplate | Automatically generated when creating a Checkpoint, saves the source Sandbox's Pod template; during Restore, the Controller gets PodSpec from SandboxTemplate. |
| SandboxSet | Can be combined with Checkpoint to warm up Sandbox pools, accelerating Sandbox acquisition after Restore. |
| SandboxClaim | Can declaratively acquire a restored Sandbox. |
| sandbox-node-agent | The node-level execution component for Checkpoint/Restore, providing snapshot creation, archive upload, snapshot download, and other services. |

### Related Examples

You can directly use the Checkpoint ID to trigger restoration via the E2B SDK:

```python
from e2b import Sandbox

# Set templateID to the Checkpoint name, automatically trigger Restore
sandbox = Sandbox(template="my-checkpoint-name")
sandbox.wait_ready()

print(f"Sandbox ID: {sandbox.sandbox_id}")
```

sandbox-manager automatically completes the following steps internally:

1. Detects that `templateID` corresponds to a Succeeded Checkpoint.
2. Finds the associated SandboxTemplate.
3. Creates a Sandbox CR (with `restore-from` annotation).
4. Controller takes over the subsequent Restore flow.
5. Waits for the Sandbox to enter the Running state.

## Installation

### Prerequisites

1. The Kubernetes cluster is v1.26 or above.
2. OpenKruise Agents (including agent-sandbox-controller, sandbox-manager) have been deployed.
3. The Sandbox uses the Kata Containers runtime (`kata-qemu`).
4. The Checkpoint and Restore Feature Gate has been enabled.

### Start Installation

1. Enable Feature Gate

    The Checkpoint and Restore features are disabled by default and need to be manually enabled in the controller startup parameters:

    ```
    --feature-gates=SandboxCheckpoint=true,SandboxRestore=true
    ```

    Modify the existing deployment:

    ```bash
    kubectl edit deploy -n sandbox-system agent-sandbox-controller
    ```

    Find the `--feature-gates` parameter in the container's `args` and add `SandboxCheckpoint=true,SandboxRestore=true`.

2. Deploy sandbox-node-agent

    sandbox-node-agent runs as a DaemonSet on each node that needs to execute snapshots and restorations. Choose an online or offline deployment method based on your environment.

    **Method 1: Online Deployment (Recommended)**

    In an environment with access to the public network and image registry, directly use the one-click deployment script to complete the deployment:

    ```bash
    ./hack/deploy.sh
    ```

    The script automatically completes ServiceAccount creation, image pulling, and DaemonSet deployment. After deployment, skip to the "Verify Deployment" step.

    **Method 2: Offline Deployment**

    In an intranet or non-public network environment, you need to manually build the image and deploy.

    2.1 Build the sandbox-node-agent image:

    ```bash
    go build -o bin/node-agent ./cmd/node-agent
    docker build -t your-registry/node-agent:latest -f Dockerfile.node-agent .
    docker push your-registry/node-agent:latest
    ```

    2.2 Create ServiceAccount:

    ```bash
    kubectl apply -f config/rbac/node_agent_sa.yaml
    ```

    2.3 Modify the image address in `config/node-agent/daemonset.yaml` and deploy:

    ```yaml
    containers:
      - name: agent
        image: your-registry/node-agent:latest
    ```

    ```bash
    kubectl apply -f config/node-agent/daemonset.yaml
    ```

    **Verify Deployment**

    ```bash
    kubectl get daemonset sandbox-node-agent -n sandbox-system
    kubectl get pods -n sandbox-system -l app=sandbox-node-agent -o wide
    kubectl logs -n sandbox-system -l app=sandbox-node-agent --tail=20
    ```

    After successful startup, you should see:

    ```
    Starting node agent port=9090 workDir=/var/lib/node-agent storageBackend=local ...
    Node agent listening port=9090
    ```

    **Table 3** node-agent startup parameter descriptions

    | Parameter | Default Value | Description |
    |-----------|---------------|-------------|
    | `--grpc-port` | `9090` | gRPC service port. |
    | `--work-dir` | `/var/lib/node-agent` | Working directory. |
    | `--storage-backend` | `local` | Storage backend type: `local`, `s3`, `minio`, `csi`, `nfs`. |
    | `--checkpoint-dir` | - | Local snapshot directory (bypasses storage-backend when set). |
    | `--containerd-mode` | `real` | containerd mode: `real` (connect to socket) or `mock` (for testing). |
    | `--max-concurrent-downloads` | `2` | Maximum concurrent snapshot downloads. |
    | `--snapshot-root` | `/var/lib/kata-restore` | Kata recovery convention path root directory. |

3. Configure Storage Backend

    node-agent supports multiple storage backends to save snapshot archives, specified via the `--storage-backend` parameter, and each backend is configured through environment variables. When using `--storage-backend`, you must **not set** the `--checkpoint-dir` parameter. If `--checkpoint-dir` is also set, node-agent will use a direct-write path and completely bypass the storage-backend configuration.

    3.1 Local Backend

    **Table 4** Local backend environment variables

    | Environment Variable | Default Value | Required | Description |
    |----------------------|---------------|----------|-------------|
    | `STORAGE_LOCAL_BASE_DIR` | `/var/lib/checkpoints/local` | No | Local storage root directory |

    Run the following command to configure Local backend environment variables:

    ```bash
    kubectl set env daemonset/sandbox-node-agent -n sandbox-system \
      STORAGE_LOCAL_BASE_DIR=/var/lib/checkpoints
    ```

    Checkpoint artifact archive path: `<STORAGE_LOCAL_BASE_DIR>/<namespace>/<checkpointID>/checkpoint.tar.gz`

    3.2 CSI Backend

    **Table 5** CSI backend environment variables

    | Environment Variable | Default Value | Required | Description |
    |----------------------|---------------|----------|-------------|
    | `STORAGE_CSI_VOLUME_PATH` | - | Yes | CSI volume mount path |

    Run the following command to configure CSI backend environment variables:

    ```bash
    kubectl set env daemonset/sandbox-node-agent -n sandbox-system \
      STORAGE_CSI_VOLUME_PATH=/mnt/csi-checkpoints
    ```

    Checkpoint artifact archive path: `<STORAGE_CSI_VOLUME_PATH>/checkpoints/<namespace>/<checkpointID>/checkpoint.tar.gz`

    3.3 NFS Backend

    **Table 6** NFS backend environment variables

    | Environment Variable | Default Value | Required | Description |
    |----------------------|---------------|----------|-------------|
    | `STORAGE_NFS_MOUNT_PATH` | - | Yes | NFS mount path (must be mounted on all nodes) |

    Run the following command to configure NFS backend environment variables:

    ```bash
    kubectl set env daemonset/sandbox-node-agent -n sandbox-system \
      STORAGE_NFS_MOUNT_PATH=/mnt/nfs-checkpoints
    ```

    Checkpoint artifact archive path: `<STORAGE_NFS_MOUNT_PATH>/checkpoints/<namespace>/<checkpointID>/checkpoint.tar.gz`

    3.4 S3 Backend

    **Table 7** S3 backend environment variables

    | Environment Variable | Default Value | Required | Description |
    |----------------------|---------------|----------|-------------|
    | `STORAGE_S3_ENDPOINT` | - | Yes | S3 service endpoint |
    | `STORAGE_S3_ACCESS_KEY_ID` | - | Yes | Access key ID |
    | `STORAGE_S3_SECRET_ACCESS_KEY` | - | Yes | Access key |
    | `STORAGE_S3_BUCKET_NAME` | - | Yes | Bucket name |
    | `STORAGE_S3_REGION` | `us-east-1` | No | Region |
    | `STORAGE_S3_USE_SSL` | `true` | No | Whether to use SSL |
    | `STORAGE_S3_PREFIX` | `checkpoints` | No | Object key prefix |

    Run the following command to configure S3 backend environment variables:

    ```bash
    kubectl set env daemonset/sandbox-node-agent -n sandbox-system \
      STORAGE_S3_ENDPOINT=s3.amazonaws.com \
      STORAGE_S3_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE \
      STORAGE_S3_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY \
      STORAGE_S3_BUCKET_NAME=my-checkpoints
    ```

    Checkpoint artifact archive path: `s3://<bucket>/<prefix>/<namespace>/<checkpointID>/checkpoint.tar.gz`

    3.5 MinIO Backend

    **Table 8** MinIO backend environment variables

    | Environment Variable | Default Value | Required | Description |
    |----------------------|---------------|----------|-------------|
    | `STORAGE_MINIO_ENDPOINT` | - | Yes | MinIO service endpoint |
    | `STORAGE_MINIO_ACCESS_KEY_ID` | - | Yes | Access key ID |
    | `STORAGE_MINIO_SECRET_ACCESS_KEY` | - | Yes | Access key |
    | `STORAGE_MINIO_BUCKET_NAME` | - | Yes | Bucket name |
    | `STORAGE_MINIO_REGION` | `us-east-1` | No | Region |
    | `STORAGE_MINIO_USE_SSL` | `false` | No | Whether to use SSL |
    | `STORAGE_MINIO_PREFIX` | `checkpoints` | No | Object key prefix |

    Run the following command to configure MinIO backend environment variables:

    ```bash
    kubectl set env daemonset/sandbox-node-agent -n sandbox-system \
      STORAGE_MINIO_ENDPOINT=minio:9000 \
      STORAGE_MINIO_ACCESS_KEY_ID=minioadmin \
      STORAGE_MINIO_SECRET_ACCESS_KEY=minioadmin \
      STORAGE_MINIO_BUCKET_NAME=checkpoints
    ```

    Checkpoint artifact archive path: `minio://<bucket>/<prefix>/<namespace>/<checkpointID>/checkpoint.tar.gz`
    
4. Deploy Kata

    4.1 Install Kata Containers
    ```bash
    # Method 1: Use kata-deploy (recommended)
    KATA_VERSION="3.30.0"
    kubectl apply -f "https://github.com/kata-containers/kata-containers/releases/download/${KATA_VERSION}/kata-deploy-base.yaml"
    kubectl apply -f "https://github.com/kata-containers/kata-containers/releases/download/${KATA_VERSION}/kata-deploy-stable.yaml"

    # Wait for kata-deploy to complete
    kubectl -n kube-system wait --for=condition=Ready pod -l name=kata-deploy --timeout=300s

    # Verify RuntimeClass
    kubectl get runtimeclass kata-qemu
    # Expected output: NAME       HANDLER     AGE
    #                  kata-qemu  kata-qemu  ...

    # Method 2: Manual installation (if kata-deploy is unavailable)
    # Download precompiled package
    wget "https://github.com/kata-containers/kata-containers/releases/download/${KATA_VERSION}/kata-static-${KATA_VERSION}-x86_64.tar.xz"
    sudo tar -xvf kata-static-${KATA_VERSION}-x86_64.tar.xz -C /

    # Verify kata installation
    /opt/kata/bin/kata-runtime kata-env | head -20
    
    ```
     4.2 Configure containerd
    ```bash
    # Ensure containerd configuration supports kata
    sudo mkdir -p /etc/containerd

    # If config.toml does not exist yet, generate default configuration
    if [ ! -f /etc/containerd/config.toml ]; then
        sudo containerd config default | sudo tee /etc/containerd/config.toml
    fi

    # Add kata runtime configuration (if kata-deploy did not auto-configure)
    sudo tee -a /etc/containerd/config.toml <<'EOF'

    [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.kata-qemu]
      runtime_type = "io.containerd.kata-qemu.v2"
      privileged_without_host_devices = true
      runtime_path = "/opt/kata/bin/containererd-shim-kata-v2"
      pod_annnotations = ["io.katacontainers.*", "agents.kruise.io/*"]

    [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.kata-qemu.options]
      ConfigPath = "/opt/kata/share/defaults/kata-containers/configuration-qemu.toml"
    EOF
    
    # Restart containerd
    sudo systemctl restart containerd

    # Verify kata runtime is available
    crictl --runtime-endpoint unix:///var/run/containerd/containerd.sock info 2>/dev/null | head -5

    ```
    4.3 Compile custom kata-shim
    ```bash
    # Clone kata-containers (use the branch with Checkpoint/Restore implementation)
    cd /opt
    git clone <your-kata-repo-url> kata-containers
    cd kata-containers
    git checkout kata-shim-v2 # Branch with Checkpoint/Restore implementation

    # Generate config-settings.go
    cd src/runtime
    cp pkg/katautils/config-settings.go.in pkg/katautils/config-settings.go
    sed -i \
        -e 's|@RUNTIME_NAME@|containerd-shim-kata-v2|g' \
        -e 's|@PROJECT_NAME@|Kata Containers|g' \
        -e 's|@PROJECT_TYPE@|kata|g' \
        -e 's|@PROJECT_URL@|https://github.com/kata-containers|g' \
        -e 's|@PROJECT_ORG@|kata-containers|g' \
        -e 's|@PKGRUNDIR@|/run/kata-containers|g' \
        -e 's|@COMMIT@|HEAD|g' \
        -e 's|@VERSION@|3.30.0|g' \
        -e 's|@CONFIG_PATH@|/opt/kata/share/defaults/kata-containers/configuration.toml|g' \
        -e 's|@SYSCONFIG@|/etc/kata-containers/configuration.toml|g' \
        pkg/katautils/config-settings.go

    # Compile
    GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build \
        -o /tmp/containerd-shim-kata-v2 \
        ./cmd/containerd-shim-kata-v2

    echo "Build completed: /tmp/containerd-shim-kata-v2"

    # Verify binary contains Checkpoint/Restore related symbols
    strings /tmp/containerd-shim-kata-v2 | grep -E "Checkpointing sandbox|Detected restore snapshot|restore-ready"
    # Expected output:
    #   Checkpointing sandbox
    #   Detected restore snapshot, will restore from snapshot
    #   restore-ready

     ```
    4.4 Install custom kata-shim
    ```bash
    # Backup original shim
    sudo cp /opt/kata/bin/containerd-shim-kata-v2 /opt/kata/bin/containerd-shim-kata-v2.bak
    sudo cp /usr/local/bin/containerd-shim-kata-v2 /usr/local/bin/containerd-shim-kata-v2.bak 2>/dev/null || true

    # Replace with custom shim
    sudo cp /tmp/containerd-shim-kata-v2 /opt/kata/bin/containerd-shim-kata-v2
    sudo cp /tmp/containerd-shim-kata-v2 /usr/local/bin/containerd-shim-kata-v2
    sudo chmod +x /opt/kata/bin/containerd-shim-kata-v2
    sudo chmod +x /usr/local/bin/containerd-shim-kata-v2

    # Verify version
    /opt/kata/bin/containerd-shim-kata-v2 --version 2>&1 | head -3

     ```

## Using Checkpoint and Restore

### Prerequisites

- The above installation steps have been completed (Feature Gate enabled, sandbox-node-agent deployed).
- A Sandbox in `Running` state already exists (for creating a Checkpoint).
- For Restore: A Checkpoint in `Succeeded` state already exists, with `ArchiveLocation` and `RuntimeType`.
- For Restore: The corresponding SandboxTemplate has been created (automatically generated when creating a Checkpoint).

### Background Information

- Checkpoint is declaratively managed through the Checkpoint CR (short name `cp`); the Controller communicates with the sandbox-node-agent on the node via gRPC.
- Restore is triggered by setting the `agents.kruise.io/restore-from` annotation on a Sandbox; the Controller automatically creates a Restore Pod with archive information.
- Snapshot creation is performed by kata-shim at the VM level; archive files are uploaded by node-agent to the configured storage backend.
- During restoration, kata-shim downloads the snapshot to the node's local `/var/lib/kata-restore/<pod-uid>/` directory via node-agent, then restores the VM from the snapshot.

### Usage Limitations

- Only the `kata-qemu` runtime is supported; other RuntimeClasses are not supported.
- Cross-VMM type Restore is not supported (e.g., a QEMU snapshot cannot be restored to Firecracker).
- Checkpoint and Restore must be within the same namespace.
- The Checkpoint creation timeout is 10 minutes.
- When snapshot download fails, the Pod remains in Pending state and the Sandbox continues in Restoring state; it will not automatically downgrade to cold start.
- When using `--storage-backend`, you cannot simultaneously set `--checkpoint-dir`; otherwise, the storage backend configuration will be bypassed.

### Operation Steps

#### Creating a Checkpoint

1. Confirm that the target Sandbox is in `Running` state:

    ```bash
    kubectl get sandbox -n default
    ```

2. Execute `kubectl apply -f checkpoint.yaml` to create the Checkpoint resource:

    ```yaml
    # checkpoint.yaml
    apiVersion: agents.kruise.io/v1alpha1
    kind: Checkpoint
    metadata:
      name: my-checkpoint
      namespace: default
    spec:
      sandboxName: my-sandbox          # Target Sandbox name (required)
      keepRunning: true                # Whether Sandbox continues running after snapshot, default true
      persistentContents:              # Persistent content
        - memory                       # Memory snapshot
        - filesystem                   # Filesystem snapshot
      ttlAfterFinished: 1d             # Auto-expiration time after completion (supports s/m/h/d format)
      targetContainer: sandbox         # Target container name, default "sandbox"
    ```

3. View Checkpoint status:

    ```bash
    kubectl get cp -n default
    kubectl describe cp my-checkpoint -n default
    ```

    Output example:

    ```
    NAME             STATUS      AGE
    my-checkpoint    Succeeded   2m
    ```

    Checkpoint state transitions:

    ```
    Pending → Creating → Succeeded / Failed
    ```

    Conditions changes:

    **Table 9** Conditions changes

    | Phase | Progressing | Archived | Ready |
    |-------|-------------|----------|-------|
    | Creating | True (AgentInvoked) | - | - |
    | Succeeded | Removed | True (UploadComplete) | True (CheckpointComplete) |
    | Failed | Removed | - | - |

    Common error scenarios:

    **Table 10** Checkpoint common error scenarios

    | Scenario | Message Example |
    |----------|-----------------|
    | Source Sandbox does not exist | `SourceSandboxNotFound` |
    | Runtime detection failed | `RuntimeDetectionFailed` |
    | node-agent unavailable | Preflight check failed, Controller auto-retries |
    | Creation timeout (10 minutes) | `Timeout` |

#### Restore from Checkpoint

**Create Restore Sandbox via kubectl:**

1. Execute `kubectl apply -f restore-sandbox.yaml` to create a Sandbox, specifying the Checkpoint name via the `agents.kruise.io/restore-from` annotation:

    ```yaml
    # restore-sandbox.yaml
    apiVersion: agents.kruise.io/v1alpha1
    kind: Sandbox
    metadata:
      name: my-restored-sandbox
      namespace: default
      annotations:
        agents.kruise.io/restore-from: my-checkpoint    # Points to the completed Checkpoint
    spec:
      templateRef:
        name: my-sandbox-template                       # Reference SandboxTemplate
      template:
        spec:
          runtimeClassName: kata-qemu                   # Must use kata runtime
          containers:
            - name: sandbox
              image: nginx:stable-alpine3.20
              ports:
                - containerPort: 80
          restartPolicy: Never
    ```

    After the Controller detects the `restore-from` annotation, it automatically creates a RestorePod and injects the following annotations:

    **Table 11** RestorePod annotation descriptions

    | Pod Annotation | Description |
    |----------------|-------------|
    | `agents.kruise.io/archive-location` | Snapshot archive location (from Checkpoint.Status.ArchiveLocation) |
    | `agents.kruise.io/restore-checkpoint-ref` | Checkpoint name |
    | `agents.kruise.io/restore-runtime-type` | Runtime type (e.g., `kata-qemu`) |
    | `agents.kruise.io/restore-vmm-type` | VMM type (e.g., `qemu`, extracted from RuntimeType) |

    **Trigger Restore via E2B API:**

    ```python
    from e2b import Sandbox

    sandbox = Sandbox(template="my-checkpoint-name")
    sandbox.wait_ready()
    print(f"Sandbox ID: {sandbox.sandbox_id}")
    ```

2. View Restore status:

    ```bash
    kubectl get sandbox my-restored-sandbox -n default
    ```

    Output example:

    ```
    NAME                    PHASE       AGE
    my-restored-sandbox     Running     2m
    ```

    View detailed information:

    ```bash
    kubectl describe sandbox my-restored-sandbox -n default
    ```

    Output example:

    ```
    Name:         my-restored-sandbox
    Namespace:    default
    Annotations:  agents.kruise.io/restore-from: my-checkpoint
    Status:
      Phase:                    Running
      Restore From Checkpoint:  my-checkpoint
      Restore Snapshot Path:    /var/lib/kata-restore/abc-123-def
      Restore Completion Time:  2026-06-03T10:30:00Z
      Node Name:                k8s-master
      Sandbox Ip:               10.244.0.5
    Conditions:
      Type        Status   Reason             Message
      ----        ------   ------             -------
      Restoring   False    RestoreCompleted   Sandbox restored from checkpoint my-checkpoint
      Ready       True     PodReady           Sandbox is ready
    ```

    Restore state transitions:

    ```
    Sandbox creation → Restoring (Pod creating) → Running (Restore completed)
    ```

    Exception scenarios:

    **Table 12** Restore exception scenarios

    | Scenario | Sandbox Phase | Message Example |
    |----------|---------------|-----------------|
    | Checkpoint does not exist | `Failed` | `Checkpoint my-checkpoint not found` |
    | Checkpoint not successful | `Failed` | `Checkpoint my-checkpoint is not Succeeded (phase: Failed)` |
    | Checkpoint has no ArchiveLocation | `Failed` | `Checkpoint my-checkpoint has no ArchiveLocation` |
    | Checkpoint has no RuntimeType | `Failed` | `Checkpoint my-checkpoint has no RuntimeType` |
    | Snapshot download failed | `Restoring` (continuous) | Pod remains Pending, Sandbox continues Restoring |

### Follow-up Operations

- After Restore is completed, the restored Sandbox can be used normally via kubectl or E2B API.
- The Controller asynchronously cleans up node-local snapshot files (`/var/lib/kata-restore/<pod-uid>/`).
- You can create a Checkpoint again from the restored Sandbox to implement a snapshot chain.

### Related Operations

- View Snapshot Artifacts

  After Checkpoint succeeds, get the archive location:

  ```bash
  kubectl get cp my-checkpoint -n default -o jsonpath='{.status.archiveLocation}'
  ```

  Based on the configured storage backend type, go to the corresponding location to view the artifacts:

  **Table 13** Snapshot artifact viewing methods

  | Storage Backend | Viewing Method |
  |-----------------|----------------|
  | `local` | Log in to the node, view the `--checkpoint-dir` directory (e.g., `/var/lib/checkpoints`) |
  | `s3` / `minio` | View the corresponding bucket via S3 console or CLI |
  | `nfs` | View after mounting the NFS shared path |
  | `csi` | View after mounting the CSI volume |

- Delete Checkpoint

  ```bash
  kubectl delete cp my-checkpoint -n default
  ```

  When deleting a Checkpoint, the Controller automatically cleans up the remote archive files.

- Set Auto-Expiration (TTL)

  Set the expiration time in `spec.ttlAfterFinished`; the Checkpoint will be automatically deleted after completion:

  ```yaml
  spec:
    ttlAfterFinished: 30m    # Auto-delete after 30 minutes
  ```

  Supported formats:

  **Table 14** TTL time formats

  | Format | Example | Description |
  |--------|---------|-------------|
  | Seconds | `30s` | 30 seconds |
  | Minutes | `30m` | 30 minutes |
  | Hours | `2h` | 2 hours |
  | Days | `1d` | 1 day |

## Appendix

### Checkpoint Spec Field Reference

**Table 15** Checkpoint Spec fields

| Field | Type | Required | Default Value | Description |
|--------|------|----------|---------------|-------------|
| `sandboxName` | string | Yes | - | Target Sandbox name |
| `podName` | string | No | - | Target Pod name (choose one with sandboxName) |
| `keepRunning` | bool | No | `true` | Whether Sandbox continues running after snapshot |
| `persistentContents` | []string | No | - | Persistent content: `memory`, `filesystem` |
| `ttlAfterFinished` | string | No | - | Auto-expiration time after completion |
| `targetContainer` | string | No | `sandbox` | Target container name |
| `backend` | string | No | `containerd` | Backend type: `containerd` or `envd` |
| `runtimeOptions` | object | No | - | Runtime-specific options |

### Checkpoint Status Field Reference

**Table 16** Checkpoint Status fields

| Field | Type | Description |
|--------|------|-------------|
| `phase` | string | Current phase: `Pending`, `Creating`, `Succeeded`, `Failed`, `Terminating` |
| `archiveLocation` | string | Snapshot archive URI (e.g., `local:///var/lib/checkpoints/...`) |
| `archiveSize` | int64 | Archive file size (bytes) |
| `checksum` | string | Archive file SHA256 checksum |
| `nodeName` | string | Node name that executed the snapshot |
| `containerID` | string | Container ID |
| `runtimeType` | string | Runtime type (e.g., `kata-qemu`) |
| `vmmType` | string | VMM type (e.g., `qemu`) |
| `completionTime` | time | Completion time |
| `conditions` | []Condition | Lifecycle conditions (Progressing, Archived, Ready) |

### Restore Status Field Reference

**Table 17** Restore Status fields

| Field | Type | Description |
|--------|------|-------------|
| `restoreFromCheckpoint` | string | Source Checkpoint name |
| `restoreSnapshotPath` | string | Node-local snapshot path (`/var/lib/kata-restore/<pod-uid>`) |
| `restoreCompletionTime` | time | Restore completion time |

### Required Host Path Mounts for node-agent

**Table 18** node-agent host path mounts

| Path | Purpose |
|------|---------|
| `/run/containerd` | Connect to containerd socket |
| `/var/lib/node-agent` | Agent working directory |
| `/var/lib/checkpoints` | Snapshot storage directory |
| `/var/lib/kata-restore` | Kata recovery convention path |

### Snapshot Artifact Structure

After Checkpoint succeeds, the storage backend writes the following directory structure:

```
<storage-root>/
└── <namespace>/
    └── <checkpointID>/
        ├── checkpoint.tar.gz         # Snapshot archive file
        ├── checkpoint.tar.gz.sha256  # SHA256 checksum file
        └── metadata.json             # Snapshot metadata
```

**metadata.json** contains the following fields:

**Table 19** metadata.json fields

| Field | Description |
|------|-------------|
| `CheckpointID` | Checkpoint CR name |
| `Namespace` | Kubernetes namespace |
| `ContainerID` | Container ID |
| `NodeName` | Node name that executed the snapshot |
| `RuntimeType` | Runtime type (e.g., `kata-qemu`) |
| `CreationTime` | Snapshot creation time |
| `ArchiveSize` | Archive file size (bytes) |
| `ChecksumSHA256` | SHA256 checksum of the archive file |

**Data Integrity Verification**:

```bash
# Calculate SHA256 of the actual file
sha256sum /var/lib/checkpoints/<namespace>/<cpID>/checkpoint.tar.gz

# Read the recorded checksum
cat /var/lib/checkpoints/<namespace>/<cpID>/checkpoint.tar.gz.sha256

# Compare whether the two match
```

If the checksums do not match, the file may be corrupted or tampered with, and that snapshot should not be used for restoration.

## FAQ

1. Checkpoint remains in Pending state after creation

    Check whether the Feature Gate is enabled (`SandboxCheckpoint=true`) and whether sandbox-node-agent is running on the target node.

2. Checkpoint creation failed, status is Failed

    **Possible Causes**

    - Source Sandbox does not exist or is not running (`SourceSandboxNotFound`)
    - Runtime detection failed, not using kata-qemu (`RuntimeDetectionFailed`)
    - node-agent unavailable or Preflight check failed
    - Snapshot creation timeout (10 minutes)

    **Solution**

    Use `kubectl describe cp <name>` to view the `Message` field to identify the cause, and check node-agent logs: `kubectl logs -n sandbox-system -l app=sandbox-node-agent`.

3. Sandbox remains in Restoring state after Restore

    **Symptom Description**

    The Sandbox continues in the `Restoring` phase, and the Pod remains in Pending state.

    **Possible Causes**

    - Snapshot download failed (archive file does not exist or storage backend unreachable)
    - kata-shim restoration failed

    **Solution**

    Check Pod Events (`kubectl describe pod <pod-name>`) and node-agent logs. When snapshot download fails, the Pod will not automatically downgrade to cold start; you must manually troubleshoot storage backend connectivity.

4. Can I restore across VMM types (e.g., QEMU snapshot to Firecracker)

    No. Restore must use the same runtime type (`RuntimeType`) as the Checkpoint.
