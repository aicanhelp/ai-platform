# victoriametrics-controller

victoriametrics-controller is used to quickly deploy and unload VictoriaMetrics monitoring stack on K8s cluster.

## Image Build
### Environment Preparation
If you use `nerdctl` for building, you can ignore this step.

When using `Docker` for building, to support multi-architecture image building, you need to additionally install plugin `buildx` to extend `Docker` build functionality. For specific installation method, see [buildx official installation guide](https://github.com/docker/buildx).

### Build Commands

#### Build and push to specified OCI repository

**Using docker**

```bash
docker buildx build . -f <path/to/dockerfile> \
    -o type=image,name=<oci/repository>:<tag>,oci-mediatypes=true,rewrite-timestamp=true,push=true \
    --platform=linux/amd64,linux/arm64 \
    --provenance=false 
```

**Using nerdctl**

```bash
nerdctl build . -f <path/to/dockerfile> \
    -o type=image,name=<oci/repository>:<tag>,oci-mediatypes=true,rewrite-timestamp=true,push=true \
    --platform=linux/amd64,linux/arm64 \
    --provenance=false
```

Where `<path/to/dockerfile>` is the Dockerfile path, `<oci/repository>` is the image address, and `<tag>` is the image tag.

#### Build and export OCI Layout to local tarball

**Using docker**

```bash
docker buildx build . -f <path/to/dockerfile> \
    -o type=oci,name=<oci/repository>:<tag>,dest=<path/to/oci-layout.tar>,rewrite-timestamp=true \
    --platform=linux/amd64,linux/arm64 \
    --provenance=false
```

**Using nerdctl**

```bash
nerdctl build . -f <path/to/dockerfile> \
    -o type=oci,name=<oci/repository>:<tag>,dest=<path/to/oci-layout.tar>,rewrite-timestamp=true \
    --platform=linux/amd64,linux/arm64 \
    --provenance=false
```

Where `<path/to/dockerfile>` is the Dockerfile path, `<oci/repository>` is the image address, `<tag>` is the image tag, and `path/to/oci-layout.tar` is the tar package path.

#### Build and export image rootfs to local directory

**Using docker**

```bash
docker buildx build . -f <path/to/dockerfile> \
    -o type=local,dest=<path/to/output>,platform-split=true \
    --platform=linux/amd64,linux/arm64 \
    --provenance=false
```

**Using nerdctl**

```bash
nerdctl build . -f <path/to/dockerfile> \
    -o type=local,dest=<path/to/output>,platform-split=true \
    --platform=linux/amd64,linux/arm64 \
    --provenance=false
```

Where `<path/to/dockerfile>` is the Dockerfile path and `path/to/output` is the local directory path.

## Deployment

### Deploy using BKE cluster
Reference: [victoriametrics_controller usage guide](https://gitcode.com/openFuyao/sig-installation/blob/openFuyao-v26.03/docs/en/user_guide/cluster_installation_deployment/victoriametrics_controller_user_guide.md)

### Deploy non-BKE cluster
1. Edit image address and parameter configuration in `deploy.yaml`.
    
    Modify `spec.template.spec.initContainers.image` and `spec.template.Containers.image` to build image address `<oci/repository>:<tag>`, as follows.
    ```
    initContainers:
       - name: installer
         image: <oci/repository>:<tag>
         imagePullPolicy: Always
 
     ...
 
    containers:
       - name: victoriametric-controller
         env:
           - name: OPENFUYAO_REGISTRY
             value: "cr.openfuyao.cn/openfuyao"
         securityContext:
           privileged: true
         image: <oci/repository>:<tag>
         imagePullPolicy: Always
    ```
    Modify `spec.template.spec.initContainers.env` field under victoriametrics related configuration.
    ```
     - name: OPENFUYAO_REGISTRY
       value: "cr.openfuyao.cn/openfuyao"
     - name: VERSION
       value: "0.58.2"
     - name: useVMSingle
       value: "false"
     - name: vmSingleStorageSize
       value: "50Gi"
     - name: vmAgentAllowStatefulSet
       value: "true"
     - name: vmAgentCpuCount
       value: "0.5"
     - name: vmAgentMemorySize
       value: "2Gi"
     - name: vmAgentStorageSize
       value: "10Gi"
     - name: vmAgentReplicaCount
       value: "1"
     -.name: vmAgentShareCount
       value: "2"
 
     ...
    ```

2. Execute the following command to apply the configuration file.
    ```
    kubectl apply -f deployment.yaml
    ```
    Wait for victoriametrics to complete installation.
