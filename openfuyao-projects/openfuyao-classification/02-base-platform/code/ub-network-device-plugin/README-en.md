# ub-network-device-plugin

UB network device plugin

## Compilation Preparation

1. Install make, gcc, docker.

   Can be installed via yum install.

2. Install go, version >= 1.24.0

   Visit official download address https://go.dev/dl/go1.24.0.linux-arm64.tar.gz to get release package, execute following command to extract to environment:
   ```bash
   tar zxvf go1.24.0.linux-arm64.tar.gz -C /usr/local
   export PATH=/usr/local/go/bin:$PATH
   ```

## Build

Execute build in the code root directory.

```bash
# Build binary
make build

# Build image
make image

# Build binary and image
make
```

## Deployment

1. Import binary artifact generated during build into cluster.

   ```bash
   ctr -n k8s.io image import ub-network-device-plugin-v1.0.1.tar
   ```
2. Please refer to yaml file example under example, modify image repository and version number, and apply in the environment.

   ```bash
   kubectl apply -f ub-network-device-plugin.yaml
   ```

## Device Usage

Configure resources in containers as shown in the following example.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: urma-test
  namespace: test
spec:
  containers:
  - name: urma-test
    image: busybox:latest
    resources:
      requests:
        cpu: "1"
        memory: "128Mi"
        unifiedbus.com/ub_net_device: "1"
      limits:
        cpu: "1"
        memory: "128Mi"
        unifiedbus.com/ub_net_device: "1"
```