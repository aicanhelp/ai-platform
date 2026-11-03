# Online Manual Deployment of openFuyao-system-controller

## Background

This document describes how to manually prepare `patch-config`, render, and deploy `openFuyao-system-controller` in an online scenario within a Kubernetes environment where `openFuyao-system-controller` has not been automatically deployed via BKE, thereby bringing up the openFuyao management plane components.

In the online scenario, `openFuyao-system-controller` and management plane components pull images directly from the online image registry and pull Charts from the online Helm repository.

## Prerequisites

The following machines have been prepared:

| Role | Example IP | Purpose |
| --- | --- | --- |
| Business cluster node | `192.168.200.216` | An existing Kubernetes cluster where `openFuyao-system-controller` is manually deployed and the management plane is brought up |

For machine hardware configuration, refer to the requirements in [Quick Start](https://docs.openfuyao.cn/en/docs/v26.03/quick_start.html#quick-installation).

The following information has been prepared:

| Information | Purpose | Preparation Method |
| --- | --- | --- |
| `openFuyao-system-controller.yaml` | Deploy the business cluster controller component | Copied and rendered from the local template directory in [Section 3](#3-prepare-openfuyao-system-controller-yaml) later |
| `VersionConfig-latest.yaml` | Component version configuration used for `latest` version installation | This example reads from `/bke/mount/source_registry/files/patches/VersionConfig-latest.yaml` |

> ![icon](../figures/icon-note.gif) **Note:**<br />This document uses the `latest` version as an example. You can confirm the current version via the version configuration file: if you are using `VersionConfig-latest.yaml` and the `openFuyaoVersion` field in the file is `latest`, then it is the `latest` version; if you are using a specific version configuration file such as `Core-VersionConfig-v26.06-rc.1.yaml`, then it is the corresponding version. For versions other than `latest`, replace the image tag, Chart repository address, and version configuration file with the actual values of the corresponding version.

## Usage Restrictions

Online installation is required, and the business cluster node must be able to access the following addresses:

- Online image registry: `cr.openfuyao.cn/openfuyao`
- Online Helm repository: `https://helm.openfuyao.cn/_core`

## Tutorial

### 1. Check the Target Environment

#### 1.1 Confirm Management Plane Components Are Not Installed

Run on the business cluster node:

```bash
kubectl get ns openfuyao-system-controller openfuyao-system monitoring ingress-nginx
kubectl get deploy -A | grep -E "openfuyao|console|oauth|marketplace|monitoring"
kubectl get nodes
```

If namespaces such as `openfuyao-system-controller`, `openfuyao-system`, `monitoring`, `ingress-nginx` or management plane components already exist, please first confirm whether the current environment needs to be redeployed to avoid overwriting the existing management plane.

#### 1.2 Confirm BKE/CAPI Control Plane Is Not Installed

```bash
kubectl get crd | grep -E "bke|cluster.x-k8s"
kubectl get pods -n cluster-system
```

If there are no BKE/CAPI related CRDs in the environment, you cannot directly apply `BKECluster` type resources. This document uses the method of manually deploying `openFuyao-system-controller` to install the management plane.

#### 1.3 Confirm the Deployment Template Exists

```bash
ls /etc/openFuyao/addons/manifests/kubernetes/openfuyao-system-controller/latest/openfuyao-system-controller.yaml
grep -n "{{" /etc/openFuyao/addons/manifests/kubernetes/openfuyao-system-controller/latest/openfuyao-system-controller.yaml
```

The template usually contains the following variables and cannot be applied directly:

```text
{{ .repo }}
{{ .helmRepo }}
{{ .tagVersion }}
```

### 2. Create patch-config

#### 2.1 Confirm the Version Configuration File

Run on the business cluster node to confirm that the version configuration file used to create `patch-config` already exists:

```bash
ls /bke/mount/source_registry/files/patches/
grep -n "^openFuyaoVersion:" /bke/mount/source_registry/files/patches/VersionConfig-latest.yaml
```

If `/bke/mount/source_registry/files/patches/VersionConfig-latest.yaml` does not exist on the business cluster node, please first prepare the version configuration file at that path. This document uses the `latest` version as an example. You can download it with the following command:

```bash
mkdir -p /bke/mount/source_registry/files/patches
curl -fLo /bke/mount/source_registry/files/patches/VersionConfig-latest.yaml \
  https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/VersionConfig-latest.yaml
```

If `openFuyaoVersion` is `latest`, you can continue using this example; if it is another version, replace it with the corresponding version configuration file and synchronously replace the file name in subsequent commands.

#### 2.2 Create namespace and ConfigMap

```bash
kubectl create ns openfuyao-system-controller --dry-run=client -o yaml | kubectl apply -f -

kubectl create cm patch-config \
  -n openfuyao-system-controller \
  --from-file=patch-data=/bke/mount/source_registry/files/patches/VersionConfig-latest.yaml \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl get cm patch-config -n openfuyao-system-controller
```

### 3. Prepare openFuyao-system-controller YAML

#### 3.1 Copy the Deployment Template

```bash
cp /etc/openFuyao/addons/manifests/kubernetes/openfuyao-system-controller/latest/openfuyao-system-controller.yaml /tmp/openfuyao-system-controller.yaml
```

#### 3.2 Render Online Parameters

```bash
sed -i 's#{{ if .repo }}{{ .repo }}{{ else }}cr.openfuyao.cn/openfuyao/{{ end }}openfuyao-system-controller:{{ .tagVersion }}#cr.openfuyao.cn/openfuyao/openfuyao-system-controller:latest#g' /tmp/openfuyao-system-controller.yaml
sed -i 's#value: "{{ .repo }}"#value: "cr.openfuyao.cn/openfuyao"#g' /tmp/openfuyao-system-controller.yaml
sed -i 's#value: "{{ .helmRepo }}"#value: "https://helm.openfuyao.cn/_core"#g' /tmp/openfuyao-system-controller.yaml
```

The key configurations after rendering are as follows:

```yaml
image: cr.openfuyao.cn/openfuyao/openfuyao-system-controller:latest
OPENFUYAO_REGISTRY: cr.openfuyao.cn/openfuyao
HELM_REPORITORY_URL: https://helm.openfuyao.cn/_core
```

Check whether the template variables have been cleaned up:

```bash
grep -n "{{" /tmp/openfuyao-system-controller.yaml
```

If there is no output, the template variables have been cleaned up.

Check the key fields:

```bash
grep -nE "image:|OPENFUYAO_REGISTRY|HELM_REPORITORY_URL|entrypoint.sh" /tmp/openfuyao-system-controller.yaml
grep -nA2 "OPENFUYAO_REGISTRY\|HELM_REPORITORY_URL" /tmp/openfuyao-system-controller.yaml
```

### 4. Deploy openFuyao-system-controller

Run on the business cluster node:

```bash
kubectl apply -f /tmp/openfuyao-system-controller.yaml
kubectl get pods -n openfuyao-system-controller -w
```

When the Pod is in `Init:0/1`, it means the installer initContainer is executing the installation process.

View the installation logs:

```bash
kubectl logs -n openfuyao-system-controller deploy/openfuyao-system-controller -c installer -f
```

The logs should show the online image registry and online Helm repository addresses:

```text
cr.openfuyao.cn/openfuyao
https://helm.openfuyao.cn/_core
```

### 5. Check the Installation Results

#### 5.1 View All Components

```bash
kubectl get pods -A
```

#### 5.2 View Management Plane Components

```bash
kubectl get pods -n openfuyao-system
kubectl get pods -n ingress-nginx
kubectl get pods -n monitoring
```

On success, the core components should be `Running` or `Completed`, including:

- `console-service`
- `console-website`
- `local-harbor`
- `marketplace-service`
- `monitoring-service`
- `oauth-server`
- `oauth-webhook`
- `plugin-management-service`
- `user-management-operator`
- `web-terminal-service`

#### 5.3 View Abnormal Pods

```bash
kubectl get pods -A | grep -Ev "Running|Completed"
```

If there is no abnormal output, the `openFuyao-system-controller` and the management plane components it brought up have been deployed successfully.

<!-- ### 6. Uninstall and Restore Installation Verification (Optional)

#### 6.1 Generate install and uninstall YAML

```bash
cp /tmp/openfuyao-system-controller.yaml /tmp/openfuyao-system-controller-install-online.yaml
cp /tmp/openfuyao-system-controller.yaml /tmp/openfuyao-system-controller-uninstall-online.yaml

sed -i 's#sh /home/openfuyao-system/entrypoint.sh -o install#sh /home/openfuyao-system/entrypoint.sh -o uninstall#g' /tmp/openfuyao-system-controller-uninstall-online.yaml
grep -n "entrypoint.sh" /tmp/openfuyao-system-controller-uninstall-online.yaml
```

Confirm the output contains:

```text
sh /home/openfuyao-system/entrypoint.sh -o uninstall
```

#### 6.2 Execute uninstall

```bash
kubectl apply -f /tmp/openfuyao-system-controller-uninstall-online.yaml
kubectl rollout restart deploy/openfuyao-system-controller -n openfuyao-system-controller
kubectl logs -n openfuyao-system-controller deploy/openfuyao-system-controller -c installer -f
```

Verify the uninstall results:

```bash
kubectl get pods -n openfuyao-system
kubectl get pods -n ingress-nginx
kubectl get pods -n monitoring
kubectl get ns | grep -E "openfuyao-system|monitoring|ingress-nginx|session-secret"
```

When `openfuyao-system` related components are deleted or the namespace does not exist, it means the uninstall process has taken effect.

#### 6.3 Restore install

```bash
kubectl apply -f /tmp/openfuyao-system-controller-install-online.yaml
kubectl rollout restart deploy/openfuyao-system-controller -n openfuyao-system-controller
kubectl logs -n openfuyao-system-controller deploy/openfuyao-system-controller -c installer -f
```

Verify the restoration results:

```bash
kubectl get pods -n openfuyao-system-controller
kubectl get pods -n openfuyao-system
kubectl get pods -n ingress-nginx
kubectl get pods -n monitoring
kubectl get pods -A | grep -E "CrashLoopBackOff|ImagePullBackOff|ErrImagePull|Error|Pending"
```

When `openfuyao-system-controller` is restored to `1/1 Running`, the main components of `openfuyao-system`, `ingress-nginx`, and `monitoring` are restored to Running, and the abnormal Pod check has no output, it means the reinstallation and restoration in the online scenario is successful. -->
