# Kunpeng Confidential Container

## Feature Introduction

Kunpeng confidential container is built based on Kunpeng TEE technology, through the complete software stack of K8s+containerd+Kata+QEMU+KVM+CoCo, enabling confidential container deployment. Kunpeng confidential container has been customized and adapted on top of the open-source Kata/CoCo communities, featuring security capabilities such as remote attestation, image signing and encryption, and confidential container device passthrough.

### Application Scenarios

When business confidentiality requires deployment in secure containers, this feature provides strong isolation similar to traditional virtual machines, preventing security issues between different containers.

### Capability Scope

- Provides rapid compilation and build capabilities for Kunpeng confidential container related components.
- Provides customized and optimized Operators for Kunpeng confidential containers, enabling rapid installation and deployment of Kunpeng confidential container environments in Kubernetes clusters managed by openFuyao.
- Defines testing methods for Kunpeng confidential container features.

### Highlight Features

The official Kata/CoCo community only supports confidential containers on Intel, AMD, IBM and other hardware platforms, but does not support Kunpeng confidential containers. The openFuyao platform adds a rapid deployment solution for Kunpeng confidential container environments.

## Installation

### Prerequisites

#### Kunpeng Confidential Container Component Build

The build.sh script compiles and builds Kunpeng confidential container related components based on the TEE technology provided by Kunpeng virtCCA.
- Compiles and builds the confidential container runtime kata-deploy image from source, including shim-v2, qemu, kernel and other components.
- Compiles trustee components from source, building AS, KBS, RVPS images.
- Compiles trustee-operator from source, building trustee-operator image and deployment files.

##### Environment Preparation

The build environment uses an ARM operating system with kernel version v5.0 or above.

##### Prerequisites
- The build environment requires git, make, openssl, and docker to be installed.

  Docker can pull images, which can be verified by pulling the following image.

   ```shell
   docker pull ubuntu:22.04
   ```

##### Start Building
Use the root user for build compilation.

```shell
git clone https://gitcode.com/openFuyao/confidential-containers-deployment.git
cd confidential-containers-deployment/build/kunpeng
chmod 755 ./build.sh
```

1. Execute the following command to prepare the source code.

   ```shell
   ./build.sh prepare_src
   ```

2. Execute the following command to compile all kata-deploy components and build the kata-deploy image.

   ```shell
   ./build.sh compile_kdeploy_all
   ```

The final build artifact is the kata-deploy image `kata-deploy_3.15.0.tar`, located in: `./kata-containers/tools/packaging/kata-deploy`

3. Execute the following command, input the component name, to perform single-point compilation of kata-deploy components.
   
   ```shell
   ./build.sh compile_kdeploy_components [component name]
   
   # Available component names are as follows
   kernel                       - make kernel-tarball
   kernel-virtcca-confidential  - make kernel-virtcca-confidential-tarball
   nydus                        - make nydus-tarball
   shim-v2                      - make shim-v2-tarball
   virtiofsd                    - make virtiofsd-tarball
   qemu                         - make qemu-tarball
   qemu-virtcca-experimental    - make qemu-virtcca-experimental-tarball
   agent                        - make agent-tarball
   pause-image                  - make pause-image-tarball
   coco-guest-components        - make coco-guest-components-tarball
   rootfs-image                 - make rootfs-image-tarball
   rootfs-image-experimental    - make rootfs-image-experimental-tarball
   merge-builds                 - make merge-builds to combine all tarballs
   ```

When a component name is input, the build artifact is the component package `kata-static-component-name.tar.xz`, located in: `./kata-containers/tools/packaging/kata-deploy/local-build/build`

When `merge-builds` is input, all required components are checked and the kata-deploy image `kata-deploy_3.15.0.tar` is built, located in: `./kata-containers/tools/packaging/kata-deploy`

4. Execute the following command to compile remote attestation component images.
   
   ```shell
   ./build.sh compile_coco
   ```

The build artifacts are `rvps-grpc_v0.12.0.tar`, `kbs-grpc-as_v0.12.0.tar`, `as-grpc_v0.12.0.tar`, located in: `./kata-containers/build/trustee/output`

5. Execute the following command to compile the trustee-operator image.

```shell
./build.sh compile_trustee_operator
```

The build artifacts include (all stored in the `./kata-containers/build/trustee/output` directory):
- `trustee-operator_v0.3.0.tar` image file
- `deploy.yaml` deployment manifest file
- `kbsconfig.yaml` configuration example file

#### Pre-deployment Preparation

1. Install a K8s cluster. Because the Remote Snapshotter function of containerd is required, the container runtime must be containerd, with version no lower than v1.7.

2. The following container images are needed for deployment, ensure cluster nodes can pull them normally.

   - **Operator Controller** image: quay.io/confidential-containers/operator:v0.13.0
   - **Pre-install Payload** image: quay.io/confidential-containers/reqs-payload:latest

3. The kata-deploy image needs to be imported into the deployment environment. For build instructions, refer to the [Kunpeng Confidential Container Component Build](#start-building) section.
   ```shell
   # Importing into containerd image repository as an example
   ctr -n k8s.io i import kata-deploy_3.15.0.tar
   ```
   
### Start Installation

#### Deploy cc-Operator

Execute the following command to deploy cc-operator-controller-manager.

```shell
kubectl apply -k github.com/confidential-containers/operator/config/release?ref=v0.13.0
```

Execute the following command to deploy CcRuntime.

```shell
# Add a label to the node that needs to run, for example: kubectl label node node1 node.kubernetes.io/worker=
# Get the CcRuntime deployment yaml for virtCCA
wget https://raw.gitcode.com/openeuler/virtCCA_sdk/raw/master/kata-v3.15.0/conf/virtcca-kata-deploy.yaml
# The payloadImage field specifies the kata-deploy image path, for example: docker.io/library/kata-deploy:3.15.0, modify according to the imported image repository
kubectl apply -f virtcca-kata-deploy.yaml
```

After successful creation, there is one `cc-operator-controller` and two daemonset-managed Pods under the `confidential-containers-system` namespace.

```shell
# kubectl get pod -n confidential-containers-system -owide
NAMESPACE                           NAME
confidential-containers-system      cc-operator-controller-manager-685bdf4fbb-mllt5
confidential-containers-system      cc-operator-daemon-install-h4v42
confidential-containers-system      cc-operator-pre-install-daemon-rcfdt
```

The RuntimeClass corresponding to virtCCA has been registered in the K8s cluster.

```shell
# kubectl get runtimeclass
NAME                  HANDLER
kata                  kata-qemu
kata-qemu             kata-qemu
kata-qemu-coco-dev    kata-qemu-coco-dev
kata-qemu-virtcca     kata-qemu-virtcca
```

###### Create Confidential Container

Verify that the Kunpeng confidential container runtime is correctly configured and effective by deploying a test Pod.

Deploy test Pod

Run a confidential container Pod based on the `kata-qemu-virtcca` container runtime:

```yaml
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: test-kata-qemu-virtcca
  annotations:
    io.containerd.cri.runtime-handler: "kata-qemu-virtcca"
    io.katacontainers.config.hypervisor.kernel_params: "agent.debug_console agent.log=debug"
spec:
  runtimeClassName: kata-qemu-virtcca
  terminationGracePeriodSeconds: 5
  containers:
  - name: box-1
    image: registry.hw.com:5000/busybox:latest
    imagePullPolicy: Always
    command:
    - sh
    tty: true
EOF
```

##### Deploy trustee-Operator

Execute the following command to import the trustee-operator image.

```shell
ctr -n k8s.io i import trustee-operator_v0.3.0.tar
ctr -n k8s.io i import rvps_v0.12.0.tar
ctr -n k8s.io i import kbs-grpc-as_v0.12.0.tar
ctr -n k8s.io i import as-grpc_v0.12.0.tar
```

Deploy trustee-operator.

```shell
kubectl apply -f /workdir/deploy.yaml
# After successful creation, the cluster has the following pod, kubectl get pod -n trustee-operator-system
NAMESPACE                   NAME
trustee-operator-system     trustee-operator-controller-manager-6fcbf48f96-pgwg5
```

Deploy remote attestation components.

```shell
# Ensure the following directories exist on the host:
# - /etc/attestation/attestation-service/verifier/virtcca
# - /opt/confidential-containers/attestation-service
# - /opt/confidential-containers/kbs/repository/default
kubectl apply -f /workdir/kbsconfig.yaml
# After successful creation, the cluster has the following pod, kubectl get pod -n trustee-operator-system
NAMESPACE                   NAME
trustee-operator-system     trustee-deployment-694bf879d5-7wzwp
```

## Using Image Encryption and Image Signing Features

Confidential containers using image encryption and image signing features require the remote attestation service and installation of image signing tools.

### Configure Remote Attestation Service

#### Operation Steps

1. Configure the policy for verifying VirtCCA measurement reports.

   ```shell
   mkdir -p /opt/confidential-containers/attestation-service/token/simple/policies/opa
   vim /opt/confidential-containers/attestation-service/token/simple/policies/opa/default.rego
   
   # default.rego content is as follows
   package policy
   import future.keywords.every
   import future.keywords.if
   default allow := false
   allow if {
       print("Full Input:", input)
       print("Rim:", input["virtcca.realm.rim"])
       print("Ref:", data.reference)
       input["virtcca.realm.rim"] in data.reference["virtcca.realm.rim"]
   }
   ```

2. Pre-configure remote attestation root certificates and secondary CA certificate files on the remote attestation service node.

   ```shell
   mkdir -p /etc/attestation/attestation-service/verifier/virtcca/ && cd /etc/attestation/attestation-service/verifier/virtcca/
   ````

   Visit the website to download certificates, and place the following certificates in the current directory, keeping the downloaded certificate names unchanged.

   ```shell
   Huawei Equipment Root CA.pem
   Huawei IT Product CA.pem
   ```

  - RSA root certificate download link: <https://download.huawei.com/dl/download.do?actionFlag=download&nid=PKI1000000002&partNo=3001&mid=SUP_PKI>
  - RSA secondary CA certificate download link: <https://download.huawei.com/dl/download.do?actionFlag=download&nid=PKI1000000040&partNo=3001&mid=SUP_PKI>
  - ECC root certificate download link: <https://download.huawei.com/dl/download.do?actionFlag=download&nid=PKI1100000224&partNo=3001&mid=SUP_PKI>
  - ECC secondary CA certificate download link: <https://download.huawei.com/dl/download.do?actionFlag=download&nid=PKI1100000225&partNo=3001&mid=SUP_PKI>

3. Add RIM baseline values through the rvps tool.

   ```shell
   cd /opt/confidential-containers/attestation-service
   cat << EOF > sample
   {
       "virtcca.realm.rim": [
           "0f7733fcbaa9059d4d579fab25743868a2b4027290b09dfbc59964fc4b642kkk"
       ]
   }
   EOF
   provenance=$(cat sample | base64 --wrap=0)
   cat << EOF > message
   {
       "version" : "0.1.0",
       "type": "sample",
       "payload": "$provenance"
   }
   EOF
   
   # Register baseline values to rvps service
   kubectl exec -it -n trustee-operator-system trustee-deployment-694bf879d5-7wzwp -c rvps -- bash
   ./rvps-tool register --path /opt/confidential-containers/attestation-service/message --addr http://127.0.0.1:50003
   ```
   > ![image](figures/icon-note.gif)**Note:**
   >
   > The RIM baseline values (list) added for virtcca.realm.rim can be generated using the [Baseline Value Generation Tool](https://gitcode.com/openeuler/virtCCA_sdk/blob/master/attestation/rim_ref/README.md).
### Install Basic Tools
#### Operation Steps

1. Install golang.

   ```shell
   wget https://golang.google.cn/dl/go1.22.4.linux-arm64.tar.gz
   tar -zxvf go1.22.4.linux-arm64.tar.gz -C /usr/local
   echo "export GO_HOME=/usr/local/go" >> /etc/profile
   echo "export GOPATH=/home/work/go" >> /etc/profile
   source /etc/profile
   echo "export PATH=${GO_HOME}/bin:${PATH}" >> /etc/profile
   source /etc/profile
   ```

2. Install the cosign image signing tool.

   ```shell
   wget https://github.com/sigstore/cosign/releases/download/v2.5.0/cosign-linux-arm64
   chmod +x cosign-linux-arm64
   mv cosign-linux-arm64 /usr/local/bin/cosign
   ```

3. Compile and install skopeo.

   ```shell
   git clone -b v1.15.1 https://github.com/containers/skopeo.git $GOPATH/src/github.com/containers/skopeo
   yum install -y gpgme-devel device-mapper-devel
   cd $GOPATH/src/github.com/containers/skopeo && make bin/skopeo
   cp bin/skopeo /usr/local/bin
   
   # Test pulling image locally, you can see the image is successfully pulled
   mkdir -p /home/work/images && cd /home/work/images
   skopeo copy --insecure-policy   
   docker://docker.io/library/busybox:latest dir:busybox:latest
   ```

### Container Image Encryption and Decryption

Official reference documentation: <https://github.com/confidential-containers/guest-components/tree/main/attestation-agent/coco_keyprovider> <https://confidentialcontainers.org/docs/features/encrypted-images/>

#### Prerequisites
- Complete [Remote Attestation Service Configuration](#configure-remote-attestation-service)
- Complete [Basic Tools Installation](#install-basic-tools)

#### Operation Steps

Generate encryption key and encrypt container image

```shell
# Generate encryption key
KEY_FILE="image_key"
head -c 32 /dev/urandom | openssl enc > "$KEY_FILE"

# Create encrypted image generation directory
mkdir output

# Create container and encrypt plaintext image using generated key in the container (-s specifies the original image to be encrypted)
docker run -v "$PWD/output:/output" \\
-e http_proxy="http://IP:PORT" \\
-e https_proxy="http://IP:PORT" \\
ghcr.io/confidential-containers/staged-images/coco-keyprovider:latest /encrypt.sh \\
-k "$(base64 < image_key)" \\
-i kbs:///default/image_key/busybox_enc \\
-s docker://busybox:latest \\
-d dir:/output

# Push encrypted image to local image repository
skopeo copy dir:output docker://registry.hw.com:5000/busybox_enc:latest

# Check whether image is successfully encrypted through the following command
skopeo inspect docker://registry.hw.com:5000/busybox_enc:latest

# Copy encryption key to specified path
cp ./image_key /opt/confidential-containers/kbs/repository/default/image_key/busybox_enc
```

Start encrypted image container

```shell
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: enc-test
  annotations:
    io.containerd.cri.runtime-handler: "kata-qemu-virtcca"
    io.katacontainers.config.hypervisor.kernel_params: "agent.debug_console agent.log=debug agent.image_policy_file=kbs:///default/security-policy/test agent.enable_signature_verification=true agent.guest_components_rest_api=all agent.aa_kbc_params=cc_kbc::http:[attestation service pod IP]:8080"
spec:
  runtimeClassName: kata-qemu-virtcca
  terminationGracePeriodSeconds: 5
  containers:
  - name: box
    image: registry.hw.com:5000/busybox_enc:latest
    imagePullPolicy: Always
    command:
    - sh
    tty: true
EOF
```

Fill in the `agent.aa_kbc_params` parameter in `pod.metadata.annotations` with the Pod IP address of the remote attestation service, as shown in the example below.

```shell
# View remote attestation service pod, kubectl get pod -n trustee-operator-system -owide
NAMESPACE                  NAME                                    READY   STATUS    IP
trustee-operator-system    trustee-deployment-694bf879d5-7wzwp     3/3     Running   10.244.0.231

# In pod annotations, agent.aa_kbc_params=cc_kbc::http:10.244.0.231:8080"
```

### Container Image Signing

CoCo community reference documentation: <https://confidentialcontainers.org/docs/features/signed-images/>

#### Prerequisites
- Complete [Remote Attestation Service Configuration](#configure-remote-attestation-service)
- Complete [Basic Tools Installation](#install-basic-tools)

#### Operation Steps

Sign image

```shell
# Generate key pair (press Enter twice to skip password setup, user decides based on own requirements)
cosign generate-key-pair

# --tlog-upload=false means not uploading to cosign official (i.e., offline signing, user decides based on own requirements)
cosign sign --key cosign.key --tlog-upload=false registry.hw.com:5000/busybox:latest

# Deploy signing public key
mkdir -p /opt/confidential-containers/kbs/repository/default/cosign-key
cp ./cosign.pub /opt/confidential-containers/kbs/repository/default/cosign-key/1
```

Specify image repository verification policy

```
vim /opt/confidential-containers/kbs/repository/default/security-policy/test

{
    "default": [
        {
            "type": "reject"
        }
    ],
    "transports": {
        "docker": {
            "registry.hw.com:5000": [
                {
                    "type": "sigstoreSigned",
                    "keyPath": "kbs:///default/cosign-key/1"
                }
            ]
        }
    }
}
```

> ![image](figures/icon-note.gif)**Note:**
> 
> Set the type field to specify specific verification requirements. "reject" rejects any image, "sigstoreSigned" requires signature verification, "insecureAcceptAnything" performs no verification.

Verify signature and start container

Signature verification demo:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: sign-test
  annotations:
    io.containerd.cri.runtime-handler: "kata-qemu-virtcca"
    io.katacontainers.config.hypervisor.kernel_params: "agent.debug_console agent.log=debug agent.image_policy_file=kbs:///default/security-policy/test agent.enable_signature_verification=true agent.guest_components_rest_api=all agent.aa_kbc_params=cc_kbc::http:[attestation service pod IP]:8080"
spec:
  runtimeClassName: kata-qemu-virtcca
  terminationGracePeriodSeconds: 5
  containers:
  - name: box
    image: registry.hw.com:5000/busybox:latest
    imagePullPolicy: Always
    command:
      - sh
    tty: true
```

Fill in the `agent.aa_kbc_params` parameter in `pod.metadata.annotations` with the Pod IP address of the remote attestation service, as shown in the example below.

```shell
# View remote attestation service pod, kubectl get pod -n trustee-operator-system -owide
NAMESPACE                  NAME                                    READY   STATUS    IP
trustee-operator-system    trustee-deployment-694bf879d5-7wzwp     3/3     Running   10.244.0.231

# In pod annotations, agent.aa_kbc_params=cc_kbc::http:10.244.0.231:8080"
```
