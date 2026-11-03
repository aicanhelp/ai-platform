# 鲲鹏机密容器

## 特性介绍

鲲鹏机密容器基于鲲鹏TEE技术，通过k8s+containerd+Kata+QEMU+KVM+CoCo的整套软件栈进行构建，实现机密容器部署。鲲鹏机密容器在开源Kata/CoCo社区的基础上进行了定制和适配，具备远程证明、镜像签名和加密、机密容器设备直通等安全特性。

### 应用场景

当业务机密性强需要部署在安全容器中时，本特性提供了类似传统虚拟机的强隔离，防止不同容器之间的安全问题。

### 能力范围

- 提供鲲鹏机密容器相关组件的快速编译构建能力。
- 提供针对鲲鹏机密容器定制和优化的Operator，在openFuyao管理的Kubernetes集群中实现鲲鹏机密容器环境的快速安装部署。
- 制定鲲鹏机密容器特性测试方法。

### 亮点特征

Kata/CoCo官方社区只支持Intel、AMD、IBM等硬件平台的机密容器，不支持鲲鹏机密容器。在openFuyao平台增加对鲲鹏机密容器环境的快速部署解决方案。

## 安装

### 前提条件

#### 鲲鹏机密容器组件构建

build.sh脚本以鲲鹏virtCCA提供的TEE技术为基础，编译构建鲲鹏机密容器相关组件。
- 从源码编译构建机密容器运行时kata-deploy镜像，其中包括shim-v2，qemu，kernel等组件。
- 源码编译trustee组件，构建AS、KBS、RVPS镜像。
- 源码编译trustee-operator，构建trustee-operator镜像和部署文件。

##### 环境准备

构建环境使用ARM操作系统，内核版本为v5.0以上。

##### 前提条件
- 构建环境需安装git、make、openssl、docker。

  其中docker可以拉取镜像，可通过拉取下面镜像测试。

   ```shell
   docker pull ubuntu:22.04
   ```

##### 开始构建
使用root用户进行构建编译。

```shell
git clone https://gitcode.com/openFuyao/confidential-containers-deployment.git
cd confidential-containers-deployment/build/kunpeng
chmod 755 ./build.sh
```

1. 执行如下命令，进行源码准备。

   ```shell
   ./build.sh prepare_src
   ```

2. 执行如下命令，编译kata-deploy所有组件并构建kata-deploy镜像。

   ```shell
   ./build.sh compile_kdeploy_all
   ```

最终构建产物为kata-deploy镜像`kata-deploy_3.15.0.tar`，所在目录：`./kata-containers/tools/packaging/kata-deploy`

3. 执行如下命令，输入组件名，进行kata-deploy组件单点编译。
   
   ```shell
   ./build.sh compile_kdeploy_components [组件名]
   
   # 可用组件名如下
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

当输入组件名时，构建产物为组件包`kata-static-组件名.tar.xz`，所在目录：`./kata-containers/tools/packaging/kata-deploy/local-build/build`

当输入`merge-builds`时，检查所有所需组件并构建kata-deploy镜像`kata-deploy_3.15.0.tar`，所在目录：`./kata-containers/tools/packaging/kata-deploy`

4. 执行如下命令， 编译远程证明组件镜像。
   
   ```shell
   ./build.sh compile_coco
   ```

构建产物为`rvps-grpc_v0.12.0.tar`、`kbs-grpc-as_v0.12.0.tar`、`as-grpc_v0.12.0.tar`，所在目录：`./kata-containers/build/trustee/output`

5. 执行如下命令，编译trustee-operator镜像。

```shell
./build.sh compile_trustee_operator
```

构建产物包括（均存放于 `./kata-containers/build/trustee/output` 目录）：
- `trustee-operator_v0.3.0.tar` 镜像文件
- `deploy.yaml` 部署清单文件
- `kbsconfig.yaml` 配置示例文件

#### 部署前准备

1. 安装K8s集群，因为依赖containerd的Remote Snapshotter功能，要求容器运行时为containerd，且版本不低于v1.7。

2. 部署需要用到以下容器镜像，确保集群节点可正常拉取。

   - **Operator Controller** 镜像：quay.io/confidential-containers/operator:v0.13.0
   - **Pre-install Payload** 镜像：quay.io/confidential-containers/reqs-payload:latest

3. kata-deploy镜像需导入到部署环境，构建参考[鲲鹏机密容器组件构建](#开始构建)章节。
   ```shell
   # 以导入到containerd镜像仓为例
   ctr -n k8s.io i import kata-deploy_3.15.0.tar
   ```
   
### 开始安装

#### 部署cc-Operator

执行如下命令，部署cc-operator-controller-manager。

```shell
kubectl apply -k github.com/confidential-containers/operator/config/release?ref=v0.13.0
```

执行如下命令，部署CcRuntime。

```shell
# 给需要运行的节点添加标签，例如：kubectl label node node1 node.kubernetes.io/worker=
# 获取virtCCA的CcRuntime部署yaml
wget https://raw.gitcode.com/openeuler/virtCCA_sdk/raw/master/kata-v3.15.0/conf/virtcca-kata-deploy.yaml
# 其中payloadImage字段指定了kata-deploy镜像路径，例如：docker.io/library/kata-deploy:3.15.0，需根据导入的镜像仓库修改
kubectl apply -f virtcca-kata-deploy.yaml
```

创建成功后`confidential-containers-system` 命名空间下有一个`cc-operator-controller`和两个daemonset管理的Pod。

```shell
# kubectl get pod -n confidential-containers-system -owide
NAMESPACE                           NAME
confidential-containers-system      cc-operator-controller-manager-685bdf4fbb-mllt5
confidential-containers-system      cc-operator-daemon-install-h4v42
confidential-containers-system      cc-operator-pre-install-daemon-rcfdt
```

K8s集群中已注册virtCCA对应的RuntimeClass。

```shell
# kubectl get runtimeclass
NAME                  HANDLER
kata                  kata-qemu
kata-qemu             kata-qemu
kata-qemu-coco-dev    kata-qemu-coco-dev
kata-qemu-virtcca     kata-qemu-virtcca
```

###### 创建机密容器

通过部署一个测试Pod，验证鲲鹏机密容器运行时已正确配置并生效。

部署测试Pod

基于`kata-qemu-virtcca`容器运行时运行机密容器Pod：

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

##### 部署trustee-Operator

执行如下命令，导入trustee-operator镜像。

```shell
ctr -n k8s.io i import trustee-operator_v0.3.0.tar
ctr -n k8s.io i import rvps_v0.12.0.tar
ctr -n k8s.io i import kbs-grpc-as_v0.12.0.tar
ctr -n k8s.io i import as-grpc_v0.12.0.tar
```

部署trustee-operator。

```shell
kubectl apply -f /workdir/deploy.yaml
# 创建成功后集群中有下面pod，kubectl get pod -n trustee-operator-system
NAMESPACE                   NAME
trustee-operator-system     trustee-operator-controller-manager-6fcbf48f96-pgwg5
```

部署远程证明组件。

```shell
# 确保宿主机上有以下目录：
# - /etc/attestation/attestation-service/verifier/virtcca
# - /opt/confidential-containers/attestation-service
# - /opt/confidential-containers/kbs/repository/default
kubectl apply -f /workdir/kbsconfig.yaml
# 创建成功后集群中有下面pod，kubectl get pod -n trustee-operator-system
NAMESPACE                   NAME
trustee-operator-system     trustee-deployment-694bf879d5-7wzwp
```

## 使用镜像加密和镜像签名功能

机密容器使用镜像加密和镜像签名功能，需要基于远程证明服务，以及安装镜像签名工具。

### 配置远程证明服务

#### 操作步骤

1. 配置验证VirtCCA度量报告的policy。

   ```shell
   mkdir -p /opt/confidential-containers/attestation-service/token/simple/policies/opa
   vim /opt/confidential-containers/attestation-service/token/simple/policies/opa/default.rego
   
   # default.rego内容如下
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

2. 远程证明服务节点预置远程证明根证书、二级CA证书文件。

   ```shell
   mkdir -p /etc/attestation/attestation-service/verifier/virtcca/ && cd /etc/attestation/attestation-service/verifier/virtcca/
   ````

   访问网站下载证书，并将下面证书放到当前目录，保持下载的证书名称不变。

   ```shell
   Huawei Equipment Root CA.pem
   Huawei IT Product CA.pem
   ```

  - RSA根证书下载链接：<https://download.huawei.com/dl/download.do?actionFlag=download&nid=PKI1000000002&partNo=3001&mid=SUP_PKI>
  - RSA二级CA证书下载链接：<https://download.huawei.com/dl/download.do?actionFlag=download&nid=PKI1000000040&partNo=3001&mid=SUP_PKI>
  - ECC根证书下载链接：<https://download.huawei.com/dl/download.do?actionFlag=download&nid=PKI1100000224&partNo=3001&mid=SUP_PKI>
  - ECC二级CA证书下载链接：<https://download.huawei.com/dl/download.do?actionFlag=download&nid=PKI1100000225&partNo=3001&mid=SUP_PKI>

3. 通过rvps工具添加RIM基线值。

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
   
   # 将基线值注册到rvps服务
   kubectl exec -it -n trustee-operator-system trustee-deployment-694bf879d5-7wzwp -c rvps -- bash
   ./rvps-tool register --path /opt/confidential-containers/attestation-service/message --addr http://127.0.0.1:50003
   ```
   > ![image](figures/icon-note.gif)**说明：**
   >
   > virtcca.realm.rim 添加的RIM基线值（列表）可通过[基线值生成工具](https://gitcode.com/openeuler/virtCCA_sdk/blob/master/attestation/rim_ref/README.md)生成。
### 安装基础工具
#### 操作步骤

1. 安装golang。

   ```shell
   wget https://golang.google.cn/dl/go1.22.4.linux-arm64.tar.gz
   tar -zxvf go1.22.4.linux-arm64.tar.gz -C /usr/local
   echo "export GO_HOME=/usr/local/go" >> /etc/profile
   echo "export GOPATH=/home/work/go" >> /etc/profile
   source /etc/profile
   echo "export PATH=${GO_HOME}/bin:${PATH}" >> /etc/profile
   source /etc/profile
   ```

2. 安装cosign镜像签名工具。

   ```shell
   wget https://github.com/sigstore/cosign/releases/download/v2.5.0/cosign-linux-arm64
   chmod +x cosign-linux-arm64
   mv cosign-linux-arm64 /usr/local/bin/cosign
   ```

3. skopeo编译安装。

   ```shell
   git clone -b v1.15.1 https://github.com/containers/skopeo.git $GOPATH/src/github.com/containers/skopeo
   yum install -y gpgme-devel device-mapper-devel
   cd $GOPATH/src/github.com/containers/skopeo && make bin/skopeo
   cp bin/skopeo /usr/local/bin
   
   # 测试拉取镜像到本地，可以看到镜像被成功拉取
   mkdir -p /home/work/images && cd /home/work/images
   skopeo copy --insecure-policy   
   docker://docker.io/library/busybox:latest dir:busybox:latest
   ```

### 容器镜像加解密

官方参考文档： <https://github.com/confidential-containers/guest-components/tree/main/attestation-agent/coco_keyprovider> <https://confidentialcontainers.org/docs/features/encrypted-images/>

#### 前提条件
- 完成[远程证明服务配置](#配置远程证明服务)
- 完成[基础工具安装](#安装基础工具)

#### 操作步骤

生成加密秘钥并加密容器镜像

```shell
# 生成加密秘钥
KEY_FILE="image_key"
head -c 32 /dev/urandom | openssl enc > "$KEY_FILE"

# 创建加密镜像生成目录
mkdir output

# 创建容器并在容器中使用生成的秘钥加密明文镜像（-s指定被加密的原始镜像）
docker run -v "$PWD/output:/output" \\
-e http_proxy="http://IP:PORT" \\
-e https_proxy="http://IP:PORT" \\
ghcr.io/confidential-containers/staged-images/coco-keyprovider:latest /encrypt.sh \\
-k "$(base64 < image_key)" \\
-i kbs:///default/image_key/busybox_enc \\
-s docker://busybox:latest \\
-d dir:/output

# 推送加密镜像到本地镜像仓
skopeo copy dir:output docker://registry.hw.com:5000/busybox_enc:latest

# 通过下述命令检查镜像是否加密成功
skopeo inspect docker://registry.hw.com:5000/busybox_enc:latest

# 复制加密秘钥到指定路径
cp ./image_key /opt/confidential-containers/kbs/repository/default/image_key/busybox_enc
```

启动加密镜像容器

```shell
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: enc-test
  annotations:
    io.containerd.cri.runtime-handler: "kata-qemu-virtcca"
    io.katacontainers.config.hypervisor.kernel_params: "agent.debug_console agent.log=debug agent.image_policy_file=kbs:///default/security-policy/test agent.enable_signature_verification=true agent.guest_components_rest_api=all agent.aa_kbc_params=cc_kbc::http:[证明服务pod IP]:8080"
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

补充`pod.metadata.annotations`中的`agent.aa_kbc_params`参数，需填入远程证明服务的Pod IP地址，示例如下。

```shell
# 查看远程证明服务pod，kubectl get pod -n trustee-operator-system -owide
NAMESPACE                  NAME                                    READY   STATUS    IP
trustee-operator-system    trustee-deployment-694bf879d5-7wzwp     3/3     Running   10.244.0.231

# pod annotations里agent.aa_kbc_params=cc_kbc::http:10.244.0.231:8080"
```

### 容器镜像签名

CoCo社区参考文档：<https://confidentialcontainers.org/docs/features/signed-images/>

#### 前提条件
- 完成[远程证明服务配置](#配置远程证明服务)
- 完成[基础工具安装](#安装基础工具)

#### 操作步骤

签名镜像

```shell
# 生成密钥对(两次回车可无需设置密码，用户视自身诉求而定)
cosign generate-key-pair

# --tlog-upload=false表示不上传到cosign官方（即离线签名，用户视自身诉求而定）
cosign sign --key cosign.key --tlog-upload=false registry.hw.com:5000/busybox:latest

# 部署签名公钥
mkdir -p /opt/confidential-containers/kbs/repository/default/cosign-key
cp ./cosign.pub /opt/confidential-containers/kbs/repository/default/cosign-key/1
```

指定镜像仓验证策略

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

> ![image](figures/icon-note.gif)**说明：**
> 
> 设置type字段指定具体验证要求，"reject"为拒绝任何镜像，"sigstoreSigned"为要求验证签名，"insecureAcceptAnything"为不进行任何验证。

验签启动容器

签名验签demo：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: sign-test
  annotations:
    io.containerd.cri.runtime-handler: "kata-qemu-virtcca"
    io.katacontainers.config.hypervisor.kernel_params: "agent.debug_console agent.log=debug agent.image_policy_file=kbs:///default/security-policy/test agent.enable_signature_verification=true agent.guest_components_rest_api=all agent.aa_kbc_params=cc_kbc::http:[证明服务pod IP]:8080"
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

补充`pod.metadata.annotations`中的`agent.aa_kbc_params`参数，需填入远程证明服务的Pod IP地址，示例如下。

```shell
# 查看远程证明服务pod，kubectl get pod -n trustee-operator-system -owide
NAMESPACE                  NAME                                    READY   STATUS    IP
trustee-operator-system    trustee-deployment-694bf879d5-7wzwp     3/3     Running   10.244.0.231

# pod annotations里agent.aa_kbc_params=cc_kbc::http:10.244.0.231:8080"
```



