# 机密容器部署指南

本项目提供基于国产硬件的机密容器（Confidential Containers）安装部署能力，支持海光 CSV 和鲲鹏 virtCCA 两种硬件场景。两种方案均基于 [Kata Containers](https://github.com/kata-containers/kata-containers) 和 [Confidential Containers](https://github.com/confidential-containers) 社区技术，通过 CoCo Operator 在 Kubernetes 集群中部署机密容器运行时。

| 场景 | 硬件 | 架构 | TEE 技术 | RuntimeClass |
| ---- | ---- | ---- | -------- | ------------ |
| 海光 CSV | 海光 CPU | x86 | CSV（国密 SM4 内存加密） | `kata-qemu-csv` |
| 鲲鹏 virtCCA | 鲲鹏 CPU | ARM | virtCCA | `kata-qemu-virtcca` |

## 目录结构

```
├── build/
│   ├── csv/          # 海光 CSV 组件构建（build.sh / Makefile / 12步脚本）
│   └── kunpeng/      # 鲲鹏 virtCCA 组件构建（build.sh）
├── deploy/
│   ├── csv/          # CSV 环境配置与部署文档
│   └── kunpeng/      # virtCCA 环境部署文档
├── docs/             # 功能验证文档
└── test/
```

## 版本信息

| 组件 | 海光 CSV | 鲲鹏 virtCCA |
| ---- | -------- | ------------ |
| Kata Containers | 3.13.0 | v3.15.0 |
| Trustee | 0.11.0 | v0.12.0 |
| Guest Components | 0.10.0 | v0.12.0 |
| CoCo Operator | v0.12.0 | v0.13.0 |
| Trustee Operator | 0.3.0 | v0.3.0 |
| virtCCA SDK | — | v0.1.37 |

## 环境要求

**通用**：K8s 集群、containerd ≥ v1.7、git/make/openssl/docker

| 项目 | 海光 CSV | 鲲鹏 virtCCA |
| ---- | -------- | ------------ |
| 硬件 | 海光 CPU 物理机/裸金属 | 鲲鹏处理器服务器 |
| 操作系统 | RHEL 系（BCLinux/openEuler），x86 | ARM Linux，内核 ≥ v5.0 |
| 内核 | 支持 CSV（5.10 或 6.6） | 随 Kata 版本 |
| 特殊要求 | BIOS 启用 SMEE；内核参数 `kvm-amd.sev=1 sev_es=1`；导入 HGSC 证书 | virtCCA SDK |

## 快速开始

### 海光 CSV

```bash
# 1. 配置 CSV 环境（BIOS/内核/证书），参考 deploy/csv/配置海光CSV环境.md
hag general check          # 验证 CSV: enabled

# 2. 组件构建
cd build/csv
sudo make all              # 或 sudo ./build.sh，支持 --only-steps/--skip-steps

# 3. 部署运行时
git clone https://gitee.com/hanliyang-kata-coco/operator.git -b 0.12.0-hygon-arch-sig coco-operator
cd coco-operator && kubectl apply -k config/release
kubectl apply -k config/samples/ccruntime/default
kubectl label nodes $NODE_NAME node.kubernetes.io/worker=

# 4. 创建机密容器（runtimeClassName: kata-qemu-csv）
```

### 鲲鹏 virtCCA

```bash
# 1. 组件构建
cd build/kunpeng
./build.sh prepare_src                  # 准备源码并应用补丁
./build.sh compile_kdeploy_all          # 编译 kata-deploy 镜像
./build.sh compile_coco                 # 编译 RVPS/KBS/AS 镜像
./build.sh compile_trustee_operator     # 编译 trustee-operator

# 2. 部署运行时
ctr -n k8s.io i import kata-deploy_3.15.0.tar
kubectl apply -k github.com/confidential-containers/operator/config/release?ref=v0.13.0
kubectl apply -f virtcca-kata-deploy.yaml   # 从 virtCCA_sdk 获取，注意修改 payloadImage

# 3. 部署远程证明服务
ctr -n k8s.io i import trustee-operator_v0.3.0.tar
kubectl apply -f deploy.yaml
kubectl apply -f kbsconfig.yaml

# 4. 创建机密容器（runtimeClassName: kata-qemu-virtcca）
```

## 相关文档

**海光 CSV**
- [构建机密容器运行时组件](build/csv/README.md)
- [配置海光CSV环境](deploy/csv/配置海光CSV环境.md)
- [快速部署CSV机密容器运行环境](deploy/csv/快速部署基于kata-3.13的CSV机密容器运行环境.md)
- [测试CSV容器的基本特性](docs/测试CSV容器的基本特性.md)

**鲲鹏 virtCCA**
- [鲲鹏机密容器组件构建](build/kunpeng/鲲鹏机密容器组件构建.md)
- [鲲鹏机密容器环境部署](deploy/kunpeng/鲲鹏机密容器环境部署.md)
- [鲲鹏机密容器验证](docs/kunpeng/鲲鹏机密容器验证.md)

## 参考链接

- [Kata Containers](https://github.com/kata-containers/kata-containers)
- [Confidential Containers](https://github.com/confidential-containers)
- [Hygon Arch 技术社区](https://openanolis.cn/sig/Hygon-Arch)
- [virtCCA SDK](https://gitcode.com/openeuler/virtCCA_sdk)

# 联系我们
 	 
orchestration-engine@list.openfuyao.cn