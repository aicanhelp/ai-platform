# 配置海光 CSV 环境

## 更新历史

- 2025-11-24：创建文档

## 参考文档

- [安全功能配置要求](https://openanolis.cn/sig/Hygon-Arch/doc/1076446632387394260)
- [安装操作系统和内核](https://openanolis.cn/sig/Hygon-Arch/doc/865622272844371792)
- [海光机密计算配置汇总](https://openanolis.cn/sig/Hygon-Arch/doc/1375766483729154058)
- [导入通用安全证书](https://openanolis.cn/sig/Hygon-Arch/doc/865622274698254162)
- [hygon-devkit 工具仓库](https://gitee.com/anolis/hygon-devkit/blob/master/common/README.md)
- [Hygon Arch 技术社区主页](https://openanolis.cn/sig/Hygon-Arch)

## 目录索引

- [海光 CSV 介绍](#海光-csv-介绍)
- [CSV 硬件环境配置](#csv-硬件环境配置)
  - [1. 硬件要求](#1-硬件要求)
  - [2. BIOS 设置](#2-bios-设置)
- [CSV 软件环境配置](#csv-软件环境配置)
  - [1. 操作系统和内核](#1-操作系统和内核)
  - [2. 设置内核启动参数](#2-设置内核启动参数)
  - [3. 导入通用安全证书](#3-导入通用安全证书)
    - [获取 hag 工具](#获取-hag-工具)
    - [在线导入证书](#在线导入证书)
    - [离线导入证书](#离线导入证书)
- [检查软硬件配置](#检查软硬件配置)
  - [通用状态检查](#通用状态检查)
  - [CSV 平台能力检查](#csv-平台能力检查)

## 海光 CSV 介绍

海光 **CSV**（China Secure Virtualization）是海光自主研发的基于国产海光 CPU 硬件的安全虚拟化技术，其设计目标是通过 CSV 虚拟机构建可信执行环境（Trusted Execution Environment，TEE），广泛适用于云计算、机密计算等对数据安全要求严苛的场景。

该技术历经 **CSV1、CSV2、CSV3** 三代迭代：
- **CSV1** 实现了虚拟机内存加密能力，采用 **国密 SM4** 算法，不同虚拟机配备独立加密密钥，由海光安全处理器统一管理，主机无法解密虚拟机加密内存；
- **CSV2** 新增虚拟机状态加密机制；
- **CSV3** 进一步提供虚拟机内存隔离支持。

基于 CSV 构建的机密计算方案（如 CSV 机密容器），可同时为用户提供虚拟机内存与状态的端到端加密保护。即使云平台主机被攻陷，云厂商或外部攻击者也无法监控或篡改 CSV 实例内部的运行状态，例如进程信息、敏感计算数据等。

在硬件层面，海光 CPU 通过 ASID（Address Space ID）区分不同 CSV 虚拟机与主机，为每个 CSV 虚拟机分配独立的 TLB、Cache 等硬件资源，实现强隔离；在功能层面，CSV 虚拟机支持安全启动、代码验证、远程认证等功能。写内存数据时 CPU 硬件自动加密，读数据时自动解密，构建出安全的硬件级可信执行环境。

如需了解更多技术细节，可参考 [Hygon Arch](https://openanolis.cn/sig/Hygon-Arch) 相关文档。

## CSV 硬件环境配置

### 1. 硬件要求

需使用搭载 **海光 CPU** 的物理机或裸金属服务器。
  
不同 CPU 型号对 CSV1/CSV2/CSV3 的支持情况，请查阅文档《[安全功能配置要求](https://openanolis.cn/sig/Hygon-Arch/doc/1076446632387394260)》中的 **表 3：机密计算硬件能力细分说明**。

### 2. BIOS 设置

启用海光加密虚拟化功能（CSV1/CSV2/CSV3）前，需在 BIOS 中完成以下配置：
- **SMEE Control**: 设为 `Enabled`  
- **ASID Space Limit**: 设为 `1`  
- **RDSEED and RDRAND**: 设为 `Auto`

> 具体选项名称和默认值可能因主板厂商或 BIOS 版本而异，建议以《[安全功能配置要求](https://openanolis.cn/sig/Hygon-Arch/doc/1076446632387394260)》中的官方说明为准。

## CSV 软件环境配置

### 1. 操作系统和内核

使用 CSV 功能需依赖支持该特性的 Linux 内核。海光已将 CSV 相关代码合入以下开源社区内核仓库：
- OpenAnolis Cloud Kernel: https://gitee.com/anolis/cloud-kernel  
- openEuler Kernel: https://gitee.com/openeuler/kernel

因此，**推荐优先选用 OpenAnolis 或 openEuler 的较新发行版**。但请注意：由于 CSV 支持是逐步合入的，早期正式版可能不包含完整功能。不同发行版的具体版本选择建议如下：

**OpenAnolis 系列**：
- 推荐使用 Anolis OS 8.10，并将内核升级至最新版本
- 如需使用 Anolis OS 23，建议尝试 23.3 Nightly build

**openEuler 系列**：
- 推荐使用 24.09 或 25.09 版本
- 如需使用 5.10 内核，建议尝试 22.03-LTS-SP4 dailybuild

若当前操作系统内核未集成 CSV 支持，可参考文档《[安装操作系统和内核](https://openanolis.cn/sig/Hygon-Arch/doc/865622272844371792)》，从海光定制内核仓库（https://gitee.com/anolis/hygon-cloud-kernel.git） 自行构建并安装 5.10 或 6.6 版本的内核。

### 2. 设置内核启动参数

参考文档《[海光机密计算配置汇总](https://openanolis.cn/sig/Hygon-Arch/doc/1375766483729154058)》的 **表 8 主机 Linux 内核启动的内核命令行参数**，启用 CSV 功能至少需要设置以下参数：
```bash
kvm-amd.sev=1
kvm-amd.sev_es=1
```

### 3. 导入通用安全证书

为使 CSV 虚拟机完成远程认证，需预先导入海光通用安全证书（HGSC）。详细流程请参考官方文档《[导入通用安全证书](https://openanolis.cn/sig/Hygon-Arch/doc/865622274698254162)》。

#### 获取 hag 工具

hag 是海光提供的证书与平台管理工具，可通过 [hygon-devkit](https://gitee.com/anolis/hygon-devkit/blob/master/common/README.md) 中的脚本安装：
```bash
sudo ./install_hag.sh
```
默认安装路径为 `/opt/hygon/bin/hag`。

#### 在线导入证书

若主机可访问互联网，直接运行：
```bash
sudo /opt/hygon/bin/hag general hgsc_import
```

#### 离线导入证书

若主机处于离线环境，请按以下步骤操作：

1. 获取芯片 ID 和证书版本号：
```bash
sudo /opt/hygon/bin/hag general get_id
sudo /opt/hygon/bin/hag general hgsc_version
```

2. 在联网机器上访问 https://cert.hygon.cn/hgsc, 输入上述信息下载证书文件。

3. 将证书文件上传至目标主机，并执行导入命令（替换实际路径）：
```bash
sudo /opt/hygon/bin/hag general hgsc_import -offline -in /realpath/to/hgsc_certchain
```

## 检查软硬件配置

可使用 hag 工具快速验证当前主机的 CSV 软硬件支持状态。

### 通用状态检查

运行以下命令查看系统整体安全配置：
```bash
hag general check
```

**示例输出**：
```bash
firewalld:                  inactive
selinux:                    Disabled
kernel version:             5.10.134-csv
model name:                 Hygon C86 7285 32-core Processor
Hygon C86 7285 32-core Processor
is Hygon CPU:               YES
failed check sme with psp cmd, retry by x86 method
HW SME:                     enabled
Linux SME support:          YES!
SME:                        enabled
CSV:                        enabled
/dev/sev:                   exist
ERR: do_tpcm_get_info:76: psp_dispatch_cmd failed with ret: -65534, pspret: 65534
csv api version:            1.3
firmware version:           1667
is HGSC imported:           YES
chip id:                    NRKG901061105
/dev/tdm:                   exist
TDM:                        enabled
tdm api version:            1.4
/dev/tpm0:                  nonexist
tpm_acpi:                   nonexist
/dev/tcm0:                  nonexist
tcm_acpi:                   nonexist
```

**结果异常解读**：
- 若`CSV`字段显示`disabled`：表明 BIOS 中 SMEE 等 CSV 相关功能未正确配置；
- 若`/dev/sev`字段显示`nonexist`：表明系统内核未集成 CSV 支持，或缺少必要的启动参数（如 kvm-amd.sev=1）；
- 若`is HGSC imported`字段显示`NO`：表明尚未完成通用安全证书的导入操作。

### CSV 平台能力检查

运行以下命令查看 CSV 特性支持详情：
```bash
hag csv platform_status
```

**示例输出**：
```bash
api_major:          1
api_minor:          3
platform_state:     CSV_STATE_INIT
owner:              PLATFORM_STATE_SELF_OWN
chip_secure:        SECURE
fw_enc:             ENCRYPTED
fw_sign:            SIGNED
CSV:                CSV CSV2
build id:           1667
guest_count:        0
is HGSC imported:   YES
supported csv guest:11
```

**关键指标校验**：
- `platform_state`正常取值应为`CSV_STATE_INIT`或`CSV_STATE_WORKING`，代表 CSV 平台处于初始化或工作状态；
- `CSV`字段会显示当前主机支持的 CSV 版本（如示例中的`CSV CSV2`），反映硬件对 CSV 迭代版本的兼容情况。
