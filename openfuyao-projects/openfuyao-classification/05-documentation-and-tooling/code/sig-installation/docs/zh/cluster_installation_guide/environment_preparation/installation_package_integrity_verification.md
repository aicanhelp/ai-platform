# 安装工具下载及校验

本文档以ARM64架构为参考介绍安装包的完整性校验，AMD64架构的类似处理。

## 简介

为了确认安装包在传输过程中是否由于网络连接、存储设备、恶意攻击等原因出现数据不完整或遭到篡改的问题，在获取到安装包后，需要对安装包进行完整性校验，通过了校验的安装包才能被部署。

这里通过对比校验文件中记录的校验值和手动方式计算的安装包文件校验值，判断安装包是否完整；而校验文件的合法性则通过签名文件（`.asc`）和PGP公钥来验证。完整的信任链如下所示。

```
PGP公钥（导入后展示的主密钥）<--> asc签名校验输出的主密钥 --> 校验文件 --> 安装包
```

> ![输入图片说明](../figures/icon-note.gif) **说明：**<br />社区PGP主密钥会随密钥轮换而更新，文档不再提供静态指纹。校验时只需确认：本机导入公钥后展示的主密钥，与使用`.asc`文件校验sha256文件时gpg输出的主密钥一致，即可判定签名校验通过。展示的密钥无需与任何固定指纹或其他渠道的密钥进行匹配。

## 前提条件

在校验安装包完整性之前，需要准备如下文件。

- openFuyao社区PGP公钥：`openfuyao.gpg`，[单击下载](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/openfuyao.gpg)。
- 安装包文件：`bkeadm_linux_arm64`，[单击下载](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/bkeadm_linux_arm64)。
- sha256校验文件：`bkeadm_linux_arm64.sha256`，[单击下载](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/bkeadm_linux_arm64.sha256)。
- gpg签名文件：`bkeadm_linux_arm64.sha256.asc`，[单击下载](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/bkeadm_linux_arm64.sha256.asc)。

## 操作指导

将上面四个文件放置到同一个目录下，文件完整性校验操作步骤如下。

1. 用户首次下载PGP公钥的情况下，需要执行如下命令导入公钥。

   ```shell
   gpg --import openfuyao.gpg
   ```

   导入完成后，执行如下命令展示当前导入的公钥及其主密钥指纹。密钥信息以本机实际输出为准（动态加载，随公钥文件更新），请记录输出中的主密钥指纹，供下一步比对。

   ```shell
   gpg --list-keys --with-fingerprint --with-subkey-fingerprint contact@openfuyao.cn
   ```

2. 执行如下命令，使用`.asc`签名文件校验sha256文件，并从校验输出中动态读取本次签名使用的主密钥。

   ```shell
   gpg --verify bkeadm_linux_arm64.sha256.asc bkeadm_linux_arm64.sha256
   ```

   命令执行完成后，同时满足以下条件则sha256校验文件合法：

   - 输出包含`Good signature from "openFuyao <contact@openfuyao.cn>"`。
   - 输出中的密钥指纹与步骤1展示的主密钥指纹一致。指纹字段名因 GPG 版本而异，常见为 `Primary key fingerprint` 或 `Key fingerprint`，**以实际 `gpg --verify` 输出为准**；比对时以指纹取值一致为准，不依赖固定英文字段名。

   展示的密钥不需要与文档或其他渠道的固定指纹匹配；只要步骤1展示的主密钥与本步骤gpg校验输出的主密钥一致，即说明校验通过。

   若校验失败（例如出现 `BAD signature`、无法验证签名，或主密钥指纹与步骤1不一致），按以下步骤处理：

   - 删除本目录下已下载的 `openfuyao.gpg`、`bkeadm_linux_arm64.sha256`、`bkeadm_linux_arm64.sha256.asc`。
   - 重新从本文[前提条件](#前提条件)中的官方下载地址获取上述文件，并确认文件名与架构一致。
   - 如本机此前已导入过旧公钥，可先删除旧密钥后再重新导入：

     ```shell
     gpg --delete-keys contact@openfuyao.cn
     gpg --import openfuyao.gpg
     ```

   - 按上文步骤1、步骤2重新执行导入与 `gpg --verify`。

   若按上述步骤重新获取并校验后仍失败，请联系openFuyao sig-installation邮箱[installation@list.openfuyao.cn](mailto:installation@list.openfuyao.cn)或openFuyao 社区邮箱 [contact@openfuyao.cn](mailto:contact@openfuyao.cn) ，并附上 `gpg --verify` 的完整输出。

3. 执行命令如下，校验安装包文件完整性。

   ```shell
   sha256sum -c <(cat bkeadm_linux_arm64.sha256) < bkeadm_linux_arm64
   ```

   命令执行完成后，若输出`-: OK`，则表示安装包文件完整性未破坏。否则说明安装包完整性已破坏，请重新下载安装包与对应的 `.sha256` / `.sha256.asc` 文件后再次校验；若仍失败，请联系openFuyao sig-installation邮箱[installation@list.openfuyao.cn](mailto:installation@list.openfuyao.cn)或openFuyao 社区邮箱 [contact@openfuyao.cn](mailto:contact@openfuyao.cn)。
