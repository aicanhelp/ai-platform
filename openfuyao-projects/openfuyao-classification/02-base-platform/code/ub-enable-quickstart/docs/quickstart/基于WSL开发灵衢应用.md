灵衢应用是基于华为最近一代计算机架构开发的应用，当前只有arm架构，要在windows本地开发，则需要借助WSL+Docker Buildx来实现，使得x86支持基于arm架构的代码开发编译。

## 安装 WSL2.0

### Windows配置

在 Windows 中启用 WSL（Windows Subsystem for Linux）和虚拟机平台功能，可以让你运行 Linux 命令行工具和应用程序。下面是详细步骤：

#### 方式一：手动启用功能

1. **打开 Windows 功能设置**
   控制面板 → 程序和功能 → 启用或关闭 Windows 功能。
2. **勾选以下选项**
   * ✅ 虚拟机平台（Virtual Machine Platform）
   * ✅ 适用于 Linux 的 Windows 子系统（Windows Subsystem for Linux）
3. **点击“确定”并重启电脑**

#### 方式二：使用命令手动启用功能

1. 以管理员身份打开 PowerShell，执行：
   ```powershell
   dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
   dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
   ```
2. 重启电脑

### 更新 WSL2 到最新版本

#### 方式一：在线更新

```powershell
wsl --update
```

#### 方式二：通过 GitHub 下载最新版本

```powershell
wsl --update --web-download
```

#### 方式三：手工更新（微软官方软件仓库）

* 从微软官方下载安装程序[WSL2 更新包下载（wsl_update_x64.msi）](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi) 双击运行即可完成安装。

#### 方式四：手工更新（微软官方 WSL 发布页）

去微软官方 WSL 发布页下载 `.msi` 安装包：

* [WSL 官方安装包（GitHub Releases）](https://github.com/microsoft/WSL/releases)
  下载最新的 `wsl_update_x64.msi`，双击安装即可。

### 设置默认 WSL 版本为 WSL2

```bash
wsl --set-default-version 2
```

### 安装 Linux 发行版（如 Ubuntu24.04）

#### 方式一：在线安装

1. 列出所有 WSL 发行版
   ```bash
   wsl --list --all
   ```
2. 在线安装
   ```bash
   wsl --install -d Ubuntu-24.04
   ```

#### 方式二：导入安装

1. 下载 [Ubuntu](https://ubuntu.com/desktop/wsl)安装包。
2. 使用 `wsl --import` 命令导入：

```bash
wsl --import Ubuntu-24.04 D:\WSL\Ubuntu-24.04 D:\WSL\Ubuntu-24.04\ubuntu-24.04.3-wsl-amd64.gz --version 2
```

> 假设下载后的文件名为：ubuntu-24.04.3-wsl-amd64.gz

3. 初始化系统

首次启动：

```powershell
wsl -d Ubuntu-24.04
```

按提示设置 Linux 用户名和密码。

#### 验证安装与版本

```bash
wsl -l -v
```

### WSL2 中启用微软官方支持的 systemd

在 WSL2 中使用支持 systemd 的 Ubuntu（如 Ubuntu 22.04 或更高版本），可以直接通过 apt 安装并运行 Docker，无需依赖 Docker Desktop。

**前提条件**

- Windows 11（建议 22H2 或更高，已支持 systemd）
- **WSL 版本**：`0.67.6` 或更高
  
  - 查看版本：
    
    ```powershell
    wsl --version
    ```
  - 如果版本低于要求或提示不支持 `--version`，请更新：
    
    ```powershell
    wsl --update
    ```
- 发行版需为 WSL2（`wsl -l -v` 查看）
- Ubuntu 22.04+ 已安装并设置为默认 WSL 发行版

#### 启用 systemd

编辑 WSL 配置文件 `/etc/wsl.conf`：

```bash
sudo nano /etc/wsl.conf
```

添加以下内容：

```ini
[boot]
systemd=true
```

然后在 Windows 中关闭并重启 WSL：

```powershell
wsl --shutdown
```

验证 systemd 是否启用：

```bash
ps --no-headers -o comm 1
```

如果输出是 `systemd`，说明已启用成功。
也可以：

```bash
systemctl list-units --type=service
```

## 安装 Docker

在 Ubuntu 或 Debian 系统中使用 `apt` 安装 Docker 是一种稳定且官方推荐的方式。以下是完整步骤，确保你安装的是最新版 Docker Engine：

### 一：卸载旧版本（如已安装）

```bash
sudo apt remove docker docker-engine docker.io containerd runc
```

### 二：更新系统并安装依赖

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y ca-certificates curl gnupg lsb-release
```

### 三：添加 Docker 官方 GPG 密钥

```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

### 四：添加 Docker 官方 APT 软件源

```bash
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

### 五：安装 Docker 引擎及组件

```bash
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io 
# sudo apt install -y docker-buildx-plugin docker-compose-plugin
```

### 六 启动并设置开机自启

```bash
sudo systemctl enable --now docker
```

### 七：验证安装是否成功

```bash
sudo docker run hello-world
```

如果看到 `Hello from Docker!` 的输出，说明安装成功。

### 八：（可选）配置非 root 用户使用 Docker

```bash
sudo usermod -aG docker $USER
newgrp docker  # 或重启终端以生效
```

### 九：（可选）公司内网访问外网场景

1. 获取公司自签名根证书
   
   - 向 IT 部门索取 `.crt` 或 `.pem` 格式的根证书文件（例如 `internet-proxy-ca.crt`）
   - 如果你只能通过浏览器访问代理，可以在浏览器里导出证书（通常是根证书或中间证书）
2. 安装证书到系统信任库
   
   Ubuntu 24.04 使用 `ca-certificates` 管理系统信任的 CA 证书。
   
   ```bash
   # 确保 ca-certificates 已安装
   sudo apt update
   sudo apt install -y ca-certificates
   
   # 将证书复制到系统 CA 目录
   sudo cp internet-proxy-ca.crt /usr/local/share/ca-certificates/
   
   # 更新证书信任库
   sudo update-ca-certificates
   ```
   
   执行后会提示：
   
   ```bash
   Updating certificates in /etc/ssl/certs...
   1 added, 0 removed; done.
   ```
   
   说明证书已被系统信任。
3. 重启 Docker

```bash
  sudo systemctl daemon-reexec
  sudo systemctl restart docker
```

### 十：安装docker buildx

在 Ubuntu 中安装 Docker 后，​**是否自动安装 Docker Buildx 取决于 Docker 的版本和安装方式**​

* **Docker 20.10.0+ 版本**​：`docker buildx` 已作为 **内置插件** 随 Docker 引擎一同安装（无需额外操作，直接可用）。
* **Docker < 20.10.0 版本**​：默认不安装 `buildx`，需手动安装。

#### 1. 验证是否已安装 buildx

在 Ubuntu 终端（WSL 或原生 Ubuntu）执行以下命令，即可判断：
```bash
docker buildx version
```

* 若输出类似 `github.com/docker/buildx v0.10.4 14b2e3a0a1d86153644e4144998f18ee7f50530d`，说明已安装。
* 若提示 `docker: 'buildx' is not a docker command`，说明未安装。

#### 2. 补充说明

##### （1）安装方式对版本的影响

* 若通过 Ubuntu 官方源安装（`sudo apt install docker.io`）：
  部分旧版 Ubuntu（如 20.04 早期版本）的官方源中，Docker 版本可能低于 20.10.0（例如 19.03.x），此时不会自动安装 `buildx`。
* 若通过 Docker 官方源安装（推荐，版本更新）：
  会默认安装 20.10.0+ 版本，自带 `buildx`。
  👉 官方源安装命令（确保版本最新）：bash
  
  ```bash
  # 卸载旧版（若有）
  sudo apt remove docker.io docker-compose docker-compose-v2
  
  # 添加 Docker 官方源
  sudo apt update
  sudo apt install -y ca-certificates curl gnupg lsb-release
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/trusted.gpg.d/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  
  # 安装最新版 Docker Engine
  sudo apt update
  sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  ```

##### （2）若未安装，手动安装 buildx（简单步骤）

若 Docker 版本低于 20.10.0，或未自带 `buildx`，执行以下命令手动安装：

```bash
# 下载最新版 buildx（适配 Linux amd64 架构）
mkdir -p ~/.docker/cli-plugins
curl -SL https://github.com/docker/buildx/releases/latest/download/buildx-linux-amd64 -o ~/.docker/cli-plugins/docker-buildx

# 赋予执行权限
chmod +x ~/.docker/cli-plugins/docker-buildx

# 验证安装
docker buildx version
```

## 导入arm架构开发镜像

### 验证 Buildx 支持的架构

```
docker buildx ls  # 查看当前构建器及支持的架构（如 linux/amd64、linux/arm64、linux/arm/v7 等）
```

### 直接拉取现成的 ARM 架构镜像

如果镜像仓库（如 Docker Hub）已提供 ARM 架构的镜像（如 `arm64`、`arm/v7`），可通过 `--platform` 指定架构拉取，无需构建。

1.​**拉取指定 ARM 架构的镜像**​：

拉取 `arm64` 架构（适用于 ARMv8 设备，如 Raspberry Pi 4、M1/M2 Mac）：

```bash
docker pull --platform linux/arm64 <镜像名>:<标签>
```

示例（拉取 ARM64 版本的 Ubuntu）：bash

```bash
docker pull --platform linux/arm64 ubuntu:22.04
```

2. 导入已有的arm镜像

```
docker load -i /path/to/arm64-image.tar  # 替换为你的 .tar 镜像路径
```

3. 验证镜像架构

```bash
docker inspect --format '{{.Architecture}} {{.Os}}/{{.Variant}}' <镜像名>:<标签>
```

输出示例（ARM64 镜像）：

```plaintext
arm64 linux/
```

## 开发代码

参考 [10分钟完成第一个UB开发小demo](https://gitcode.com/openFuyao/ub-enable-quickstart/blob/master/docs/quickstart/README.md)导入里面的ARM开发镜像，配置好远程链接即可在WSL环境上直接开发ARM应用。

