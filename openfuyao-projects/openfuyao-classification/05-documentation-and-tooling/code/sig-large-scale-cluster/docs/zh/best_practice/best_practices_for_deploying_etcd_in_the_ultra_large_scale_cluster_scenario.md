# 最佳实践

## 超大规模集群场景下etcd部署最佳实践

etcd是Kubernetes中核心组件，其主要作用如下。

- 配置存储：etcd存储Kubernetes集群的所有配置信息，包括节点信息、Pod信息、服务配置、网络配置等。它是Kubernetes API服务器的后端存储。
- 集群状态管理：etcd维护整个集群的状态数据，确保各个节点的状态信息在集群中一致。API服务器会从etcd获取和存储集群状态数据，从而实现集群管理和调度。
- 服务发现：etcd存储服务注册信息，使得集群内的服务能够相互发现和通信。Kubernetes使用这些信息来管理服务的生命周期和负载均衡。
- 高可用性和数据一致性：etcd使用Raft一致性算法来保证数据的一致性和高可用性。在多节点的etcd集群中，数据会自动在不同节点之间进行同步，从而确保即使某些节点发生故障，数据也不会丢失。

因此，etcd是Kubernetes的数据存储和分布式协调中心，确保了集群的可靠性和一致性。

### 目标

本最佳实践聚集在保障稳定性的条件下提升etcd性能，以支持超大规模场景下集群的正常运行。主要包含如下优化措施。

- 根据资源类型切分etcd集群，减少单个etcd实例的负载。
- 调整etcd配置参数。
- 使用性能更高的物理设备，调整资源和系统配置。

### 前提条件

系统将部署三个独立的etcd集群（每个集群3个节点），分别承载不同业务数据。

- etcd-pods集群：专用于存储Pod资源。  
- etcd-events集群：专用于存储Events与Leases资源。  
- etcd-data集群：用于存储除上述资源外的所有其他资源。  

需指定一台已配置好SSH密钥（可免密登录全部9个etcd节点）的服务器作为部署执行机，统一运行集群初始化与配置命令。

### 使用限制

- 支持Kubernetes v1.28.15，配套etcd v3.5.18。
- 支持Kubernetes v1.34.1，配套etcd v3.6.7。
- 3个etcd集群均为3节点组成；其中event集群数据存放内存；data集群和pod集群节点需要挂载高速SSD硬盘（8KB顺序IOPS ≥ 500，读写速度 ≥ 400MiB/s，最好是NVMe接口）。

### 背景信息

在超大规模集群场景下，etcd请求数量和大小成倍增长，存在如下性能瓶颈。

- 网络延迟：etcd使用Raft一致性算法，其性能受网络延迟影响较大。在大规模集群中，网络延迟会增加，导致请求处理时间增加。
- 磁盘IO延迟：etcd需要将数据同步到磁盘，磁盘IO延迟会影响性能。尤其是在使用传统硬盘（HDD）的情况下，磁盘IO延迟会更加明显。
- 写入吞吐量：随着集群规模增加，etcd的写入吞吐量可能会受到限制。高并发写入请求会导致请求处理时间增加，影响整体性能。
- 数据一致性维护：在大规模集群中，维护数据一致性的复杂度会增加。etcd需要处理大量的数据同步和复制操作，这会消耗更多的资源和时间。
- 资源管理：大规模集群需要更多的计算资源和存储资源，如果资源分配不当，可能会导致性能瓶颈。

### 操作步骤

操作步骤为半自动化安装etcd，需要在etcd集群中选择一个节点作为执行启动脚本节点，在启动脚本内已添加自动数据压缩与碎片化整理。

1. 以root身份登录执行启动脚本节点。

2. 安装etcd，部署etcd节点均需要安装。

    ```shell
    ARCH=$(uname -m)
    case $ARCH in
        x86_64) ARCH="amd64";;
        aarch64) ARCH="arm64";;
    esac
    VERSION="v3.5.18"

    # 下载etcd安装包
    wget https://openfuyao.obs.cn-north-4.myhuaweicloud.com/etcd-io/etcd/releases/download/${VERSION}/etcd-${VERSION}-linux-${ARCH}.tar.gz

    # 安装etcd
    tar -xvf etcd-"${VERSION}"-linux-${ARCH}.tar.gz
    cp -rf etcd-"${VERSION}"-linux-${ARCH}/etcd* /usr/local/bin/
    chmod +x /usr/local/bin/{etcd,etcdctl,etcdutl}
    ```

3. 安装yq工具，只在执行启动脚本节点安装。

    ```shell
    ARCH=$(uname -m)
    case $ARCH in
        x86_64) ARCH="amd64";;
        aarch64) ARCH="arm64";;
    esac

    wget https://openfuyao.obs.cn-north-4.myhuaweicloud.com/mikefarah/yq/releases/download/v4.43.1/yq_linux_${ARCH}

    cp -f yq_linux_${ARCH} /usr/local/bin/
    chmod +x /usr/local/bin/yq
    ```

4. 安装step工具，只在执行启动脚本节点安装。

    ```shell
    ARCH=$(uname -m)
    case $ARCH in
        x86_64) ARCH="amd64";;
        aarch64) ARCH="arm64";;
    esac
    VERSION="0.28.2"

    wget https://openfuyao.obs.cn-north-4.myhuaweicloud.com/smallstep/cli/releases/download/v${VERSION}/step_linux_"${VERSION}"_${ARCH}.tar.gz

    tar -xvf step_linux_"${VERSION}"_${ARCH}.tar.gz
    mv step_"${VERSION}"/bin/step /usr/local/bin/step
    chmod +x /usr/local/bin/step
    ```

5. 配置免密登录，配置执行启动脚本节点到其他节点的免密登录。

    ```shell
    # 生成公钥，有提示直接回车即可
    ssh-keygen

    # 上传登录公钥到其他etcd节点
    for ip in <节点ip地址, eg:192.168.200.238 192.168.200.237 192.168.200.236 192.168.200.235 192.168.200.234 192.168.200.233 192.168.200.232 192.168.200.231 192.168.200.230>; do
        # 此处会提示输入密码，可直接输入密码并按回车
        ssh-copy-id -i ~/.ssh/id_rsa.pub root@${ip}
    done
    ```

6. 安装基础组件，所有etcd节点均安装。

    ```shell
    yum install -y systemd-pam
    ```

7. 在执行启动脚本节点上保存下面启动脚本到etcd-bootstrap.sh，[单击获取启动脚本etcd-bootstrap.sh](#启动脚本etcd-bootstrapsh配置详解)。

8. 执行如下命令，在执行启动脚本节点上执行etcd-bootstrap.sh启动etcd data集群。

    ```shell
    mkdir /root/etcd-install

    # 需要替换真实ip地址
    bash etcd-bootstrap.sh <etcd data节点ip, eg: 192.168.200.238 192.168.200.237 192.168.200.236> -c /root/etcd-install -p etcdData
    ```

9. 执行如下命令，在执行启动脚本节点上执行etcd-bootstrap.sh启动etcd pod集群。

    ```shell
    mkdir /root/etcd-install

    # 需要替换真实ip地址
    bash etcd-bootstrap.sh <etcd data节点ip, eg: 192.168.200.235 192.168.200.234 192.168.200.233> -c /root/etcd-install -p etcdPods
    ```

10. 执行如下命令，在执行启动脚本节点上执行etcd-bootstrap.sh启动etcd events-leases集群。

    ```shell
    mkdir /root/etcd-install

    # 需要替换真实ip地址
    bash etcd-bootstrap.sh <etcd data节点ip, eg: 192.168.200.232 192.168.200.231 192.168.200.230> -c /root/etcd-install -p etcdPods --use-tmpfs
    ```

11. 确认etcd集群状态。

- 在所有etcd节点执行`systemctl status etcd`，若出现`running`字样则说明etcd已经启动。
- 在所有etcd节点执行如下命令查看etcd集群是否健康，若输出表格中`HEALTH`列均为true，则表明集群是健康状态。
    ```shell
    ETCDCTL_API=3 etcdctl --endpoints=https://127.0.0.1:2379 \
    --cacert=/usr/local/share/etcd/ca.crt \     # 替换为实际证书地址
    --cert=/usr/local/share/etcd/server.crt \   # 替换为实际证书地址
    --key=/usr/local/share/etcd/server.key \    # 替换为实际证书地址
    endpoint health --write-out=table
    ```

### 结论

此部署形态和参数调优已经过模拟测试，可支撑1.6w节点的K8s集群稳定运行，下表是根据模拟测试结果给出的推荐参数优化配置。

**表1** etcd关键性能参数分析

| 参数 | 描述                                                                             | 值 |
|-|--------------------------------------------------------------------------------|-|
| snapshot-count | 触发一次snapshot的最大transaction数量；增加该值会导致内存/磁盘占用增加，减少该值会导致磁盘频繁IO，时延增加。              | etcd-data：100，etcd-event：1 |
| heatbeat-interval | 心跳时间间隔；建议配置范围：$(0.5 * rtt_{max}, 1.5 * rtt_{max})$，其中$rtt_{max}$为所有节点间rtt的最大值。 | 根据环境配置 |
| election-timeout | 选举超时时间；建议配置范围：$(5 * rtt_{max}, 15 * rtt_{max})$，其中$rtt_{max}$为所有节点间rtt的最大值。    | 根据环境配置 |
| max-snapshots | 磁盘上保留的最大snapshot数量。                                                            | etcd-data：10，etcd-event：1 |
| max-wals | 磁盘上保留的最大wal文件数量，只有已经snapshot的wal文件才可以被删除。                                      | etcd-data：10，etcd-event：1 |
| quota-backend-bytes | DB最大占用空间，超过该值后会导致写入失败。                                                         | etcd-data：68719476736（64GiB），etcd-event：8589934592（8GiB） |
| backend-batch-interval | transaction commit最大时间间隔；增加该值会降低时延，但是对磁盘性能要求较高。                                | etcd-data：10000000，etcd-event：10000 |
| backend-batch-limit | 触发一次commit的最大transaction数量；增加该值会降低时延，但是对磁盘性能要求较高。                              | etcd-data：100，etcd-event：1 |
| max-txn-ops | 单次transaction包含的最大op数量。                                                        | 16000 |
| max-request-bytes | 单次请求最大数据量。                                                                     | etcd-data：128000000（128MB），etcd-event：16000000（16MB） |
| max-concurrent-streams | 单个client上允许的最大并发stream数量。                                                      | 1024 |
| auto-compaction-retention | 何时执行自动compact；设置为0则关闭自动compact。                                                | etcd-data：0，etcd-event：5m |
| auto-compaction-mode | 自动compact模式；设置为periodic则定时执行，设置为revision则每隔固定revision执行。                       | etcd-data：空，etcd-event：periodic |
| unsafe-no-fsync | 是否禁用fdatasync()。                                                               | etcd-data：false，etcd-event：true |


### 参考资料

  [etcd官方安装文档](https://etcd.io/docs/v3.6/install/)

### 附录
#### 启动脚本etcd-bootstrap.sh配置详解
```shell
#!/bin/bash
###############################################################
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# installer is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
###############################################################

main() {
    local workspace
    local version=3.5.18
    local hosts=()
    local prefix=etcd
    local use_tmpfs=false
    local out_certs=.
    while (($# > 0)); do
        case "$1" in
            -c|--workspace) workspace=$2; shift;;
            --workspace=*) workspace=${1#--workspace=};;
            -V|--version) version=$2; shift;;
            --version=*) version=${1#--version=};;
            -p|--prefix) prefix=$2; shift;;
            --prefix=*) prefix=${1#--prefix=};;
            -t|--use-tmpfs) use_tmpfs=true;;
            -o|--out-certs) out_certs=$2; shift;;
            --out-certs=*) out_certs=${1#--out-certs=};;
            --);;
            -*|--*) echo "Unknown option: $1"; exit 1;;
            *) hosts+=("$1");;
        esac
        shift
    done

    local config=$(get_embedded etcd-config)
    local script=$(get_embedded client-script)
    local defrag_script=$(get_embedded defrag-script)

    os=$(get_os)
    arch=$(get_arch)
    . <(get_embedded pkgman)
    prepare_ws "$workspace"
    check_prerequisites "$version" "$os" "$arch"
    gen_jwt_auth
    gen_etcd_ca
    echo "$defrag_script" > "local/bin/etcd-defrag.sh"
    chmod +x "local/bin/etcd-defrag.sh"
    local config=$(update_config "$config" "$use_tmpfs" "$(step crypto rand --format=hex)")
    local i host name args
    for i in "${!hosts[@]}"; do
        host=${hosts[$i]}
        name=$prefix-$i
        args=("$script" _ "$use_tmpfs")
        gen_etcd_certs "$name" "$host"
        ssh -o StrictHostKeyChecking=no root@"$host" bash -c "$(printf '%q ' "${args[@]}")" < <(
            create_payload "$(update_peer_config "$config" "$name" "$host")"
        )
    done
    gen_apiserver_cert
    output_certs "$out_certs"

    exit
}

get_os() {
    local os=$(uname | tr '[:upper:]' '[:lower:]')
    case "$os" in
        darwin) echo 'darwin';;
        linux) echo 'linux';;
        freebsd) echo 'freebsd';;
        mingw*|msys*|cygwin*) echo 'windows';;
        *) echo "Unsupported OS: ${os}" >&2; exit 1;;
    esac
}

get_arch() {
    local arch=$(uname -m)
    case "$arch" in
        amd64|x86_64) echo 'amd64';;
        i386) echo '386';;
        ppc64) echo 'ppc64';;
        ppc64le) echo 'ppc64le';;
        s390x) echo 's390x';;
        armv6*|armv7*) echo 'arm';;
        aarch64) echo 'arm64';;
        *) echo "Unsupported architecture: ${arch}" >&2; exit 1;;
    esac
}

get_embedded() {
    local embedded=$1
    sed -n "/^# >>>>> BEGIN $embedded\$/,/^# <<<<< END $embedded\$/{//!p}" "$0" | head -n-1 | tail -n+2
}

prepare_ws() {
    local workspace=$1
    local ws=${workspace:-$(mktemp -d)}
    export PATH=$PATH:$ws/bin
    [ -z "$workspace" ] &&
        trap "{
            cd /
            rm -rf '$ws'
        }" EXIT
    mkdir -p "$ws" && cd "$ws"
    mkdir -p bin local/{bin,{etc,share}/etcd}
}

check_prerequisites() {
    local version=$1
    local os=$2
    local arch=$3
    cat <<'EOF' > "step-ca.json"
{
    "subject": {{toJson .Subject}},
    "issuer": {{toJson .Subject}},
    "keyUsage": ["digitalSignature", "keyEncipherment", "certSign"],
    "basicConstraints": {
        "isCA": true
    }
}
EOF
    cat <<'EOF' > "step-leaf.json"
{
    "subject": {{toJson .Subject}},
    "sans": {{toJson .SANs}},
    "keyUsage": ["digitalSignature", "keyEncipherment"],
    "extKeyUsage": ["serverAuth", "clientAuth"]
}
EOF
}

gen_jwt_auth() {
    step crypto keypair local/share/etcd/jwt_ec384{.pub,} \
        --kty=EC --crv=P-384 \
        -f --insecure --no-password
}

gen_etcd_ca() {
    [ -f "etcd-ca.crt" ] && [ -f "etcd-ca.key" ] ||
        step certificate create etcd-ca etcd-ca.{crt,key} \
            --kty=OKP --crv=Ed25519 \
            --not-after=87600h \
            --template "step-ca.json" \
            -f --insecure --no-password
    cp -alf {etcd-,local/share/etcd/}ca.crt
}

gen_etcd_certs() {
    local name=$1
    local host=$2
    step certificate create "$name" local/share/etcd/server.{crt,key} \
        --kty=OKP --crv=Ed25519 \
        --ca="etcd-ca.crt" --ca-key="etcd-ca.key" \
        --not-after=87600h \
        --san="$name" --san=localhost --san=127.0.0.1 --san=0:0:0:0:0:0:0:1 --san="$host" \
        --template "step-leaf.json" \
        -f --insecure --no-password
    step certificate create "$name" local/share/etcd/peer.{crt,key} \
        --kty=OKP --crv=Ed25519 \
        --ca="etcd-ca.crt" --ca-key="etcd-ca.key" \
        --not-after=87600h \
        --san="$name" --san=localhost --san=127.0.0.1 --san=0:0:0:0:0:0:0:1 --san="$host" \
        --template "step-leaf.json" \
        -f --insecure --no-password
}

gen_apiserver_cert() {
    step certificate create apiserver-etcd-client apiserver-etcd-client.{crt,key} \
        --kty=OKP --crv=Ed25519 \
        --ca="etcd-ca.crt" --ca-key="etcd-ca.key" \
        --not-after=87600h \
        --template "step-leaf.json" \
        -f --insecure --no-password
}

create_payload() {
    local config=$1
    echo "$config" > "local/etc/etcd/config.yaml.tmpl"
    tar Cczf "local" - bin etc share
}

output_certs() {
    local out=$1
    tar czf "$out/etcd-certs.tar.gz" {etcd-ca,apiserver-etcd-client}.{crt,key}
}

update_config() {
    local config=$1
    local use_tmpfs=$2
    local token=$3
    local i cluster
    for i in "${!hosts[@]}"; do
        cluster="$cluster$prefix-$i=https://${hosts[$i]}:2380,"
    done
    cluster=${cluster::-1}
    config=$(yq "
        .initial-cluster = \"$cluster\" |
        .initial-cluster-token = \"$token\"
    " <<< "$config")
    if "$use_tmpfs"; then
        yq "
            .quota-backend-bytes = 8589934592 |
            .backend-batch-interval = 10000000 |
            .backend-batch-limit = 100 |
            .auto-compaction-mode = \"periodic\"
        " <<< "$config"
    else
        yq "
            .quota-backend-bytes = 68719476736 |
            .backend-batch-interval = 100000000 |
            .backend-batch-limit = 1000 |
            .auto-compaction-mode = \"\"
        " <<< "$config"
    fi
}

update_peer_config() {
    local config=$1
    local name=$2
    local host=$3
    yq "
        .name = \"$name\" |
        .listen-peer-urls = \"https://$host:2380\" |
        .listen-client-urls = \"https://$host:2379,https://localhost:2379\" |
        .initial-advertise-peer-urls = \"https://$host:2380\" |
        .advertise-client-urls = \"https://$host:2379\"
    " <<< "$config"
}

main "$@"

# >>>>> BEGIN client-script

set -e

# >>>>> BEGIN pkgman

has_cmd() {
    command -v "$1" &> /dev/null
}

_install_pkg_apt() {
    apt install -y --no-install-recommends "$@"
}

_install_pkg_dnf() {
    dnf install -y --setopt=install_weak_deps=False "$@"
}

_has_pkg_apt() {
    dpkg --get-selections | awk '{print $1}' | grep -qE "^$1(:|$)"
}

_has_pkg_dnf() {
    dnf list --installed | awk -F. '{print $1}' | grep -qE "^$1$"
}

_pkg_of_file_apt() {
    local file=$1
    pkg=$(dpkg -S "$file" | awk -F: '{print $1}')
    if [ -z "$pkg" ]; then
        echo "No package found for file: $file"
        exit 1
    fi
    echo "$pkg"
}

_pkg_of_file_dnf() {
    local file=$1
    if ! dnf repoquery -q --whatprovides "$file" --qf '%{name}'; then
        echo "No package found for file: $file"
        exit 1
    fi
}

shopt -s expand_aliases
for pkgman in apt dnf; do
    if has_cmd "$pkgman"; then
        alias install_pkg="_install_pkg_$pkgman"
        alias has_pkg="_has_pkg_$pkgman"
        alias pkg_of_file="_pkg_of_file_$pkgman"
        break
    fi
done
if ! has_cmd install_pkg; then
    echo "Unsupported package manager"
    exit 1
fi

# <<<<< END pkgman

use_tmpfs=$1

if [ "$(id -u)" != 0 ]; then
    echo "client install script must be run as root"
    exit 1
fi

has_cmd systemctl ||
    install_pkg systemd
pkg=$(pkg_of_file '*/pam_systemd.so')
has_pkg "$pkg" ||
    install_pkg "$pkg"
has_cmd envsubst ||
    install_pkg gettext
has_cmd tar ||
    install_pkg tar
has_cmd python3 ||
    install_pkg python3
has_cmd mkfs.xfs ||
    install_pkg xfsprogs
systemctl daemon-reload

export ETCD_DATA_DIR=/usr/local/share/etcd
export ETCD_CONFIG_DIR=/usr/local/etc/etcd
export ETCD_STATE_DIR=/var/lib/etcd
export ETCD_LOG_DIR=/var/log/etcd

[ -f "$ETCD_STATE_DIR/.disk-uuid" ] &&
    uuid=$(< "$ETCD_STATE_DIR/.disk-uuid")

rm -rf --one-file-system "$ETCD_STATE_DIR" || true
rm -rf "$ETCD_STATE_DIR/member"/{.,}* || true

mkdir -p "$ETCD_STATE_DIR"
echo "$uuid" > "$ETCD_STATE_DIR/.disk-uuid"
tar Cxzf "/usr/local" -

units=(etcd-defrag.timer etcd.service var-lib-etcd-member.mount)
for unit in "${units[@]}"; do
    unit_file=/etc/systemd/system/$unit
    if [ -f "$unit_file" ]; then
        systemctl disable --now "$unit" || true
        rm -f "$unit_file"
    fi
done
systemctl daemon-reload

if grep -q "$ETCD_STATE_DIR/member " /etc/mtab; then
    umount "$ETCD_STATE_DIR/member" || true
    sed -i "\\:$ETCD_STATE_DIR/member :d" /etc/fstab
fi

envsubst < "$ETCD_CONFIG_DIR/config.yaml.tmpl" > "$ETCD_CONFIG_DIR/config.yaml"

cat <<EOF > /usr/local/sbin/etcd-tune.sh
#!/bin/bash -x
[ -e /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ] &&
    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null
EOF
chmod +x /usr/local/sbin/etcd-tune.sh
cat <<EOF > /etc/systemd/system/etcd-tune.service
[Unit]
Description=etcd tuning
After=local-fs.target var-lib-etcd-member.mount
Wants=local-fs.target var-lib-etcd-member.mount

[Service]
ExecStart=/usr/local/sbin/etcd-tune.sh
Type=oneshot
RemainAfterExit=yes

[Install]
WantedBy=default.target
EOF

chmod 700 "$ETCD_STATE_DIR"
mkdir -p "$ETCD_STATE_DIR/member"
if "$use_tmpfs"; then
    cat <<EOF > "/etc/systemd/system/var-lib-etcd-member.mount"
[Unit]
Description=etcd data disk
Before=local-fs.target

[Mount]
What=tmpfs
Where=$ETCD_STATE_DIR/member
Type=tmpfs
Options=nosuid,nodev,uid=0,gid=0,mode=700,size=16384M
TimeoutSec=60s

[Install]
WantedBy=multi-user.target
EOF
else
    [ -n "$uuid" ] && [ -h "/dev/disk/by-uuid/$uuid" ] &&
        dev=$(realpath "/dev/disk/by-uuid/$uuid")
    if [ -z "$dev" ] || [ "$(blkid "$dev" | sed -E 's/.* TYPE="([^"]+)".*/\1/')" != xfs ]; then
        dev=
        for blk in $(lsblk -o NAME,MOUNTPOINT | awk '{if ($2 == "") print $1}'); do
            set +e
            [ -b "/dev/$blk" ] &&
                blkid "/dev/$blk"
            status=$?
            set -e
            if [ "$status" == 2 ]; then
                dev=/dev/$blk
                echo "found unpartitioned disk: $dev"
                uuid=$(cat /proc/sys/kernel/random/uuid)
                mkfs.xfs -f "$dev" -m "uuid=$uuid"
                echo "$uuid" > "$ETCD_STATE_DIR/.disk-uuid"
                udevadm settle
                # 挂载磁盘
                cat <<EOF > "/etc/systemd/system/var-lib-etcd-member.mount"
[Unit]
Description=etcd data disk
Before=local-fs.target

[Mount]
What=/dev/disk/by-uuid/$uuid
Where=$ETCD_STATE_DIR/member
Type=xfs
Options=nosuid,nodev,noatime,nodiratime
TimeoutSec=60s

[Install]
WantedBy=multi-user.target
EOF
              serial=$(udevadm info -n "$dev" | grep ID_SERIAL= | awk -F= '{print $2}')
              mkdir -p /usr/local/sbin
              # 设置磁盘为写直通模式，避免数据丢失
              cat <<EOF >> /usr/local/sbin/etcd-tune.sh
serial=$serial
devname=\$(find /dev/disk/by-id -regex ".*-\$serial$")
devpath=/sys\$(udevadm info -n "\$devname" | grep devpath= | awk -F= '{print \$2}')
echo 'write through' > "\$(find -L "\$devpath" -name cache_type -print -quit 2> /dev/null)"
EOF
                break
            fi
        done
        if [ -z "$dev" ]; then
            echo 'no unpartitioned disk found, use / to store etcd data'
            mkdir -p "$ETCD_STATE_DIR/member"
            dev="$ETCD_STATE_DIR/member"
        fi
    fi
fi
echo 'exit 0' >> /usr/local/sbin/etcd-tune.sh

mkdir -p "/etc/systemd/system"
cat <<EOF > "/etc/systemd/system/etcd.service"
[Unit]
Description=etcd
After=network-online.target local-fs.target remote-fs.target time-sync.target
Wants=network-online.target local-fs.target remote-fs.target time-sync.target

[Service]
Type=simple
ExecStart=/usr/local/bin/etcd --config-file=$ETCD_CONFIG_DIR/config.yaml
TimeoutSec=0
Restart=always
RestartSec=3
StartLimitBurst=20
StartLimitInterval=60s
#LimitNOFILE=infinity
#LimitNPROC=infinity
#LimitCORE=infinity
#TasksMax=infinity
Delegate=yes
KillMode=mixed
# 设置CPU优先级
CPUSchedulingPolicy=rr
CPUSchedulingPriority=99
# 设置IO优先级
IOSchedulingClass=realtime
IOSchedulingPriority=0

[Install]
WantedBy=default.target
EOF
cat <<EOF > "/etc/systemd/system/etcd-defrag.service"
[Unit]
Description=etcd auto compact/defrag
After=etcd.service
Wants=etcd.service etcd-defrag.timer

[Service]
Environment=ETCD_CONFIG_DIR=$ETCD_CONFIG_DIR
Environment=ETCDCTL_CACERT=$ETCD_DATA_DIR/ca.crt
Environment=ETCDCTL_CERT=$ETCD_DATA_DIR/server.crt
Environment=ETCDCTL_KEY=$ETCD_DATA_DIR/server.key
ExecStart=/usr/local/bin/etcd-defrag.sh
Type=oneshot

[Install]
WantedBy=default.target
EOF
cat <<EOF > "/etc/systemd/system/etcd-defrag.timer"
[Unit]
Description=etcd auto compact/defrag timer

[Timer]
Unit=etcd-defrag.service
OnCalendar=*-*-* *:00/5:00

[Install]
WantedBy=timers.target
EOF

for file in /etc/{bash.bashrc,profile.d/etcdctl.sh}; do
    [ -f "$file" ] &&
        sed -i '/^# BEGIN external-etcd-envs$/,/^# END external-etcd-envs$/d' "$file"
    cat <<EOF >> "$file"

# BEGIN external-etcd-envs
# The following lines are managed by external etcd installer, please do not modify them manually.
export ETCD_DATA_DIR=$ETCD_DATA_DIR
export ETCD_CONFIG_DIR=$ETCD_CONFIG_DIR
export ETCD_STATE_DIR=$ETCD_STATE_DIR
export ETCD_LOG_DIR=$ETCD_LOG_DIR
export ETCDCTL_CACERT=\$ETCD_DATA_DIR/ca.crt
export ETCDCTL_CERT=\$ETCD_DATA_DIR/server.crt
export ETCDCTL_KEY=\$ETCD_DATA_DIR/server.key
# END external-etcd-envs
EOF
done

mkdir -p "$ETCD_LOG_DIR"
systemctl daemon-reload
mapfile -td '' units < <(printf '%s\0' "${units[@]}" | tac -s '')
for unit in "${units[@]}"; do
    if systemctl list-unit-files | grep -q "^$unit"; then
        systemctl enable --now "$unit"
    else
        echo "Warning: unit $unit not found, skipping"
    fi
done

# <<<<< END client-script

# >>>>> BEGIN etcd-config

# Human-readable name for this member.
name: etcd

# Path to the data directory.
data-dir: ${ETCD_STATE_DIR}

# Path to the dedicated wal directory.
# wal-dir: ${ETCD_STATE_DIR}/member-wal/wal

# List of URLs to listen on for peer traffic.
listen-peer-urls: https://localhost:2380

# List of URLs to listen on for client grpc traffic (and http as long as --listen-client-http-urls is not specified).
listen-client-urls: https://localhost:2379

# List of this member's peer URLs to advertise to the rest of the cluster.
initial-advertise-peer-urls: https://localhost:2380

# List of this member's client URLs to advertise to the public. The client URLs advertised should be accessible to
# machines that talk to etcd cluster. etcd client libraries parse these URLs to connect to the cluster.
advertise-client-urls: https://localhost:2379

# Initial cluster configuration for bootstrapping.
initial-cluster: etcd-0=https://etcd-0:2380,etcd-1=https://etcd-1:2380,etcd-2=https://etcd-2:2380

# Initial cluster state ('new' when bootstrapping a new cluster or 'existing' when adding new members to an existing
# cluster). After successful initialization (bootstrapping or adding), flag is ignored on restarts.
initial-cluster-state: new

# Initial cluster token for the etcd cluster during bootstrap. Specifying this can protect you from unintended
# cross-cluster interaction when running multiple clusters.
initial-cluster-token: random-token

# Number of committed transactions to trigger a snapshot to disk.
snapshot-count: 100000 # **

# Time (in milliseconds) of a heartbeat interval.
heartbeat-interval: 250 # **

# Time (in milliseconds) for an election to timeout. See tuning documentation for details.
election-timeout: 2500 # **

# Whether to fast-forward initial election ticks on boot for faster election.
initial-election-tick-advance: true

# Maximum number of snapshot files to retain (0 is unlimited).
max-snapshots: 10 # **

# Maximum number of wal files to retain (0 is unlimited).
max-wals: 10 # **

# Raise alarms when backend size exceeds the given quota (0 defaults to low space quota).
quota-backend-bytes: 34359738368 # **

# Maximum time before commit the backend transaction.
backend-batch-interval: 100000000 # **

# Maximum operations before commit the backend transaction.
backend-batch-limit: 1000 # **

# Maximum number of operations permitted in a transaction.
max-txn-ops: 16000 # **

# Maximum client request size in bytes the server will accept.
max-request-bytes: 128000000 # **

# Maximum concurrent streams that each client can open at a time.
max-concurrent-streams: 20000 # **

# Enable GRPC gateway.
enable-grpc-gateway: true

# Minimum duration interval that a client should wait before pinging server.
grpc-keepalive-min-time: 5000000000 # **

# Frequency duration of server-to-client ping to check if a connection is alive (0 to disable).
grpc-keepalive-interval: 7200000000000 # **

# Additional duration of wait before closing a non-responsive connection (0 to disable).
grpc-keepalive-timeout: 20000000000 # **

# Enable to run an additional Raft election phase.
pre-vote: true

# Auto compaction retention length. 0 means disable auto compaction.
auto-compaction-retention: '0' # **

# Interpret 'auto-compaction-retention', one of: periodic|revision. 'periodic' for duration based retention, defaulting
# to hours if no time unit is provided (e.g. '5m'). 'revision' for revision number based retention.
auto-compaction-mode: periodic # **

client-transport-security:
  # Path to the client server TLS cert file.
  cert-file: ${ETCD_DATA_DIR}/server.crt

  # Path to the client server TLS key file.
  key-file: ${ETCD_DATA_DIR}/server.key

  # Enable client cert authentication.
  client-cert-auth: true

  # Path to the client server TLS trusted CA cert file.
  trusted-ca-file: ${ETCD_DATA_DIR}/ca.crt

peer-transport-security:
  # Path to the peer server TLS cert file.
  cert-file: ${ETCD_DATA_DIR}/peer.crt

  # Path to the peer server TLS key file.
  key-file: ${ETCD_DATA_DIR}/peer.key

  # Enable peer client cert authentication.
  client-cert-auth: true

  # Path to the peer server TLS trusted CA cert file.
  trusted-ca-file: ${ETCD_DATA_DIR}/ca.crt

# List of supported TLS cipher suites between client/server and peers (empty will
# be auto-populated by Go).
cipher-suites:
  - TLS_AES_128_GCM_SHA256
  - TLS_AES_256_GCM_SHA384
  - TLS_CHACHA20_POLY1305_SHA256
  - TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
  - TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
  - TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256
  - TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256
  - TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384
  - TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256

# Minimum TLS version supported by etcd. Possible values: TLS1.2, TLS1.3.
tls-min-version: TLS1.2

# Maximum TLS version supported by etcd. Possible values: TLS1.2, TLS1.3 (empty will be auto-populated by Go).
tls-max-version: TLS1.3

# Specify a v3 authentication token type and its options ('simple' or 'jwt').
auth-token: jwt,pub-key=${ETCD_DATA_DIR}/jwt_ec384.pub,priv-key=${ETCD_DATA_DIR}/jwt_ec384,sign-method=ES384,ttl=3600s

# Specify the cost / strength of the bcrypt algorithm for hashing auth passwords. Valid values are between 4 and 31.
bcrypt-cost: 10

# Currently only supports 'zap' for structured logging.
logger: zap

# Specify 'stdout' or 'stderr' to skip journald logging even when running under systemd, or list of output targets.
log-outputs:
  - ${ETCD_LOG_DIR}/etcd.log

# Configures log level. Only supports debug, info, warn, error, panic, or fatal.
log-level: info

# Enable log rotation of a single log-outputs file target.
enable-log-rotation: true

# Configures log rotation if enabled with a JSON logger config. MaxSize(MB), MaxAge(days, 0=no limit),
# MaxBackups(0=no limit), LocalTime(use computers local time), Compress(gzip).
log-rotation-config-json: '{"maxsize": 128, "maxage": 7, "maxbackups": 1024, "localtime": true, "compress": true}'

# ExperimentalEnableLeaseCheckpoint enables primary lessor to persist lease remainingTTL to prevent indefinite
# auto-renewal of long lived leases.
experimental-enable-lease-checkpoint: true

# Enable persisting remainingTTL to prevent indefinite auto-renewal of long lived leases. Always enabled in v3.6.
# Should be used to ensure smooth upgrade from v3.5 clusters with this feature enabled. Requires
# experimental-enable-lease-checkpoint to be enabled.
experimental-enable-lease-checkpoint-persist: true

# Disables fsync, unsafe, will cause data loss.
unsafe-no-fsync: false

# <<<<< END etcd-config

# >>>>> BEGIN defrag-script

#!/bin/bash -x
SNAPSHOT_THRESHOLD=${SNAPSHOT_THRESHOLD:-90}
DEFRAG_THRESHOLD=${DEFRAG_THRESHOLD:-90}
(( SNAPSHOT_THRESHOLD >= 100 )) &&
    SNAPSHOT_THRESHOLD=90
(( SNAPSHOT_THRESHOLD <= 0 )) &&
    SNAPSHOT_THRESHOLD=90
(( DEFRAG_THRESHOLD >= 100 )) &&
    DEFRAG_THRESHOLD=90
(( DEFRAG_THRESHOLD <= 0 )) &&
    DEFRAG_THRESHOLD=90
. "$HOME/.profile"
disk_quota=$(yq -r '.quota-backend-bytes' "$ETCD_CONFIG_DIR/config.yaml")
read disk_size db_size revision < <(
    etcdctl endpoint status -w json | yq -r '.0.Status | .dbSize + " " + .dbSizeInUse + " " + .header.revision'
)
db_usage=$((100 * db_size / disk_size))
disk_usage=$((100 * disk_size / disk_quota))
(( db_usage >= SNAPSHOT_THRESHOLD )) &&
    etcdctl compact "$revision"
(( disk_usage >= DEFRAG_THRESHOLD )) &&
    etcdctl defrag
exit 0

# <<<<< END defrag-script

```
