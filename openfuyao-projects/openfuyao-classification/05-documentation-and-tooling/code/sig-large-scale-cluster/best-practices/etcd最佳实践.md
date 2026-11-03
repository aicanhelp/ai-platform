# 最佳实践

## 超大规模集群场景下etcd部署最佳实践

在Kubernetes中，etcd是一个关键组件，主要作用如下：

- 配置存储：etcd存储Kubernetes集群的所有配置信息，包括节点信息、Pod信息、服务配置、网络配置等。它是Kubernetes API服务器的后端存储。
- 集群状态管理：etcd维护整个集群的状态数据，确保各个节点的状态信息在集群中一致。API服务器会从etcd获取和存储集群状态数据，从而实现集群管理和调度。
- 服务发现：etcd存储服务注册信息，使得集群内的服务能够相互发现和通信。Kubernetes使用这些信息来管理服务的生命周期和负载均衡。
- 高可用性和数据一致性：etcd使用Raft一致性算法来保证数据的一致性和高可用性。在多节点的etcd集群中，数据会自动在不同节点之间进行同步，从而确保即使某些节点发生故障，数据也不会丢失。

总的来说，etcd是Kubernetes的数据存储和分布式协调中心，确保了集群的可靠性和一致性。

<!-- 在超大规模AI训练任务场景下，集群中节点数量（16k+）远大于Kubernetes官方推荐的最大节点数（5k），因此etcd常规配置支持的存储大小和性能无法满足要求。 -->

### 目标

本最佳实践聚集在保障稳定性的条件下提升etcd性能，以支持超大规模场景下集群的正常运行。主要包含如下优化措施：

- 根据资源类型切分etcd集群，减少单个etcd实例的负载；
- 调整etcd配置参数；
- 使用性能更高的物理设备，调整资源和系统配置；

<!-- - 降低etcd请求时延，提高吞吐量，支撑超大规模集群场景 -->

### 前提条件(可选)

集群使用2个外部etcd集群作为分布式kv存储；一个负责存储对可靠性要求较低的event资源（下文中简称为event集群），另一个负责存储对可靠性要求较高的其他资源（下文中简称为data集群）。同时需要准备1台服务器来执行etcd集群安装命令（可以是6台etcd节点之一），该服务器上已配置好上述6个节点的SSH登录密钥。

<!--
- kubernetes集群部署之前
- 准备6台服务器供etcd集群使用；其中3台为16U16G，用于存放事件资源；另外3台为16U128G，用于存放其他资源
- 准备1台服务器执行安装命令（可以是6台etcd服务器之一），该服务器上已配置好上述6台服务器的SSH登录密钥
- 存放其他资源的3台机器需要挂载SSD磁盘，8K顺序IOPS ≥ 500，读写速度 ≥ 400MiB/s
-->

### 使用限制(可选)

- 本最佳实践基于Kubernetes v1.28版本、etcd v3.5.17版本进行分析；
- 2个etcd集群均为3节点组成；其中event集群节点规格为12U16G；data集群节点规格为16U128G，需要挂载高速SSD硬盘（8KB顺序IOPS ≥ 500，读写速度 ≥ 400MiB/s，最好是NVMe接口）；

### 背景信息(可选)

在超大规模集群场景下，etcd请求数量和大小成倍增长，存在如下性能瓶颈：

- 网络延迟：etcd使用Raft一致性算法，其性能受网络延迟影响较大。在大规模集群中，网络延迟会增加，导致请求处理时间增加。
- 磁盘IO延迟：etcd需要将数据同步到磁盘，磁盘IO延迟会影响性能。尤其是在使用传统硬盘（HDD）的情况下，磁盘IO延迟会更加明显。
- 写入吞吐量：随着集群规模增加，etcd的写入吞吐量可能会受到限制。高并发写入请求会导致请求处理时间增加，影响整体性能。
- 数据一致性维护：在大规模集群中，维护数据一致性的复杂度会增加。etcd需要处理大量的数据同步和复制操作，这会消耗更多的资源和时间。
- 资源管理：大规模集群需要更多的计算资源和存储资源，如果资源分配不当，可能会导致性能瓶颈。

<!--
- k8s事件类型资源与其他资源不同，对一致性的要求较低，可以将两者拆分到不同的etcd集群中，分别进行优化
- 存放事件资源的etcd集群可以使用tmpfs（内存）来作为底层存储，实现性能最大化
- 存放其他资源的etcd集群使用SSD磁盘作为底层存储，通过开启ext4文件系统的fast commit功能来降低fdatasync对put请求时延的影响；使用自动化脚本在感知到etcd存储空间不足时自动执行compact和defrag
-->

### 操作步骤(可选)

为方便区分，下文中将event集群的3台节点命名为etcd-event-1、etcd-event-2、etcd-event-3，data集群的3台节点命名为etcd-data-1、etcd-data-2、etcd-data-3。

<details open>
<summary>

#### 选项一：手动安装

</summary>

1. 下载PKI管理工具step

    ```bash
    OS=linux   # linux, darwin, windows
    ARCH=amd64 # amd64, arm64
    STEP_VERSION=0.28.2
    curl -fL "https://github.com/smallstep/cli/releases/download/v$STEP_VERSION/step_${OS}_${STEP_VERSION}_${ARCH}.tar.gz" |
        tar Cxzf ~/.local - --strip-components=1 --wildcards '*/bin/step'
    ```

2. 生成jwt密钥对

    ```bash
    step crypto keypair jwt_ec384{.pub,} \
        --kty=EC --crv=P-384 \
        -f --insecure --no-password
    ```

3. 配置证书生成模板

    ```bash
    cat <<'EOF' > step-ca.json
    {
        "subject": {{toJson .Subject}},
        "issuer": {{toJson .Subject}},
        "keyUsage": ["keyEncipherment", "certSign"],
        "basicConstraints": {
            "isCA": true
        }
    }
    EOF
    cat <<'EOF' > step-leaf.json
    {
        "subject": {{toJson .Subject}},
        "sans": {{toJson .SANs}},
        "keyUsage": ["keyEncipherment"],
        "extKeyUsage": ["serverAuth", "clientAuth"]
    }
    EOF
    ```

4. 生成CA证书、客户端证书及节点间通信证书

    ```bash
    step certificate create etcd-ca ca{.crt,.key} \
        --kty=OKP --crv=Ed25519 \
        --not-after=87600h \
        --template=step-ca.json \
        -f --insecure --no-password
    for hostname in etcd-{event,data}-{1..3}; do
        ip=$(ssh -G "$hostname" | grep -i '^hostname ' | awk '{print $2}')
        step certificate create "$hostname" server{.crt,.key} \
            --kty=OKP ---crv=Ed25519 \
            --ca=ca.crt --ca-key=ca.key \
            --not-after=87600h \
            --san="$hostname" --san=localhost --san=127.0.0.1 --san=0:0:0:0:0:0:0:1 --san="$ip" \
            --template=step-leaf.json \
            -f --insecure --no-password
        step certificate create "$hostname" peer{.crt,.key} \
            --kty=OKP ---crv=Ed25519 \
            --ca=ca.crt --ca-key=ca.key \
            --not-after=87600h \
            --san="$hostname" --san=localhost --san=127.0.0.1 --san=0:0:0:0:0:0:0:1 --san="$ip" \
            --template=step-leaf.json \
            -f --insecure --no-password
    done
    ```

5. 下载etcd

    ```bash
    ETCD_VERSION=3.5.17
    curl -fL "https://github.com/etcd-io/etcd/releases/download/v$ETCD_VERSION/etcd-v$ETCD_VERSION-$OS-$ARCH.tar.gz" |
        tar xzf - --strip-components=1 --wildcards '*'/etcd{,ctl,utl}
    ```

6. 下载defrag脚本

    ```bash
    curl -fLo etcd-defrag.py 'https://gitee.com/openfuyao/ultra-large-scale-cluster-utils/raw/master/etcd/defrag.py'
    chmod +x etcd-defrag.py
    ```

7. 将etcd、jwt密钥对、证书、defrag脚本上传到各节点

    ```bash
    tar czf etcd.tgz etcd{,ctl,utl,-defrag.py} jwt_ec384{.pub,} ca.crt {server,peer}{.crt,.key}
    for hostname in etcd-{event,data}-{1..3}; do
        scp etcd.tgz "root@$hostname":/tmp/etcd.tgz
    done
    ```

8. 登录节点

    以etcd-data-1为例：

    ```bash
    ssh root@etcd-data-1
    ```

9. 安装依赖

    <details open>
    <summary>apt包管理器</summary>

    ```bash
    apt install -y --no-install-recommends systemd systemd-container libpam-systemd xfsprogs bash gettext-base tar python3
    ```

    </details>

    <details>
    <summary>dnf包管理器</summary>

    ```bash
    dnf install -y --setopt=install_weak_deps=False systemd systemd-container systemd-pam xfsprogs bash gettext tar python3
    ```

    </details>

10. 新建etcdadm用户

    <details open>
    <summary>Debian衍生发行版</summary>

    ```bash
    adduser --disabled-password --gecos '' etcdadm
    ```

    </details>

    <details>
    <summary>RedHat衍生发行版</summary>

    ```bash
    adduser --comment '' etcdadm
    ```

    </details>

11. 新建etcd目录，将etcd、jwt密钥对、证书、defrag脚本拷贝到指定位置

    ```bash
    XDG_BIN_HOME=/home/etcdadm/.local/bin
    XDG_DATA_HOME=/home/etcdadm/.local/share
    XDG_CONFIG_HOME=/home/etcdadm/.config
    XDG_STATE_HOME=/home/etcdadm/.state
    mkdir -p "$XDG_BIN_HOME" "$XDG_DATA_HOME/etcd" "$XDG_CONFIG_HOME/etcd" "$XDG_STATE_HOME/etcd/member"
    tar Cxzf "$XDG_BIN_HOME" /tmp/etcd.tgz etcd{,ctl,utl,-defrag.py}
    tar Cxzf "$XDG_DATA_HOME/etcd" /tmp/etcd.tgz jwt_ec384{.pub,} ca.crt {server,peer}{.crt,.key}
    ```

12. 优化etcd数据目录文件系统

    <details open>
    <summary>etcd-data集群节点</summary>
    初始化SSD磁盘，配置xfs文件系统

    ```bash
    SSD_DEVICE=/dev/nvme0n1 # SSD设备路径，/dev/nvmexxx或/dev/sdxxx
    UUID=$(cat /proc/sys/kernel/random/uuid)
    mkfs.xfs -f "$SSD_DEVICE" -m "uuid=$UUID"
    udevadm settle
    echo "UUID=$UUID $XDG_STATE_HOME/etcd/member xfs nosuid,nodev,noatime,nodiratime,uid=etcdadm,gid=etcdadm,mode=700 0 0" >> /etc/fstab
    systemctl daemon-reload
    mount -a
    ```

    关闭写缓存，提升性能

    ```bash
    SERIAL=$(udevadm info -n /dev/sda | grep ID_SERIAL= | awk -F= '{print $2}')
    mkdir -p /usr/loacl/sbin
    cat <<EOF > /usr/loacl/sbin/etcd-data-dev.sh
    #!/bin/bash -e
    SERIAL=$SERIAL
    DEVNAME=\$(find /dev/disk/by-id -regex ".*-\$SERIAL$")
    DEVPATH=/sys\$(udevadm info -n "\$DEVNAME" | grep DEVPATH= | awk -F= '{print $2}')
    echo 'write through' > "\$(find -L "\$DEVPATH" -name cache_type -print -quit 2> /dev/null)"
    EOF
    chmod +x /usr/loacl/sbin/etcd-data-dev.sh
    cat <<EOF > /etc/systemd/system/etcd-data-dev.service
    [Unit]
    Description=etcd data device configuration
    After=local-fs.target remote-fs.target
    Wants=local-fs.target remote-fs.target

    [Service]
    ExecStart=/usr/local/sbin/etcd-data-dev.sh
    Type=oneshot
    RemainAfterExit=yes

    [Install]
    WantedBy=default.target
    EOF
    systemctl daemon-reload
    systemctl enable --now etcd-data-dev.service
    ```

    </details>

    <details>
    <summary>etcd-event集群节点</summary>
    挂载tmpfs，使用内存存储etcd数据

    ```bash
    echo "tmpfs $XDG_STATE_HOME/etcd/member tmpfs nosuid,nodev,noatime,nodiratime,uid=etcdadm,gid=etcdadm,mode=700,size=16384M 0 0" >> /etc/fstab
    systemctl daemon-reload
    mount -a
    ```

13. 优化操作系统配置

    将CPU调节至性能模式

    ```bash
    echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    ```

14. 配置etcd

    ```bash
    IP=$(who -m --ips | awk '{print $5}')
    CLUSTER=etcd-0=https://<IP0>:2380,etcd-1=https://<IP1>:2380,etcd-2=https://<IP2>:2380
    cat <<EOF > "$XDG_CONFIG_HOME/etcd/config.yaml"
    name: etcd
    data-dir: $XDG_STATE_HOME/etcd
    listen-peer-urls: https://$IP:2380
    listen-client-urls: https://$IP:2379
    initial-advertise-peer-urls: https://$IP:2380
    advertise-client-urls: https://$IP:2379
    initial-cluster: $CLUSTER
    initial-cluster-state: new
    initial-cluster-token: random-token
    snapshot-count: 100
    heartbeat-interval: 100
    election-timeout: 1000
    initial-election-tick-advance: true
    max-snapshots: 1
    max-wals: 5
    quota-backend-bytes: 137438953472
    backend-batch-interval: 0
    backend-batch-limit: 0
    max-txn-ops: 17000
    max-request-bytes: 128000000
    max-concurrent-streams: 1024
    enable-grpc-gateway: true
    grpc-keepalive-min-time: 5000000000
    grpc-keepalive-interval: 7200000000000
    grpc-keepalive-timeout: 20000000000
    pre-vote: true
    auto-compaction-retention: '5m'
    auto-compaction-mode: periodic
    client-transport-security:
      cert-file: $XDG_DATA_HOME/etcd/server.crt
      key-file: $XDG_DATA_HOME/etcd/server.key
      client-cert-auth: true
      trusted-ca-file: $XDG_DATA_HOME/etcd/ca.crt
    peer-transport-security:
      cert-file: $XDG_DATA_HOME/etcd/peer.crt
      key-file: $XDG_DATA_HOME/etcd/peer.key
      client-cert-auth: true
      trusted-ca-file: $XDG_DATA_HOME/etcd/ca.crt
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
    tls-min-version: TLS1.2
    tls-max-version: TLS1.3
    auth-token: jwt,pub-key=$XDG_DATA_HOME/etcd/jwt_ec384.pub,priv-key=$XDG_DATA_HOME/etcd/jwt_ec384,sign-method=ES384,ttl=3600s
    bcrypt-cost: 10
    logger: zap
    log-outputs:
      - $XDG_STATE_HOME/etcd/etcd.log
    log-level: info
    enable-log-rotation: true
    log-rotation-config-json: '{"maxsize": 128, "maxage": 7, "maxbackups": 10, "localtime": false, "compress": false}'
    experimental-enable-lease-checkpoint: true
    experimental-enable-lease-checkpoint-persist: true
    unsafe-no-fsync: false
    EOF
    ```

15. 配置systemd服务

    ```bash
    loginctl enable-linger etcdadm
    mkdir -p "$XDG_CONFIG_HOME/systemd/user"
    cat <<EOF > "$XDG_CONFIG_HOME/systemd/user/etcd.service"
    [Unit]
    Description=etcd
    After=network-online.target local-fs.target remote-fs.target time-sync.target
    Wants=network-online.target local-fs.target remote-fs.target time-sync.target

    [Service]
    ExecStart=$XDG_BIN_HOME/etcd --config-file=$XDG_CONFIG_HOME/etcd/config.yaml
    ExecReload=/bin/kill -s HUP \$MAINPID
    TimeoutSec=0
    Restart=always
    RestartSec=5
    StartLimitBurst=3
    StartLimitInterval=60s
    LimitNOFILE=infinity
    LimitNPROC=infinity
    LimitCORE=infinity
    TasksMax=infinity
    Delegate=yes
    Type=simple
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
    # etcd-defrag.service仅etcd-data集群节点需要配置
    cat <<EOF > "$XDG_CONFIG_HOME/systemd/user/etcd-defrag.service"
    [Unit]
    Description=etcd auto defrag
    After=etcd.service
    Wants=etcd.service

    [Service]
    ExecStart=$XDG_BIN_HOME/etcd-defrag.py
    TimeoutSec=0
    Restart=always
    RestartSec=5
    StartLimitBurst=3
    StartLimitInterval=60s
    Delegate=yes
    Type=simple
    KillMode=mixed

    [Install]
    WantedBy=default.target
    EOF
    systemd daemon-reload
    systemd --machine=etcdadm@ --user daemon-reload
    systemd --machine=etcdadm@ --user enable --now etcd.service etcd-defrag.service
    ```

</details>

<details>
<summary>

#### 选项二：使用脚本安装

</summary>

1. xxx

</details>

### 调优模型

#### etcd关键性能参数分析

| 参数 | 描述 | 值 |
|-|-|-|
| snapshot-count | 触发一次snapshot的最大transaction数量；增加该值会导致内存/磁盘占用增加，减少该值会导致磁盘频繁IO，时延增加 | etcd-data：100，etcd-event：1 |
| heatbeat-interval | 心跳时间间隔；建议配置范围：$(0.5 * rtt_{max}, 1.5 * rtt_{max})$，其中$rtt_{max}$为所有节点间rtt的最大值 | 根据环境配置 |
| election-timeout | 选举超时时间；建议配置范围：$(5 * rtt_{max}, 15 * rtt_{max})$，其中$rtt_{max}$为所有节点间rtt的最大值 | 根据环境配置 |
| max-snapshots | 磁盘上保留的最大snapshot数量 | etcd-data：10，etcd-event：1 |
| max-wals | 磁盘上保留的最大wal文件数量，只有已经snapshot的wal文件才可以被删除 | etcd-data：10，etcd-event：1 |
| quota-backend-bytes | DB最大占用空间，超过该值后会导致写入失败 | etcd-data：68719476736（64GiB），etcd-event：8589934592（8GiB） |
| backend-batch-interval | transaction commit最大时间间隔；增加该值会降低时延，但是对磁盘性能要求较高 | etcd-data：10000000，etcd-event：10000 |
| backend-batch-limit | 触发一次commit的最大transaction数量；增加该值会降低时延，但是对磁盘性能要求较高 | etcd-data：100，etcd-event：1 |
| max-txn-ops | 单次transaction包含的最大op数量 | 16000 |
| max-request-bytes | 单次请求最大数据量 | etcd-data：128000000（128MB），etcd-event：16000000（16MB） |
| max-concurrent-streams | 单个client上允许的最大并发stream数量 | 1024 |
| auto-compaction-retention | 何时执行自动compact；设置为0则关闭自动compact | etcd-data：0，etcd-event：5m |
| auto-compaction-mode | 自动compact模式；设置为periodic则定时执行，设置为revision则每隔固定revision执行 | etcd-data：空，etcd-event：periodic |
| unsafe-no-fsync | 是否禁用fdatasync() | etcd-data：false，etcd-event：true |

### 后续步骤(可选)

### 示例/案例研究(可选)

> 展示这个最佳实践的应用和效果。

### 注意事项/常见问题(可选)

> 列出可能遇到的注意事项或常见问题，并提供相应的解决方案或建议。

### 结论

> 总结这个最佳实践的重要性和价值，并强调它的潜在影响或好处。

### 参考资料

> 列出参考或引用的任何相关资料、链接或资源。

## ref
