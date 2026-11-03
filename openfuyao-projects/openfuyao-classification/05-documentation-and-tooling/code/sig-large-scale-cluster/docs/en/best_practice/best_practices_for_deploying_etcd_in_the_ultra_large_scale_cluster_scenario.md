# Best Practices

## Best Practices for Deploying etcd in the Ultra-Large-Scale Cluster Scenario

etcd is a core component in Kubernetes, with the following main functions:

- Configuration storage: etcd stores all configuration information for the Kubernetes cluster, including node information, Pod information, service configuration, network configuration, etc. It is the backend storage for the Kubernetes API server.
- Cluster state management: etcd maintains the state data of the entire cluster, ensuring that the state information of each node is consistent across the cluster. The API server retrieves and stores cluster state data from etcd to implement cluster management and scheduling.
- Service discovery: etcd stores service registration information, enabling services within the cluster to discover and communicate with each other. Kubernetes uses this information to manage the service lifecycle and load balancing.
- High availability and data consistency: etcd uses the Raft consensus algorithm to ensure data consistency and high availability. In a multi-node etcd cluster, data is automatically synchronized between different nodes, ensuring that data is not lost even if some nodes fail.

Therefore, etcd is Kubernetes' data storage and distributed coordination center, ensuring the reliability and consistency of the cluster.

### Objectives

This best practice focuses on improving etcd performance while ensuring stability to support the normal operation of clusters in ultra-large-scale scenarios. Main optimization measures include:

- Splitting etcd clusters according to resource types to reduce the load on individual etcd instances.
- Adjusting etcd configuration parameters.
- Using higher-performance physical devices, adjusting resources and system configurations.

### Prerequisites

The system will deploy three independent etcd clusters (3 nodes each), each carrying different business data:

- etcd-pods cluster: Dedicated to storing Pod resources.
- etcd-events cluster: Dedicated to storing Events and Leases resources.
- etcd-data cluster: Used to store all other resources except the above.

A server that has been configured with SSH keys (can log in without password to all 9 etcd nodes) needs to be designated as the deployment execution machine to uniformly run cluster initialization and configuration commands.

### Restrictions

- Supports Kubernetes v1.28.15, with etcd v3.5.18.
- Supports Kubernetes v1.34.1, with etcd v3.6.7.
- All 3 etcd clusters consist of 3 nodes; the event cluster stores data in memory; data cluster and pod cluster nodes need to mount high-speed SSD disks (8KB sequential IOPS ≥ 500, read/write speed ≥ 400MiB/s, preferably NVMe interface).

### Background

In ultra-large-scale cluster scenarios, the number and size of etcd requests increase exponentially, with the following performance bottlenecks:

- Network latency: etcd uses the Raft consensus algorithm, and its performance is greatly affected by network latency. In large-scale clusters, network latency increases, leading to increased request processing time.
- Disk IO latency: etcd needs to sync data to disk, and disk IO latency affects performance. Especially when using traditional hard drives (HDD), disk IO latency is more noticeable.
- Write throughput: As the cluster scale increases, etcd's write throughput may be limited. High concurrent write requests lead to increased request processing time, affecting overall performance.
- Data consistency maintenance: In large-scale clusters, the complexity of maintaining data consistency increases. etcd needs to handle a large number of data synchronization and replication operations, consuming more resources and time.
- Resource management: Large-scale clusters require more computing and storage resources. If resources are not allocated properly, performance bottlenecks may occur.

### Procedure

The procedure is for semi-automated etcd installation. A node needs to be selected from the etcd cluster as the execution startup script node. Automatic data compaction and defragmentation have been added to the startup script.

1. Log in to the execution startup script node as root.

2. Install etcd, all etcd nodes need to be installed.

    ```shell
    ARCH=$(uname -m)
    case $ARCH in
        x86_64) ARCH="amd64";;
        aarch64) ARCH="arm64";;
    esac
    VERSION="v3.5.18"

    # Download etcd installation package
    wget https://openfuyao.obs.cn-north-4.myhuaweicloud.com/etcd-io/etcd/releases/download/${VERSION}/etcd-${VERSION}-linux-${ARCH}.tar.gz

    # Install etcd
    tar -xvf etcd-"${VERSION}"-linux-${ARCH}.tar.gz
    cp -rf etcd-"${VERSION}"-linux-${ARCH}/etcd* /usr/local/bin/
    chmod +x /usr/local/bin/{etcd,etcdctl,etcdutl}
    ```

3. Install yq tool, only install on the execution startup script node.

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

4. Install step tool, only install on the execution startup script node.

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

5. Configure passwordless login, configure passwordless login from the execution startup script node to other nodes.

    ```shell
    # Generate public key, just press Enter when prompted
    ssh-keygen

    # Upload login public key to other etcd nodes
    for ip in <node ip addresses, eg:192.168.200.238 192.168.200.237 192.168.200.236 192.168.200.235 192.168.200.234 192.168.200.233 192.168.200.232 192.168.200.231 192.168.200.230>; do
        # You will be prompted to enter password, enter password and press Enter
        ssh-copy-id -i ~/.ssh/id_rsa.pub root@${ip}
    done
    ```

6. Install basic components, install on all etcd nodes.

    ```shell
    yum install -y systemd-pam
    ```

7. Save the following startup script to etcd-bootstrap.sh on the execution startup script node. [Click to get startup script etcd-bootstrap.sh](#startup-script-etcd-bootstrapsh-configuration-details).

8. Execute the following command on the execution startup script node to run etcd-bootstrap.sh to start the etcd data cluster.

    ```shell
    mkdir /root/etcd-install

    # Need to replace with actual ip addresses
    bash etcd-bootstrap.sh <etcd data node ip, eg: 192.168.200.238 192.168.200.237 192.168.200.236> -c /root/etcd-install -p etcdData
    ```

9. Execute the following command on the execution startup script node to run etcd-bootstrap.sh to start the etcd pod cluster.

    ```shell
    mkdir /root/etcd-install

    # Need to replace with actual ip addresses
    bash etcd-bootstrap.sh <etcd data node ip, eg: 192.168.200.235 192.168.200.234 192.168.200.233> -c /root/etcd-install -p etcdPods
    ```

10. Execute the following command on the execution startup script node to run etcd-bootstrap.sh to start the etcd events-leases cluster.

    ```shell
    mkdir /root/etcd-install

    # Need to replace with actual ip addresses
    bash etcd-bootstrap.sh <etcd data node ip, eg: 192.168.200.232 192.168.200.231 192.168.200.230> -c /root/etcd-install -p etcdPods --use-tmpfs
    ```

11. Confirm etcd cluster status.

- Execute `systemctl status etcd` on all etcd nodes. If the word `running` appears, it means etcd has started.
- Execute the following command on all etcd nodes to check if the etcd cluster is healthy. If the `HEALTH` column in the output table is all true, it indicates the cluster is in a healthy state.
    ```shell
    ETCDCTL_API=3 etcdctl --endpoints=https://127.0.0.1:2379 \
    --cacert=/usr/local/share/etcd/ca.crt \     # Replace with actual certificate path
    --cert=/usr/local/share/etcd/server.crt \   # Replace with actual certificate path
    --key=/usr/local/share/etcd/server.key \    # Replace with actual certificate path
    endpoint health --write-out=table
    ```

### Conclusion

This deployment form and parameter tuning have been tested through simulation and can support the stable operation of a K8s cluster with 16,000 nodes. The following table provides recommended parameter optimization configurations based on simulation test results.

**Table 1** etcd Key Performance Parameter Analysis

| Parameter | Description                                                                             | Value |
|-|-----------------------------------------------------------------------------------------|-------|
| snapshot-count | Maximum number of transactions to trigger a snapshot; increasing this value will increase memory/disk usage, decreasing it will cause frequent disk IO and increased latency.              | etcd-data: 100, etcd-event: 1 |
| heatbeat-interval | Heartbeat time interval; recommended configuration range: $(0.5 * rtt_{max}, 1.5 * rtt_{max})$, where $rtt_{max}$ is the maximum rtt between all nodes. | Configure according to environment |
| election-timeout | Election timeout; recommended configuration range: $(5 * rtt_{max}, 15 * rtt_{max})$, where $rtt_{max}$ is the maximum rtt between all nodes.    | Configure according to environment |
| max-snapshots | Maximum number of snapshots to keep on disk.                                                            | etcd-data: 10, etcd-event: 1 |
| max-wals | Maximum number of wal files to keep on disk; only wal files that have been snapshotted can be deleted.                                      | etcd-data: 10, etcd-event: 1 |
| quota-backend-bytes | Maximum DB space usage, writes will fail after exceeding this value.                                                         | etcd-data: 68719476736 (64GiB), etcd-event: 8589934592 (8GiB) |
| backend-batch-interval | Maximum time interval for transaction commit; increasing this value will reduce latency but requires higher disk performance.                                | etcd-data: 10000000, etcd-event: 10000 |
| backend-batch-limit | Maximum number of transactions to trigger a commit; increasing this value will reduce latency but requires higher disk performance.                              | etcd-data: 100, etcd-event: 1 |
| max-txn-ops | Maximum number of ops in a single transaction.                                                        | 16000 |
| max-request-bytes | Maximum data size per request.                                                                     | etcd-data: 128000000 (128MB), etcd-event: 16000000 (16MB) |
| max-concurrent-streams | Maximum number of concurrent streams allowed on a single client.                                                      | 1024 |
| auto-compaction-retention | When to perform automatic compaction; set to 0 to disable automatic compaction.                                                | etcd-data: 0, etcd-event: 5m |
| auto-compaction-mode | Auto compaction mode; set to periodic for periodic execution, set to revision for execution every fixed revision.                       | etcd-data: empty, etcd-event: periodic |
| unsafe-no-fsync | Whether to disable fdatasync().                                                               | etcd-data: false, etcd-event: true |


### References

  [etcd Official Installation Documentation](https://etcd.io/docs/v3.6/install/)

### Appendix
#### Startup Script etcd-bootstrap.sh Configuration Details
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
                # Mount disk
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
              # Set disk to write-through mode to avoid data loss
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
# Set CPU priority
CPUSchedulingPolicy=rr
CPUSchedulingPriority=99
# Set IO priority
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
