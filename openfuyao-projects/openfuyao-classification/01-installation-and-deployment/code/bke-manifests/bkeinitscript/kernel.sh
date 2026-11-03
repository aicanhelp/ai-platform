#!/bin/bash

cat > /etc/sysctl.d/99-bke-kernel.conf <<EOF
kernel.pid_max=1000000
net.core.rmem_default=56623104
net.core.wmem_default=56623104
net.core.optmem_max=40960
net.core.somaxconn=1024000
net.ipv4.tcp_max_syn_backlog=30000
net.ipv4.tcp_max_tw_buckets=2000000
net.ipv4.tcp_tw_reuse=1
net.ipv4.tcp_fin_timeout=30
net.ipv4.udp_rmem_min=8192
net.ipv4.udp_wmem_min=8192
net.ipv4.conf.all.accept_redirects=0
net.ipv4.conf.all.accept_source_route=0
net.ipv4.tcp_timestamps=0
net.ipv4.tcp_sack=1
net.ipv4.tcp_low_latency=1
net.ipv4.tcp_adv_win_scale=1
net.ipv4.tcp_adv_win_scale=1
net.netfilter.nf_conntrack_max=30000000
net.netfilter.nf_conntrack_tcp_timeout_time_wait=30
net.netfilter.nf_conntrack_tcp_timeout_fin_wait=30
net.netfilter.nf_conntrack_tcp_timeout_close_wait=30
net.ipv4.ip_local_port_range="1024 65535"
net.core.netdev_max_backlog=2000000
fs.file-max=26371830
fs.nr_open=26171830
kernel.msgmax=65535
net.core.rmem_max=61440000
net.core.wmem_max=61440000
net.ipv4.tcp_mem="94500000 915000000 927000000"
net.ipv4.tcp_rmem="32768 20480000 61440000"
net.ipv4.tcp_wmem="32768 20480000 61440000"
net.ipv4.tcp_keepalive_time=600
net.ipv4.tcp_keepalive_intvl=15
net.ipv4.neigh.default.gc_interval=3600
net.ipv4.neigh.default.gc_stale_time=3600
net.ipv6.neigh.default.gc_thresh3=160000
net.ipv4.neigh.default.gc_thresh3=160000
net.ipv4.neigh.default.gc_thresh2=4096
net.ipv4.neigh.default.gc_thresh1=2048
EOF


sysctl -p /etc/sysctl.d/99-bke-kernel.conf