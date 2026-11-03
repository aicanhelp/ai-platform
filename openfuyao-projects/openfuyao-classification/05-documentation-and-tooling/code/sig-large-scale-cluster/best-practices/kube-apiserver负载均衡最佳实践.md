# 最佳实践

## 大规模集群场景下 kube-apiserver 负载均衡最佳实践

[Keepalived](https://www.keepalived.org/) 提供 VRRP 实现，并允许配置 Linux 机器使服务负载均衡，预防单点故障。

[HAProxy](http://www.haproxy.org/) 提供可靠、高性能的负载均衡，并与 Keepalived 完美配合。

### 目标

本最佳实践聚集在指导如何通过使用 keepalive 和 haproxy 的组合实现 kube-apiserver 负载均衡功能。

### 前提条件(可选)

- 集群使用 kube-apiserver 作为 kubernetes 集群的 API 服务器，并部署在多个节点上

- 存在额外两台以上lb节点负责配置负载均衡，节点上安装 keepalived 及 haproxy 服务，且虚拟IP可在集群中连通

### 使用限制(可选)

### 背景信息(可选)

高可用 Kubernetes 集群需要确保在运行时不会出现服务中断，并将请求合理的分配在多个 kube-apiserver 实例之间，避免各别实例接收过多请求而导致资源占用过高，无法正常提供服务。

### 操作步骤(可选)

在 lb 节点上安装 keepalived 和 haproxy。当这几个 lb 节点某一个出现故障时，虚拟 IP 能够自动转移到其它 lb 节点，使集群仍然能正常运行，实现高可用。

分别登录几个 lb 节点，配置 haproxy：

##### haproxy 配置

配置 `/etc/haproxy/haproxy.cfg` 多个节点配置**相同**：

```shell
global
 maxconn 180000                                    # 最大连接数，建议配置为150000，保证虚拟节点可正常创建。
 uid 0                                            # 所属用户，根据实际情况配置。
 gid 0                                            # 所属用户组，根据实际情况配置。
 
defaults
 log global                                       # 使用global的日志。
 retries 3                                        # 尝试连接的最大次数，超过此值便认为后端服务器不可用。建议配置为3，配合连接超时时间，确保10s内发现后端服务器是否异常。
 timeout connect 3s                               # 设置等待将客户端请求转发至后端服务器的超时时间。建议配置为3s，配合连接超时时间，确保10s内发现后端服务器是否异常。
 timeout client 60s                               # 设置客户端的最长不活动的超时时间，设置为60s，以减少连接重复建立。
 timeout server 60s                               # 设置服务端的最长不活动的超时时间，设置为60s，以减少连接重复建立。
 
frontend k8s-master                               # 前台
 bind xxxx.xxxx.xxxx.xxxx:xxxx                    # 绑定前台vip ip为kubeconfig中将使用的虚拟ip地址，端口为将在kubeconfig中使用的端口，根据实际情况配置。
 mode tcp                                         # 使用tcp协议与kube-apiserver交互。
 default_backend k8s-master                       # 默认调用的后端主机，默认使用k8s-master
 use_backend k8s-master                           # 调用指定的后端主机 k8s-master
 
backend k8s-master                                # 后台
 option tcplog
 balance roundrobin                               # 负载均衡算法：建议配置为roundrobin，将请求均匀分布在多个kube-apiserver实例。
 default-server inter 3s rise 2 fall 3            # 指定后端服务器的默认设置,也可单独为下面的server配置。
                                                  # inter 3s: 监控状态检查的时间间隔，单位为毫秒，默认为2000，建议设置为3秒，每隔3秒检查kube-apiserver endpoint是否正常，确保10s内发现后端服务器是否异常。
                                                  # rise 2: 如果连续两次检查成功，则认为此kube-apiserver endpoint可用。
                                                  # fall 3: 如果连续三次检查成功，则认为此kube-apiserver endpoint不可用。
 server  master1  xxx.xxx.xxx.xxx:xxxx check      # kube-apiserver实例1的ip:port信息，根据实际情况配置。
 server  master2  xxx.xxx.xxx.xxx:xxxx check      # kube-apiserver实例2的ip:port信息，根据实际情况配置。
 server  master3  xxx.xxx.xxx.xxx:xxxx check      # kube-apiserver实例3的ip:port信息，根据实际情况配置。
```

在 /etc/rsyslog.d 目录下增加 haproxy.conf 文件，并写入内容：

```
# 此行配置harpoxy日志输出文件，可自由定制
local1.* /etc/haproxy/haproxy.log
#使用socket方式输出日志
$ModLoad imuxsock
$AddUnixListenSocket /etc/haproxy/dev/log
```

保存文件后执行以下命令重启 haproxy 后配置生效：

```shell
systemctl restart haproxy
```

将 haproxy 设置为开机后自动运行：

```shell
systemctl enable haproxy
```

接着分别登录几个 lb 节点，配置 keepalived：

##### keepalived 配置

配置 `/etc/keepalived/keepalived.conf` ，多个节点配置**不同**：

```shell
global_defs {
    router_id 1                           # 用于标识keepalived服务器，默认值是主机名。此参数无实质影响。
    vrrp_garp_master_refresh 10           # ARP老化参数，每隔指定时间，刷新路由ARP，防止路由ARP未及时刷新导致vip无法连通。建议配置为10。
}

vrrp_script chk_haproxy {                 # 配置检测脚本，每隔3s检测haproxy状态，连续失败三次则认为此节点上haproxy不可用，连续成功2次则认为可用。
  script "killall -0 haproxy"
  interval 3
  weight 100
  fall 3
  rise 2
}

vrrp_script chk_apiserver {                 # 配置检测脚本，每隔3s检测集群是否可访问状态，连续失败三次则认为此节点上haproxy不可用，连续成功2次则认为可用。
  script "/etc/keepalived/check_apiserver.sh"
  interval 3
  weight -100
  fall 3
  rise 2
}

vrrp_instance VI1 {                       # VI1是实例名，多个keepalived配置一样即可，取值可定制。
    state BACKUP                          # 两个LB节点都配置为BACKUP保证两个LB节点平等。
    priority 100                          # 此 keepalived 实例优先级，多个实例配置成一样的数值即可。
    interface eth0                        # 需要路由的ip所在网卡名，即kube-apiserver绑定ip所在的网卡名，选举将使用配置的接口进行。
    virtual_router_id 24                  # 虚拟路由ID，取值范围0-255，同一集群的多个keepalived必须使用同一个ID，取值满足范围要求即可。
    nopreempt                             # 工作模式，nopreempt表示工作在非抢占模式，默认是抢占模式 preempt，使用非抢占模式，减少vip频繁切换。
    advert_int 1                          # 心跳通告频率，多个keepalived必须配置相同的心跳通告频率。建议配置为1保证高可用性。
    virtual_ipaddress {                   # 虚拟VIP地址，配置为kubeconfig将使用的ip，kubeconfig将使用此ip进行负载均衡。
        xxx.xxx.xxx.xxx/xx dev eth0       # 虚拟VIP地址的IP/掩码及配置在哪个网卡，根据实际情况配置。
    }
    notify_master ""                      # 如果主机状态变成Master|Backup|Fault之后会去执行的通知脚本
    notify_backup ""
    notify_fault ""
    unicast_src_ip xxx.xxx.xxx.xxx        # 此lb主机ip地址，根据实际情况配置。
    unicast_peer {
        xxx.xxx.xxx.xxx                   # 其它lb主机ip地址，根据实际情况配置。          
    }
    
    track_script {
        chk_haproxy
        chk_apiserver
    }
}
```

配置集群状态检测脚本`/etc/keepalived/check_apiserver.sh`：

```shell
#!/bin/sh
VIP_ADDRESS=            # 此处填写 haproxy 绑定的前端虚拟 IP
VIP_BIND_PORT=          # 此处填写 haproxy 绑定的前端端口
if ip addr | grep -q ${VIP_ADDRESS}; then
    curl --silent --max-time 2 --insecure https://${VIP_ADDRESS}:${VIP_BIND_PORT}/healthz 
    exit $?
fi
exit 0
```

为脚本设置权限：

```shell
chmod +x /etc/keepalived/check_apiserver.sh
```

保存文件后执行以下命令重启 keepalived 后配置生效：

```shell
systemctl restart keepalived
```

将 haproxy 设置为开机后自动运行：

```shell
systemctl enable keepalived
```

##### 验证高可用

在所有 lb 节点执行命令：

```shell
ip a
```

可观察到只有一个 lb 节点在配置虚拟 IP 的网卡上存在虚拟 IP。

重启当前虚拟 IP 所在的节点上的 haproxy：

```shell
systemctl stop haproxy
```

等待10s后，再次在所有 lb 节点执行命令：

```shell
ip a
```

此时可观察到只有一个 lb 节点在配置虚拟 IP 的网卡上存在虚拟 IP，且不同于之前虚拟 IP 所在节点。证明虚拟 IP 可在多个 lb 节点间切换，实现高可用。

恢复停止的 haproxy：

```shell
systemctl restart haproxy
```

##### 配置kubeconfig

kubernetes中，各组件可以使用 kubeconfig 文件访问 kubeconfig 文件中指定的 kube-apiserver endpoint。可以通过设置虚拟IP，将请求路由到多个 kube-apiserver 上，实现负载均衡。

> 需要注意使用虚拟 IP 配置 kubeconfig，需要在签发证书时将提前规划好的虚拟 IP 也添加进去。

##### 重启所有 kubernetes 组件，使用新 kubeconfig 进行服务

### 参考资料

[Keepalived](https://www.keepalived.org/) 

[HAProxy](http://www.haproxy.org/) 
