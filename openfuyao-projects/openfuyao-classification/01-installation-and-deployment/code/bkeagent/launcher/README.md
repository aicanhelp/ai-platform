# BKEAgent-Launcher

### 背景

-   当前Konk项目的BKEAgent推送方式基于SSH，方法单一且局限性强，必须由用户提供每台机器的用户名密码。
-   在对已有集群进行纳管的时，用户对于成百上千的机器无法短时间提供所有机器的用户名密码，需要另辟蹊径。
-   在此情况下需要设计一种符合云原生的BKEAgent推送方式

### 调研

-   k8s中的Daemonset资源在创建后会在集群中每个节点上启动一个Pod
-   linux中的nsenter命令是一个可以在**指定进程的命令空间下运行指定程序的命令**，通过该命令可以从容器中执行宿主机的命令

### 目标

-   通过k8s的Daemonset将BKEAgent-Launcher推送到已有集群的每个节点之上
-   BKEAgent-Launcher能够在尽可能不破坏主机环境的情况下将BKEAgent启动
-   能对BKEAgent启动参数进行设置

### 限制

-   集群提前存在，且所有节点可调度
-   nsenter这种在容器中执行宿主机的命令并不安全

### 使用

-   BKEAgent-Launcher启动参数

| 参数         | 必填 | 默认值  | 类型     | 备注                           |
| ---------- | -- | ---- | ------ | ---------------------------- |
| kubeconfig | 是  |      | string | bkeagent启动使用的kubecofnig文件路径  |
| ntp-server | 否  |      | string | ntp服务器地址。用于bkeagent定时同步主机时间  |
| debug      | 否  | true | string | 开启bkeagent的debug模式，会有更多的日志输出 |

-   部署BKEAgent-Launcher
    ```yaml
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: bkeagent-launcher-kubeconfig
      namespace: kube-system
    data:
      config: |-
        ...
        ...

    apiVersion: apps/v1
    kind: DaemonSet
    metadata:
      name: bkeagent-launcher
      namespace: kube-system
    spec:
      updateStrategy:
        type: RollingUpdate
      selector:
        matchLabels:
          app: bkeagent
      template:
        metadata:
          labels:
            app: bkeagent
        spec:
          restartPolicy: Always
          hostNetwork: true
          hostIPC: true
          hostPID: true
          containers:
            - command:
                - launcher
                - --kubeconfig=/etc/bkeagent/config
                - --ntp-server={{ .ntpServer }}
                - --debug={{ .debug }}
              env:
                - name: container
                  value: "true"
              image: {{ .repo }}bkeagent-launcher:latest
              imagePullPolicy: Always
              name: bkeagent
              securityContext:
                privileged: true
                runAsUser: 0
              volumeMounts:
                - mountPath: /etc/bkeagent/launcher
                  name: bkeagent-launcher
              readinessProbe:
                httpGet:
                  port: 3377
                  path: /readyz
                initialDelaySeconds: 5
                periodSeconds: 5
                timeoutSeconds: 2
                successThreshold: 2
          volumes:
            - name: bkeagent-launcher
              hostPath:
                path: /etc/bkeagent/launcher
                type: DirectoryOrCreate
            - name: bkeagent-launcher-kubeconfig
              configMap:
                name: bkeagent-launcher-kubeconfig
    ```
-   构建BKEAgent-Launcher多架构镜像
    ```纯文本
    $ pwd
    path/to/bkeagent
    $ make release-launcher
    ```