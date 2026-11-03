# Preflight examples

本目录提供 `bke preflight init` 和 `bke preflight cluster-create` 的可复制示例。
示例只包含命令、输入文件和使用说明，不包含真实密码、Token 或 kubeconfig。

## 1. 选择前置检查策略

正式构建会把默认策略 `configs/preflight/preflight.yaml` 嵌入单个 `bke` 二进制，
运行时由二进制现场编译出 Bundle。

不指定配置文件时，命令使用二进制内置的默认策略。需要调整资源阈值、支持平台、端口、条件、
失败级别或修复建议时，复制一份完整的 `preflight.yaml`，修改后通过
`--preflight-config` 显式加载，无需重新构建 `bke`：

```bash
cp configs/preflight/preflight.yaml /etc/openFuyao/preflight.yaml
sudo ./bke preflight init \
  --preflight-config /etc/openFuyao/preflight.yaml \
  --hostIP 192.168.10.10
```

`cluster-create` 使用同一个参数：

```bash
sudo ./bke preflight cluster-create \
  --preflight-config /etc/openFuyao/preflight.yaml \
  --file ./examples/preflight/cluster.yaml \
  --nodes ./examples/preflight/nodes.yaml
```

配置文件是完整策略，不与内置 YAML 做字段级合并；必须保留默认策略已有的固定检查项配置键，不能
通过删除 `ports`、`resources`、`paths`、`tools`、平台、拓扑或管理集群配置来关闭发布检查。缺少
固定键、格式错误或包含未知字段时，前置检查不会启动，并在
报告中给出具体配置路径。已有发布规则仍由当前 `bke` 内置策略编译出的规则目录提供，外部策略不能
替换其规则 ID 或 Collector；`customChecks` 是唯一可以新增规则的入口。框架会把通过校验的自定义规则
编译进本次内存 Bundle，不写回文件，也不改变二进制内置策略。
端口等操作输入遵循“显式 CLI/业务文件 > policy fallback”：例如显式传入 `--kubernetesPort`
时检查该端口，否则使用外部 policy 的 `defaultPort`。policy 只改变检查计划，不修改后续安装命令。

`cluster.yaml` 或 `nodes.yaml` 的 schema/类型错误会在目标列显示路径，例如
`cluster.spec.clusterConfig.cluster.networking.podSubnet` 或 `nodes[0].spec.ip`；YAML 语法错误保留
文件和行号。Provider 本地业务校验失败也会把能确定的字段路径放入对应检查结果，避免只提示笼统的
“cluster/nodes 配置无效”。

### 1.1 自定义检查

自定义检查直接放在对应操作下：`operations.init.customChecks` 只对 init 生效，
`operations.clusterCreate.customChecks` 只对 cluster-create 生效。不要单独新建一份只含
`customChecks` 的文件；应复制完整默认策略，再把下面片段加入对应操作。

读取 `/etc/hosts` 并确认存在 IPv4 或 IPv6 回环地址：

```yaml
operations:
  init:
    customChecks:
      - key: hosts-loopback
        description: /etc/hosts 回环地址配置
        expected: 包含 127.0.0.1 或 ::1 回环地址映射
        collector:
          type: fileContent
          path: /etc/hosts
        expression: >-
          actual.matches('(?m)^\\s*(127\\.0\\.0\\.1|::1)(\\s|$)')
        failure: fail
        unknown: fail
        remediation: 补充回环地址映射后重试
```

执行节点上预装的只读检查脚本：

```yaml
operations:
  clusterCreate:
    customChecks:
      - key: runtime-health
        description: 容器运行时健康状态
        expected: 运行时为 containerd 且处于健康状态
        roles: [master, worker]
        collector:
          type: exec
          path: /opt/openfuyao/check-runtime
          args: [--format, json]
          resultType: object
          publicFields: [runtime, healthy]
        expression: actual.healthy && actual.runtime == 'containerd'
        failure: fail
        unknown: fail
        remediation: 修复并启动 containerd 后重试
```

脚本 stdout 必须只输出一个 Fact JSON，不能直接输出 PASS/FAIL：

```json
{
  "apiVersion": "preflight.openfuyao.io/fact/v1",
  "available": true,
  "type": "object",
  "value": {"runtime": "containerd", "healthy": true, "detail": "private"}
}
```

`resultType` 支持 `bool`、`integer`、`string`、`stringList`、`integerList` 和 `object`。
`object` 最多 32 个一层字段，值只能是标量或标量列表，并且必须配置 `publicFields`；只有这些字段
会进入文本/JSON 报告。`fileContent` 的原文只提供给 CEL，报告只显示“已读取（内容不展示）”。

用户不填写 digest。框架读取文件或脚本后计算观测 SHA-256，并作为 Evidence 写入详细报告。
自定义脚本必须是节点上已经存在的绝对路径普通可执行文件，不能被 group/world 写入；参数使用
结构化 `args`，不支持内联脚本、`bash -c`、上传、下载、隐式 sudo 或自定义环境变量。脚本运行时
只有固定 `PATH/LANG/LC_ALL`；脚本文件和 stdout 各最多 64 KiB，并受 `--fact-timeout`
和总 `--timeout` 控制。
cluster-create 的自定义检查依赖节点 SSH transport/auth/privilege，根依赖失败后不会继续执行脚本。

CEL 只暴露一个变量 `actual`，不注册文件、网络、时间或命令函数。表达式最长 2048 字节，并有
执行成本上限。配置加载和 Bundle 加载都会编译表达式；语法错误会阻止检查启动，运行期类型错误
会报告为框架内部错误，不会伪装成普通环境 FAIL。

## 2. Init

先构建或准备 `bke` 二进制，然后设置本机地址和安装制品地址：

```bash
export BKE_HOST_IP=192.168.10.10
export BKE_ONLINE_IMAGE=registry.example.com/openfuyao/bke-online:latest
./examples/preflight/preflight-init.sh
```

脚本等价于下面的命令：

```bash
sudo ./bke preflight init \
  --hostIP 192.168.10.10 \
  --onlineImage registry.example.com/openfuyao/bke-online:latest \
  --output text \
  --output-file /tmp/bke-preflight-init.txt
```

使用私有安装来源时，把实际地址作为参数追加到脚本后面。例如：

```bash
./examples/preflight/preflight-init.sh \
  --otherRepo https://registry.example.com:40443/openfuyao \
  --otherSource http://repo.example.com:40080 \
  --otherChart https://chart.example.com:443/chart
```

`installConsole` 和 `enableNTP` 默认开启。如果目标环境不需要它们，显式传入：

```bash
./examples/preflight/preflight-init.sh --installConsole=false --enableNTP=false
```

## 3. Cluster create

`cluster.yaml` 和 `nodes.yaml` 是输入文件样例。先将其中的 IP、域名、版本和节点认证信息
替换成真实值，再设置管理集群 kubeconfig：

```bash
export BKE_KUBECONFIG=/path/to/management-kubeconfig
chmod 600 ./examples/preflight/nodes.yaml
./examples/preflight/preflight-cluster-create.sh
```

脚本等价于下面的命令：

```bash
sudo ./bke \
  --kubeconfig /path/to/management-kubeconfig \
  preflight cluster-create \
  --file ./examples/preflight/cluster.yaml \
  --nodes ./examples/preflight/nodes.yaml \
  --max-concurrency 16 \
  --output text \
  --output-file /tmp/bke-preflight-cluster-create.txt
```

管理集群必须已经提供 Kubernetes API/Discovery、BKECluster/BKENode CRD 和
`kube-system/localkubeconfig`。preflight 只读这些信息，不会创建 Namespace、CR、Secret 或
其他 Kubernetes 资源；它也不会连接节点执行安装或清理命令。

## 4. 修改样例

- `cluster.yaml`：目标集群的 BKECluster 配置，包括控制面端点、网络、runtime 和来源。
- `nodes.yaml`：目标节点列表。示例使用 3 个 master/etcd 节点和 1 个 worker 节点。
- 节点 `password` 只是占位符，必须在本地替换；请勿把真实密码提交到仓库。
- 节点的 `metadata.labels["cluster.x-k8s.io/cluster-name"]` 必须与 BKECluster 名称一致。
- 如果不使用密码而使用 root 或免密 sudo，请按实际认证方式填写并让检查器验证。

这些文件是输入形状样例，未连接真实管理集群或节点时不能保证检查结果为 PASS。

## 5. 输出和退出码

命令行输出使用“状态、检查项、目标、期望、实际、原因及建议”六列表格，只展示需要处理的
失败/警告项。表格按当前终端宽度动态分配各列并填满窗口；操作员文本统一使用中文，节点名、
路径、命令、API、镜像地址等技术标识保持原值。目标列同时给出节点和具体端口、路径、API 或
命令，原因及建议保持为一句可执行说明。低于 72 列时自动降级为逐项纵向布局，输出不会超过
当前终端宽度。

`--output-file` 保存 UTF-8 详细文本报告，包含运行结论、运行问题以及 PASS/FAIL/WARN/SKIP
全部检查项，不包含终端 ANSI 颜色，便于复制、归档和打印。输入或计划问题导致规则没有启动时，
报告会明确显示“未启动”“已执行 0 项”和启动阻塞项，不会把它们伪装成环境检查失败。

需要机器读取时可将 `--output text` 改为 `--output json`。stdout 和 `--output-file` 都只包含一个结构化
`Report`（含 `issues` 和 `results`），不包含表格边框、颜色控制字符或 logger 前缀；进度只写入 stderr。

成功返回 0；WARN/SKIP 返回 2；FAIL 返回 3；输入或策略错误返回 4；内部错误返回 5；超时
返回 124；取消返回 130。
