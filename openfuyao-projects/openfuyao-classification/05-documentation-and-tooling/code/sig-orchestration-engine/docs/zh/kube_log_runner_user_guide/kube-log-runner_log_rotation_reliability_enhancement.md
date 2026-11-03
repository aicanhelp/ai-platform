# kube-log-runner日志轮转可靠性增强

## 特性介绍

Kubernetes社区原生的`kube-log-runner`功能单一，只提供日志重定向，本特性旨在对原生机制进行功能扩展，提供字节级精细日志轮转、磁盘满场景业务不中断功能。

### 应用场景

- `kube-log-runner`开启定制参数`--fy-ignore-nospace-error=true`。

  磁盘空间满日志写入失败：`kube-log-runner`忽略磁盘满而无法写入日志文件的error，丢弃日志服务不中断，并且记录发生故障的时间。

  磁盘空间释放：`kube-log-runner`写入日志文件成功，触发故障恢复，在日志中记录故障发生时间，恢复时间。

- `kube-log-runner`开启定制参数`--fy-rotate-enable=true`，并设置一系列日志轮转参数。

  日志文件在达到单个文件大小阈值时开启轮转，移除超限日志文件，进行转储压缩。

- `kube-log-runner`开启定制参数`--fy-rotate-enable=true`和`--fy-check-log-deleted-period=5`，每5秒钟检查日志文件是否存在。

  当日志文件被删除，检测到日志文件不存在，新创建出日志文件，在新日志中记录故障发生时间。

### 能力范围

- 磁盘空间满时丢弃日志服务不中断，并且记录发生故障的时间；故障恢复时，在日志中记录故障恢复时间。

- 日志文件达到阈值时移除超限日志文件，进行转储压缩。

- 当日志文件被删除时新创建出日志文件，并在新日志中记录故障发生时间。

### 亮点特征

- 故障场景不中断，故障恢复后记录发生时间。

- 单个日志过大时压缩存储，减少存储压力。

- 日志被误删时自动恢复。

### 实现原理

把日志文件包装为一个可扩展Writer，实现带后续处理的Write方法，轮转由单独协程实现不阻塞日志持续写入。

相关代码段`staging/src/k8s.io/component-base/fylogs/file_wrap.go:33`

```
func (f *fileWrap) Write(p []byte) (int, error) {
    f.writtenBytes, f.writtenErr = f.target.Write(p)
    if f.writtenErr == nil {
        for _, f := range f.normalFunc {
            f()
        }
    } else {
        for _, f := range f.onErrFunc {
            f()
        }
        f.writtenBytes = len(p)
    }
    return f.writtenBytes, f.writtenErr
}
```

应对日志写入速率超高场景，确保同一时刻仅有一个轮转协程。

```
func (r *rotator) startRotateLog() {
    r.rotateLock.Lock()
    r.rotateCounting++ // 轮转请求计数
    r.rotateLock.Unlock()

    // Ensure only one rotation at a time
    if r.rotateCounting != 1 { // 已经启动轮转
        return
    }

    var err error
    defer r.afterRotateLog(err)
    err = r.rotateLogs()
    if err != nil {
        r.tryPrintErr(err)
    }
}
```

### 与相关特性的关系

无特殊依赖。

## 安装

无需安装，kube-log-runner相关功能通过在启动kube-log-runner时配置参数来开启。

## 使用kube-log-runner

### 使用限制

`kube-log-runner`只对子进程产生的stdout与stderr进行重定向和日志轮转，外部注入日志到日志文件不会触发轮转。

### 操作步骤

kube-log-runner日志轮转相关功能通过配置启动参数来开启或调节配置：

**表1** kube-log-runner相关特性参数

| 参数                    | 缺省值   | 描述                         | 配置示例                       |
| ----------------------- |-------| ---------------------------- | ------------------------------ |
| fy-file-permission      | 644   | 日志文件权限                 | --fy-file-permission=750       |
| fy-compress-permission  | 440   | 压缩文件权限                 | --fy-compress-permission=500   |
| fy-ignore-nospace-error | false | 启用磁盘满丢日志，服务不中断 | --fy-ignore-nospace-error=true |
| fy-rotate-enable        | false | 启用日志轮转                 | --fy-rotate-enable=true        |

当启用日志轮转（`--fy-rotate-enable=true`）时日志轮转相关参数：

**表2** kube-log-runner日志轮转相关性能参数

| 轮转参数                    | 缺省值          | 描述                          | 配置示例                        |
| --------------------------- |--------------|-----------------------------| ------------------------------- |
| fy-each-file-size           | 20（MB）       | 单个日志文件大小上限，范围[5MB，100MB]    | --fy-each-file-size=100         |
| fy-total-size-limit         | 150（MB）      | 日志文件+转储文件总大小上限，配置为0表示无上限    | --fy-total-size-limit=100       |
| fy-total-count-limit        | 10           | 日志文件+转储文件总个数上限，配置为0表示无上限    | --fy-total-count-limit=0        |
| fy-compress-enable          | true         | 启用转储文件压缩                    | --fy-compress-enable=true       |
| fy-check-log-deleted-period | 0（not check） | 周期检测日志文件是否删除，周期时长，配置为0表示不检查 | --fy-check-log-deleted-period=5 |

### 相关操作

无。
