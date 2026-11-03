# kube-log-runner Log Rotation Reliability Enhancement

## Feature Introduction

The native `kube-log-runner` in the Kubernetes community has limited functionality, providing only log redirection. This feature aims to extend the native mechanism by providing byte-level precise log rotation and disk-full scenario service continuity.

### Use Cases

- `kube-log-runner` enables the custom parameter `--fy-ignore-nospace-error=true`.

  Log write failure due to full disk: `kube-log-runner` ignores errors caused by log file write failures due to disk full, discards logs without interrupting service, and records the time when the failure occurred.

  Disk space freed: `kube-log-runner` successfully writes to the log file, triggers failure recovery, and records both the failure time and recovery time in the log.

- `kube-log-runner` enables the custom parameter `--fy-rotate-enable=true` and sets a series of log rotation parameters.

  When a log file reaches the single-file size threshold, rotation is triggered, over-limit log files are removed, and the log files are archived and compressed.

- `kube-log-runner` enables the custom parameters `--fy-rotate-enable=true` and `--fy-check-log-deleted-period=5`, checking every 5 seconds whether the log file exists.

  When a log file is deleted and its absence is detected, a new log file is created, and the failure time is recorded in the new log.

### Supported Scope

- When disk space is full, logs are discarded without interrupting service, and the failure time is recorded; when the failure is recovered, the recovery time is recorded in the log.

- When the log threshold is reached, over-limit log files are removed and archived with compression.

- When a log file is deleted, a new log file is created, and the failure time is recorded in the new log.

### Highlights

- No service interruption in failure scenarios; records failure time after recovery.

- Compressed storage when individual logs are too large, reducing storage pressure.

- Automatic recovery when logs are accidentally deleted.

### Implementation Principle

The log file is wrapped as an extensible Writer, implementing a Write method with subsequent processing. Rotation is implemented by a separate goroutine to avoid blocking continuous log writing.

Related code segment `staging/src/k8s.io/component-base/fylogs/file_wrap.go:33`

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

To handle high log write rate scenarios, only one rotation goroutine runs at a time.

```
func (r *rotator) startRotateLog() {
    r.rotateLock.Lock()
    r.rotateCounting++ // rotation request count
    r.rotateLock.Unlock()

    // Ensure only one rotation at a time
    if r.rotateCounting != 1 { // rotation already started
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

### Relationship with Related Features

No special dependencies.

## Installation

No installation required. The kube-log-runner related functionality is enabled by configuring parameters when starting kube-log-runner.

## Using kube-log-runner

### Usage Restrictions

`kube-log-runner` only redirects and rotates logs from stdout and stderr of child processes; external injection of logs into log files will not trigger rotation.

### Procedure

kube-log-runner log rotation related features are enabled or adjusted by configuring startup parameters:

**Table 1** kube-log-runner related feature parameters

| Parameter               | Default Value | Description                              | Configuration Example                  |
| ----------------------- | ------------- | ---------------------------------------- | -------------------------------------- |
| fy-file-permission      | 644           | Log file permissions                     | --fy-file-permission=750               |
| fy-compress-permission  | 440           | Compressed file permissions              | --fy-compress-permission=500           |
| fy-ignore-nospace-error | false         | Enable log discard on disk full, no service interruption | --fy-ignore-nospace-error=true |
| fy-rotate-enable        | false         | Enable log rotation                      | --fy-rotate-enable=true                |

When log rotation is enabled (`--fy-rotate-enable=true`), the log rotation related parameters are:

**Table 2** kube-log-runner log rotation related performance parameters

| Rotation Parameter          | Default Value   | Description                                                    | Configuration Example                   |
| --------------------------- | --------------- | -------------------------------------------------------------- | --------------------------------------- |
| fy-each-file-size           | 20 (MB)         | Maximum size of a single log file, range [5MB, 100MB]          | --fy-each-file-size=100                 |
| fy-total-size-limit         | 150 (MB)        | Maximum total size of log files + archived files; 0 means no limit | --fy-total-size-limit=100           |
| fy-total-count-limit        | 10              | Maximum total count of log files + archived files; 0 means no limit | --fy-total-count-limit=0           |
| fy-compress-enable          | true            | Enable compressed archiving                                    | --fy-compress-enable=true               |
| fy-check-log-deleted-period | 0 (not check)   | Periodic check interval for whether the log file is deleted; 0 means no check | --fy-check-log-deleted-period=5 |

### Related Operations

None.
