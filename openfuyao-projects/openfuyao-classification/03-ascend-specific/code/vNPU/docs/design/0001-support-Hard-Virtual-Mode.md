## 特性说明

硬切分是昇腾官方支持的NPU虚拟化能力。vNPU项目在软切分基础上同时支持动态硬切分，通过一套调度器和Device Plugin支持两种虚拟化模式，为用户提供更多的选择。

## 设计思路

vNPU硬切分的基本思路是volcano调度器根据业务POD声明的AICore资源大小映射合适的切分模板，Device Plugin根据调度器分配的切分模板设置对应的环境变量，再由Ascend Docker Runtime根据环境变量负责切分vnpu和设备挂载。

## 使用方法

### 设置节点切分模式

节点默认为软切分模式。通过Helm部署npu-device-plugin服务时可通过values.yaml或Helm命令行参数设置节点切分模式。

例如，在values.yaml中设置节点切分模式，然后通过Helm部署。

```yaml
vnpuNodeMode:
  worker1：  # 节点名
    mode:"hard"  # 节点切分模式，hard|soft，默认为soft
  worker2:
    mode:"soft"
```

或者通过Helm命令行参数设置。例如：

```yaml
helm upgrade --install vnpu ./vnpu-1.0.0.tgz -n xpu --set vnpuNodeMode.worker1.mode="hard"
```

### POD声明AICore资源

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: testpod
  annotations:
    huawei.com/vnpu-mode: "hard" # 虚拟化切分模式，soft|hard，可选，如果不设置，则由调度器自动选择
    huawei.com/vnpu-hard-aicpu-level: "low" # AI CPU分配策略，low表示低配，high表示高配，可选，默认为low
spec:
    schedulerName: volcano
    containers:
        resources:
          limits:
            huawei.com/vnpu-number: 1  
            huawei.com/vnpu-cores: 50  # AI Core百分比
          requests:
            huawei.com/vnpu-number: 1
            huawei.com/vnpu-cores: 50
```

调度器根据POD声明的huawei.com/vnpu-cores值和设备总的AICore数量，计算实际的AICore数量，然后匹配最接近的切分模板。例如，对于310P3设备，假设POD声明huawei.com/vnpu-cores=50，310P3设备总的AICore数量为8，则分配4个AICore，映射模板为vir04_3c。如果huawei.com/vnpu-hard-aicpu-level=high，则映射到模板vir04。

调度器自动选择策略：如果AICore能最优匹配到某个切分模板（AICore的误差小于5%），则优先选择硬切分，否则优先选择软切分，即尽量减少硬切分的资源浪费。例如，对于310P3设备，假设POD声明huawei.com/vnpu-cores=50，则自动选择硬切分1切2；假设POD声明huawei.com/vnpu-cores=40，则自动选择软切分。自动选择的模式只是建议值，不是强制值，当集群剩余设备的切分模式不满足建议值时，仍然使用设备实际的切分模式。

## 其它

1. 当前只支持910b4和310p3设备，其它型号暂不支持。
2. 暂只支持根据AI Core和AI CPU数量映射切分模板，不支持根据dvpp配置映射切分模板。