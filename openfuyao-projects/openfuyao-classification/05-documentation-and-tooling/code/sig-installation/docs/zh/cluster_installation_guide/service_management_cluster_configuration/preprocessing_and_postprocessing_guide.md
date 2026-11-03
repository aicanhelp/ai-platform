# 前置/后置处理脚本自定义配置

本文档旨在介绍集群安装时前置和后置处理，其中前者在集群节点的环境初始化时执行，后者在下发所有addon安装任务后执行。

## 背景信息

使用openFuyao部署的集群，需要在集群的节点上执行前置操作和后置操作，且能实现用户自定义操作内容。

## 前提条件

无。

## 使用限制

- 前置操作和后置操作的ConfigMap数据必须放置在user-system命名空间下。
- 前置操作和后置操作均需要一个全局配置来指定要执行的操作。

## 使用场景

分为前置操作和后置操作两种，集群可以只包含其中的任意一种，也可以包含两种。

## 前置处理（PreProcess）

该章节介绍前置处理的全局配置用法，适用于所有节点统一执行脚本的场景。

### 功能与执行时机

前置处理在集群部署流程的`EnsureNodesEnv`阶段执行。
控制器会检查是否存在全局配置，若存在则对所有匹配节点下发内置命令`Preprocess`执行脚本。

### 全局配置ConfigMap

全局配置只需要创建一个`ConfigMap`，且需要满足以下要求。

- 所属命名空间为`user-system`。
- 名称为`preprocess-all-config`。
- `data`的名称为`config.json`，内容为前置操作的全局配置。


```json
apiVersion: v1
kind: ConfigMap
metadata:
  name: preprocess-all-config
  namespace: user-system
data:
  config.json: |
    {
      "scripts": [
        {
          "scriptName": "pre-post-process-test.sh",
          "order": 10,
          "params": {
            "HTTPS_REPO": "https://repo.local"
          }
        }
      ]
    }
```

字段说明：

- `scripts`：脚本列表（必填）。
- `scriptName`：脚本ConfigMap名称（必须一致）。
- `order`：执行顺序（从小到大）。
- `params`：脚本参数（可选）。

### 前置操作ConfigMap

每个前置操作必须单独创建`ConfigMap`，且需要满足以下要求。

- 所属命名空间为`user-system`。
- 名称与全局配置中的`scriptName`字段保持一致。
- 标签为`bke.preprocess.script: "true"`。
- `data`中必须有同名的key，内容为脚本文本。

下面是一个示例。

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pre-post-process-test.sh
  namespace: user-system
  labels:
    bke.preprocess.script: "true"
data:
  pre-post-process-test.sh: |
    #!/bin/sh
    echo "NODE_IP=${NODE_IP}"
    echo "HTTPS_REPO=${HTTPS_REPO}"
```

### 参数渲染与校验规则

脚本支持`${PARAM_NAME}`变量替换。

- 内置参数：`NODE_IP`，可以从节点的IP地址列表中自动获取。
- 自定义参数：来自全局配置的`params`。

参数名规则。

- 必须匹配：`^[a-zA-Z_][a-zA-Z0-9_]*$`。

参数值规则（白名单）。

- 仅允许：`a-z A-Z 0-9 - _ / . 空格 : # =`。
- 单个值最大长度：4096字节。

若传入的自定义参数不合法，该脚本会被跳过，不会导致整体失败。

### 执行与写盘

调谐器下发`Preprocess`内置命令的自定义资源到APIServer，bkeagent监控到命令后就会执行，当未显式传入`NODE_IP`时，bkeagent会自动获取当前节点IP地址（第一个非localhost的网口绑定的IP地址），最后会将渲染后的内容写入到`/etc/bkeagent/scripts`目录下。

### 使用示例（完整流程）

1. 准备前置操作的yaml文件。

   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: pre-post-process-test.sh
     namespace: user-system
     labels:
       bke.preprocess.script: "true"
   data:
     pre-post-process-test.sh: |
       #!/bin/sh
       echo "NODE_IP=${NODE_IP}"
       echo "HTTPS_REPO=${HTTPS_REPO}"
   ```

   

2. 准备全局配置的yaml文件。

   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: preprocess-all-config
     namespace: user-system
   data:
     config.json: |
       {
         "scripts": [
           {
             "scriptName": "pre-post-process-test.sh",
             "order": 10,
             "params": {
               "HTTPS_REPO": "https://repo.local"
             }
           }
         ]
       }
   ```

   

3. 创建集群的ConfigMap数据。

   执行下面命令，创建命名空间和ConfigMap数据。

   ```bash
   # 如果没有user-system的namespace，需要先创建
   kubectl create namespace user-system
   
   # 假设第1步的yaml文件名称是pre_deal.yaml，第2步的yaml文件是pre_cfg.yaml
   kubectl apply -f pre_deal.yaml -f pre_cfg.yaml
   ```

   

4. 应用以上两个ConfigMap后，所有符合条件的节点都会在`EnsureNodesEnv`阶段执行`pre-post-process-test.sh`。

## 后置处理（postprocess）


本章节文只介绍后置处理的全局配置用法，适用于所有节点统一执行脚本的场景。

### 功能与执行时机

后置处理在集群部署流程的`EnsureNodesPostProcess`阶段执行。
控制器检查是否存在全局配置，若存在则对所有匹配节点下发内置命令`Postprocess`执行脚本。

### 全局配置ConfigMap

全局配置只需要创建一个`ConfigMap`，且需要满足以下要求。

- 所属命名空间为`user-system`。
- 名称为`postprocess-all-config`。
- `data`的名称为`config.json`，内容为后置操作的全局配置。

```json
apiVersion: v1
kind: ConfigMap
metadata:
  name: postprocess-all-config
  namespace: user-system
data:
  config.json: |
    {
      "scripts": [
        {
          "scriptName": "post-clean.sh",
          "order": 10,
          "params": {
            "LOG_DIR": "/var/log"
          }
        }
      ]
    }
```

全局配置的字段说明：

- `scripts`：脚本列表（必填）。
- `scriptName`：脚本ConfigMap名称（必须一致）。
- `order`：执行顺序（从小到大）。
- `params`：脚本参数（可选）。

### 后置操作ConfigMap

每个后置操必须单独创建`ConfigMap`，且需要满足以下要求。

- 所属命名空间为`user-system`。
- 名称与全局配置中的`scriptName`字段保持一致。
- 标签为`bke.postprocess.script: "true"`。
- `data`中必须有同名的key，内容为脚本文本。

下面是一个示例。

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: post-clean.sh
  namespace: user-system
  labels:
    bke.postprocess.script: "true"
data:
  post-clean.sh: |
    #!/bin/sh
    echo "post clean on ${NODE_IP}"
    echo "log dir is ${LOG_DIR}"
```

### 参数渲染与校验规则

脚本支持`${PARAM_NAME}`变量替换。

- 内置参数：`NODE_IP`，可以从节点的IP地址列表中自动获取。
- 自定义参数：来自全局配置的`params`。

参数名规则。

- 必须匹配：`^[a-zA-Z_][a-zA-Z0-9_]*$`。

参数值规则（白名单）。

- 仅允许：`a-z A-Z 0-9 - _ / . 空格 : # =`。
- 单个值最大长度：4096字节。

若传入的自定义参数不合法，该脚本会被跳过，不会导致整体失败。

### 执行与写盘

调谐器下发`Postprocess`内置命令的自定义资源到APIServer，bkeagent监控到命令后就会执行，当未显式传入`NODE_IP`时，bkeagent会自动获取当前节点IP地址（第一个非localhost的网口绑定的IP地址），最后会将渲染后的内容写入到`/etc/bkeagent/scripts`目录下。

### 使用示例（完整流程）

1. 准备后置操作的yaml文件。

   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: post-clean.sh
     namespace: user-system
     labels:
       bke.postprocess.script: "true"
   data:
     post-clean.sh: |
       #!/bin/sh
       echo "post clean on ${NODE_IP}"
       echo "log dir is ${LOG_DIR}"
   ```

   

2. 准备全局配置的yaml文件。

   ```yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: postprocess-all-config
     namespace: user-system
   data:
     config.json: |
       {
         "scripts": [
           {
             "scriptName": "post-clean.sh",
             "order": 10,
             "params": {
               "LOG_DIR": "/var/log"
             }
           }
         ]
       }
   ```

   

3. 创建集群的ConfigMap数据。

   执行下面命令，创建命名空间和ConfigMap数据。

   ```bash
   # 如果没有user-system的namespace，需要先创建
   kubectl create namespace user-system
   
   # 假设第1步的yaml文件名称是post_deal.yaml，第2步的yaml文件是post_cfg.yaml
   kubectl apply -f post_deal.yaml -f post_cfg.yaml
   ```

   

4. 应用以上两个ConfigMap后，所有符合条件的节点都会在`EnsureNodesPostProcess`阶段执行`post-clean.sh`。

