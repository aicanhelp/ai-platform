# 面向智能路由的高实时性指标采集与上报

面向对时效性要求高的模块（如智能路由），通过NATS实现秒级指标推送，确保推理过程中的关键性能指标（如等待执行的推理请求数、NPU/GPU KV Cache利用率等）能够被及时感知并影响决策。

目前文档只支持**vLLM**部署，未来会扩展到支持其他推理引擎和服务。

## 前置条件

- Kubernetes集群已部署并可正常访问
- 已安装`kubectl`命令行工具并配置好集群访问权限
- 节点上已安装Ascend驱动和相关依赖
- NATS已部署并正常运行

## 部署步骤

选用**NATS发布/订阅**模型作为指标传输方式。`CustomStatLogger`是在**vLLM**中通过继承`StatLoggerBase`抽象类实现的自定义统计数据记录器。作为**发布者**，`CustomStatLogger`会在每次解码批次结束后，主动生成结构化的运行指标消息，并将其发布到指定的**NATS**主题；`Router`作为**订阅者**，只需在该主题上注册监听，即可在消息发布的瞬间接收并触发后续处理逻辑。

为避免首批消息丢失，订阅端（`Router`）需先于发布端（`CustomStatLogger`）完成连接与订阅。

### Router (订阅端)

`Router`作为**订阅者**，主要负责接收发布的运行指标消息并触发相应的处理逻辑。参考以下代码示例来实现`Router`，并将其部署到Kubernetes集群中

```
import logging
import json
from nats.aio.client import Client as NATS_Client

logger = logging.getLogger()

class RouterSubscriber:
    def __init__(self, server_address: str, topic: str):
        """
        初始化RouterSubscriber实例
        :param server_address: NATS服务器地址
        :param topic: 订阅的主题
        """
        self.server_address = server_address
        self.topic = topic
        self.client = NATS_Client()

    async def connect(self):
        """
        建立与NATS服务器的连接
        """
        try:
            # 建立NATS长连接
            await self.client.connect(self.server_address)
            logger.info("Connected to NATS server at %s", self.server_address)
        except Exception as e:
            logger.error("Failed to connect to NATS server: %s", e)
            raise e

    async def subscribe(self, topic: str, message_handler):
        """
        订阅指定主题，并将接收到的消息交给业务逻辑处理
        :param message_handler: 业务逻辑处理函数，用于处理接收到的消息
        """
        try:
            async def on_message(msg):
                """
                消息回调函数，处理订阅到的消息
                :param msg: NATS消息
                """
                logger.info("Received message on topic %s", topic)
                await message_handler(msg)

            # 确保NATS连接
            if not self.client.is_connected:
                await self.connect()

            # 注册订阅主题，并绑定消息回调函数
            await self.client.subscribe(topic, cb=on_message)
            logger.info("Successfully subscribed to NATS topic %s", topic)

        except Exception as e:
            logger.error("Failed to subscribe to NATS topic %s: %s", topic, e)
            raise e
            
    async def message_handler(self, msg) -> None:
        """
        NATS原始回调，负责：
        1. 从msg中解码数据；
        2. 调用parse_message做JSON反序列化；
        3. 将解析后的结构化数据交给后续业务逻辑处理。
        """
        try:
            data = msg.data.decode("utf-8")
        except Exception as exc:
            logger.error("Failed to decode NATS message: %s", exc)
            return

        data_dict = self.parse_message(data)
        if not data_dict:
            # 解析失败直接返回，避免后续逻辑异常
            return

        # TODO: 在这里处理具体业务逻辑，例如根据指标做路由决策
        ...

    def parse_message(self, data: str) -> Dict[str, Any]:
        """
        将NATS消息体从JSON字符串反序列化为字典。
        解析失败时记录日志并返回空字典。
        """
        try:
            return json.loads(data)
        except Exception as exc:
            logger.error("Failed to parse message: %s, raw: %r", exc, data)
            return {}

```

### Customstatlogger (发布端)

`CustomStatLogger`是**发布者**，它负责生成并发布实时的运行指标数据。每次解码批次结束后，`CustomStatLogger`将结构化的指标消息发布到指定的**NATS**主题（该主题应与订阅方的主题匹配，当前主题`eagle_eye.routing_metrics`），供`Router`订阅。

#### 1. 打包源代码

在项目根目录下执行以下命令，将源代码打包为压缩文件：

```bash
tar -czf source_code.tar.gz src/ requirements.txt
```

#### 2. 创建ConfigMap

将压缩包作为二进制文件导入到ConfigMap中：

```bash
# 检查是否存在namespace，如果不存在则创建它
kubectl get namespace ai-inference || kubectl create namespace ai-inference

# 如果已存在同名ConfigMap，先删除
kubectl delete configmap vllm-source-code -n ai-inference --ignore-not-found

# 创建新的ConfigMap
kubectl create configmap vllm-source-code \
  --from-file=source_code.tar.gz=source_code.tar.gz \
  -n ai-inference
```

#### 3. 部署服务

应用部署YAML文件：

```bash
kubectl apply -f ./docs/routing-metrics/vllm_eagle_eye.yaml
```

#### 4. 检查部署状态

```bash
# 查看Pod运行状态
kubectl get pods -n ai-inference -l app=vllm-eagle-eye

# 查看Pod详细信息
kubectl describe pod -n ai-inference -l app=vllm-eagle-eye

# 查看Pod日志
kubectl logs -n ai-inference -l app=vllm-eagle-eye -f
```

## 验证步骤

### 方式一：使用临时Pod验证

#### 1. 获取服务Pod IP

```bash
kubectl get pod -n ai-inference -l app=vllm-eagle-eye -o wide
```

记录输出中的`IP`地址（例如：`10.244.1.5`）。

#### 2. 启动临时测试 Pod

```bash
kubectl run curl-test --image=curlimages/curl -n ai-inference -it --rm --restart=Never -- sh
```

#### 3. 在临时Pod中发送测试请求

在临时Pod的shell中执行（将`<POD_IP>`替换为实际IP）：

```bash
curl -X POST http://<POD_IP>:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello AI"}'
```

### 方式二：使用port-forward验证

```bash
# 端口转发
kubectl port-forward -n ai-inference deployment/vllm-eagle-eye 8000:8000

# 在另一个终端发送请求
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello AI"}'
```

## 清理资源

```bash
# 删除Deployment
kubectl delete -f ./docs/routing-metrics/vllm_eagle_eye.yaml

# 删除ConfigMap
kubectl delete configmap vllm-source-code -n ai-inference

# 删除临时Pod
kubectl delete pod -n ai-inference curl-test
```

## 常见问题

### Pod启动失败

```bash
# 查看Pod事件
kubectl describe pod -n ai-inference -l app=vllm-eagle-eye

# 查看init容器日志
kubectl logs -n ai-inference <pod-name> -c code-extractor
```

### 依赖安装失败

如果集群无法联网，需要提前将依赖包打入压缩包：

```bash
# 下载依赖到本地
pip download -r requirements.txt -d packages/

# 打包时包含依赖
tar -czf source_code.tar.gz src/ requirements.txt packages/
```
