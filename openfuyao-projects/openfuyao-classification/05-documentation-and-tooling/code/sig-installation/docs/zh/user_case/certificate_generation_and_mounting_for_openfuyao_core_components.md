# openFuyao核心组件的证书生成和挂载

本文档旨提供已使用openFuyao安装部署工具进行安装后的openFuyao容器平台上的核心组件进行证书配置的通用方案。该方案需要用户将所有证书配置在`openfuyao-system`命名空间的secret中，供各组件使用。

# 为核心组件生成证书

## 文件说明

### getCA.bash

从`openfuyao-system`命名空间中获取`CA`（包括初始的和经过base64解码的）。

```bash
#!/bin/bash
# 获取ca.crt
kubectl get secret -n openfuyao-system openfuyao-system-root-ca -oyaml | grep "ca.crt: " | cut -d' ' -f4 > original_ca.crt
# base64解码
cat original_ca.crt | base64 -d > base64_ca.crt

# 获取ca.key
kubectl get secret -n openfuyao-system openfuyao-system-root-ca -oyaml | grep "ca.key: " | cut -d' ' -f4 > original_ca.key
# base64解码
cat original_ca.key | base64 -d > base64_ca.key

cp base64_ca.crt ca.crt
cp base64_ca.key ca.key
```

### createCert.bash

使用指定`CA`为服务生成证书。

```bash
#!/bin/bash
# 1.设置生成证书对象
set -e

NAMESPACE="openfuyao-system"
SERVICE_NAME="monitoring-service"
# SERVICE_NAME="user-management-operator"
# SERVICE_NAME="web-terminal-service"
# SERVICE_NAME="application-management-service"
# SERVICE_NAME="plugin-management-service"
# SERVICE_NAME="marketplace-service"
SECRET_NAME="${SERVICE_NAME}-tls"

# 2.重新生成指定对象webhook证书
echo "开始重新生成 webhook 证书..."

# 3.生成服务器私钥
echo "生成服务器私钥..."
openssl genrsa -out ${SERVICE_NAME}.key 4096

# 4.创建服务器证书配置文件
echo "创建证书配置文件..."
cat > ${SERVICE_NAME}.conf <<EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = ${SERVICE_NAME}.${NAMESPACE}.svc

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth, clientAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = ${SERVICE_NAME}
DNS.2 = ${SERVICE_NAME}.${NAMESPACE}
DNS.3 = ${SERVICE_NAME}.${NAMESPACE}.svc
DNS.4 = ${SERVICE_NAME}.${NAMESPACE}.svc.cluster.local
EOF

# 5.生成证书签名请求
echo "生成证书签名请求..."
openssl req -new -key ${SERVICE_NAME}.key -out ${SERVICE_NAME}.csr -config ${SERVICE_NAME}.conf

# 6.生成服务器证书
echo "生成服务器证书..."
openssl x509 -req -in ${SERVICE_NAME}.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out ${SERVICE_NAME}.crt -days 1095 -sha256 -extensions v3_req -extfile ${SERVICE_NAME}.conf

# 7.验证证书
echo "验证生成的证书..."
echo "证书主体："
openssl x509 -in ${SERVICE_NAME}.crt -text -noout | grep "Subject:"
echo "证书 SAN："
openssl x509 -in ${SERVICE_NAME}.crt -text -noout | grep -A 10 "Subject Alternative Name"

# 8.检查命名空间是否存在
echo "检查命名空间 ${NAMESPACE}..."
if ! kubectl get namespace ${NAMESPACE} >/dev/null 2>&1; then
    echo "创建命名空间 ${NAMESPACE}..."
    kubectl create namespace ${NAMESPACE}
fi

# 9.删除旧的Secret（如果存在）
echo "删除旧的证书 Secret..."
kubectl delete secret ${SECRET_NAME} -n ${NAMESPACE} --ignore-not-found=true

# 10.创建新的Secret，使用正确的键名匹配Helm模板
echo "创建新的证书 Secret..."
kubectl create secret generic ${SECRET_NAME} -n ${NAMESPACE} \
    --from-file=ca.crt=ca.crt \
    --from-file=tls.key=${SERVICE_NAME}.key \
    --from-file=tls.crt=${SERVICE_NAME}.crt

# 11.验证Secret创建成功
echo "验证 Secret 创建..."
kubectl get secret ${SECRET_NAME} -n ${NAMESPACE} -o yaml

# 12.编码证书为base64（用于webhook配置）
echo "编码证书..."
# CA_BUNDLE=$(cat ca.crt | base64 | tr -d '\n')
CA_BUNDLE=$(cat ca.crt | tr -d '\n')

echo ""
echo "==============================================="
echo "证书重新生成完成！"
echo "==============================================="
echo ""
echo "Secret 信息:"
echo "- 命名空间: ${NAMESPACE}"
echo "- Secret名称: ${SECRET_NAME}"
echo "- 包含的键:"
echo "  * ca.crt (CA证书)"
echo "  * tls.key (服务器私钥)"
echo "  * tls.crt (服务器证书)"
echo ""
echo "Helm 模板中的映射关系:"
echo "- ca.crt -> /ssl/ca.pem"
echo "- tls.key -> /ssl/server.key"  
echo "- tls.crt -> /ssl/server.crt"
echo ""
echo "新的 CA Bundle (用于 webhook 配置):"
echo "${CA_BUNDLE}"
echo ""
echo "您现在可以使用以下命令验证证书挂载:"
echo "kubectl exec -it <pod-name> -n ${NAMESPACE} -- ls -la /ssl/"
echo ""
echo "重启相关 Pod 以应用新证书:"
echo "kubectl rollout restart deployment/${SERVICE_NAME} -n ${NAMESPACE}"
```

## 使用说明

1. 将`getCA.bash`和`createCert.bash`放置到用户指定的文件夹下（后续操作须在此文件夹下进行）。
2. 设置上述文件为可执行`chmod u+x getCA.bash`。
3. 运行`getCA.bash`获取CA。
4. 运行`createCert.bash`生成证书（修改`SERVICE_NAME`可以为不同服务生成证书）。

    对以下6个组件生成证书：

    ```bash
    SERVICE_NAME="monitoring-service"
    # SERVICE_NAME="user-management-operator"
    # SERVICE_NAME="web-terminal-service"
    # SERVICE_NAME="application-management-service"
    # SERVICE_NAME="plugin-management-service"
    # SERVICE_NAME="marketplace-service"
    ```

# 证书挂载

## 任务一：修改`openfuyao-system`命名空间下的`console-service-config` configmap

> **说明：** 下文第一个 YAML 片段为修改前的配置现状（各服务通过 http 协议访问），第二个 YAML 片段为修改后的目标配置。http协议存在安全风险，建议您使用https安全协议。

### 操作步骤

1. 执行`kubectl edit cm -n openfuyao-system console-service-config`，对configmap的如下部分进行修改：
   
   ```yaml
   apiVersion: v1
   data:
     alert-host: http://alertmanager-main.monitoring.svc.cluster.local:9093
     application-management-host: http://application-management-service.openfuyao-system.svc.cluster.local
     console-service-host: https://console-service.openfuyao-system.svc.cluster.local:443
     console-website-host: https://console-website.openfuyao-system.svc.cluster.local:80
     insecure-skip-verify: "false"
     marketplace-host: http://marketplace-service.openfuyao-system.svc.cluster.local
     monitoring-host: http://monitoring-service.openfuyao-system.svc.cluster.local:80
     oauth-server-host: https://oauth-server.openfuyao-system.svc.cluster.local:9096
     plugin-management-host: http://plugin-management-service.openfuyao-system.svc.cluster.local
     server-name: ""
     user-management-host: http://user-management-operator.openfuyao-system.svc.cluster.local:80
     webterminal-host: http://web-terminal-service.openfuyao-system.svc.cluster.local:9072
   kind: ConfigMap
   ...
   ```

2. 修改后的结果应为：
   
   ```yaml
   apiVersion: v1
   data:
     alert-host: http://alertmanager-main.monitoring.svc.cluster.local:9093
     application-management-host: https://application-management-service.openfuyao-system.svc.cluster.local:80
     console-service-host: https://console-service.openfuyao-system.svc.cluster.local:80
     console-website-host: https://console-website.openfuyao-system.svc.cluster.local:80
     insecure-skip-verify: "false"
     marketplace-host: https://marketplace-service.openfuyao-system.svc.cluster.local:80
     monitoring-host: https://monitoring-service.openfuyao-system.svc.cluster.local:80
     oauth-server-host: https://oauth-server.openfuyao-system.svc.cluster.local:9096
     plugin-management-host: https://plugin-management-service.openfuyao-system.svc.cluster.local:80
     server-name: ""
     user-management-host: https://user-management-operator.openfuyao-system.svc.cluster.local:80
     webterminal-host: https://web-terminal-service.openfuyao-system.svc.cluster.local:9072
   kind: ConfigMap
   ...
   ```

3. 重启`console-service`的Pod，执行`kubectl delete po -n openfuyao-system console-service-xxxxxxx-xxxxx`，观察到对应的Pod重新启动并running。

## 任务二：将`user-management-operator`、`web-terminal-service`相关证书挂载到自身容器

### 操作步骤

这里以挂载`user-management-operator-tls`到容器内为例：

1. 执行命令`kubectl edit deployment/user-management-operator -n openfuyao-system`对`deployment`的部署`yaml`进行修改。
2. 找到`spec.template.spec.containers`为`user-management-operator`的`container`的配置项添加`volumeMounts`配置。

    ```yaml
                volumeMounts:
                - name: user-management-operator-tls
                  mountPath: /ssl/ca.pem
                  subPath: ca.crt
                - name: user-management-operator-tls
                  readOnly: true
                  mountPath: /ssl/server.key
                  subPath: tls.key
                - name: user-management-operator-tls
                  readOnly: true
                  mountPath: /ssl/server.crt
                  subPath: tls.crt
    ```

3. 为`volumes`添加以下配置。

    ```yaml
                - name: user-management-operator-tls
                  secret:
                    defaultMode: 0600
                    secretName: user-management-operator-tls
    ```

4. 保存退出并执行命令`kubectl describe deployment/user-management-operator -n openfuyao-system`查看是否挂载成功。

## 任务三：将`monitoring-service`、`application-management-service`、`plugin-management-service` 、`marketplace-service`的证书挂载到`oauth-proxy`容器

### 操作步骤

这里以挂载`monitoring-service-tls`到容器内为例：

1. 执行命令`kubectl edit deployment/monitoring-service -n openfuyao-system`对`deployment`的部署`yaml`进行修改，执行完命令后会进入文件修改。
2. 找到`spec.template.spec.containers`为`oauth-proxy`容器的`args`添加下方代码块中注释内容。

    ```yaml
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          annotations:
            deployment.kubernetes.io/revision: "1"
            meta.helm.sh/release-name: monitoring-service
            meta.helm.sh/release-namespace: openfuyao-system
          creationTimestamp: "2025-09-10T02:19:56Z"
          generation: 1
          labels:
            app.kubernetes.io/managed-by: Helm
          name: monitoring-service
          namespace: openfuyao-system
          resourceVersion: "3115"
          uid: 76f1e52d-5153-485b-9607-2c987ce2f0f2
        spec:
          progressDeadlineSeconds: 600
          replicas: 1
          revisionHistoryLimit: 10
          selector:
            matchLabels:
              app: monitoring-service
          strategy:
            rollingUpdate:
              maxSurge: 25%
              maxUnavailable: 25%
            type: RollingUpdate
          template:
            metadata:
              creationTimestamp: null
              labels:
                app: monitoring-service
            spec:
              containers:
              - args:
                - --https-address=:9093 # 需要修改
                - --http-address= # 需要修改
                - --email-domain=*
                - --provider=openfuyao
                - --client-id=oauth-proxy
                - --client-secret=SECRETTS
                - --tls-cert=/ssl/server.crt # 需要修改
                - --tls-key=/ssl/server.key # 需要修改
                - --tls-client-ca=/ssl/ca.pem # 需要修改
                - --upstream=http://localhost:9083  # pod内部信任域网络访问，可使用http
                - '--openfuyao-delegate-urls={"/":{"resource": "services/proxy", "group":
                  ""}}'
                - --redirect-url=/
                - --login-url=/oauth2/oauth/authorize
                - --redeem-url=/oauth2/oauth/token
                - --root-prefix=/
                - --cookie-httponly
                image: harbor.openfuyao.com/openfuyao/oauth-proxy:latest
        ...
    ```

3. 为指定容器（oauth-proxy）添加`volumeMounts`配置。

    ```yaml
                volumeMounts:
                - name: monitoring-service-tls
                  mountPath: /ssl/ca.pem
                  subPath: ca.crt
                - name: monitoring-service-tls
                  readOnly: true
                  mountPath: /ssl/server.key
                  subPath: tls.key
                - name: monitoring-service-tls
                  readOnly: true
                  mountPath: /ssl/server.crt
                  subPath: tls.crt
    ```

4. 为`oauth-proxy`容器的`volumes`添加以下配置。

    ```yaml
                - name: monitoring-service-tls
                  secret:
                    defaultMode: 0600
                    secretName: monitoring-service-tls
    ```

5. 保存退出并执行命令`kubectl describe deployment/monitoring-service -n openfuyao-system`查看是否挂载成功。

## 任务四：将`application-management-service`、 `marketplace-service`的证书挂载到自身容器中，用于RPC通信的TLS保障

### 操作步骤

这里以挂载`application-management-service-tls`到容器内为例：

1. 执行命令`kubectl edit deploy application-management-service -n openfuyao-system`对`deployment`的部署`yaml`进行修改，执行完命令后会进入文件修改。
2. 为指定容器（application-management-service）添加`volumeMounts`配置（使用任务三中挂载的相同证书）。

    ```yaml
                volumeMounts:
                - name: application-management-service-tls
                  mountPath: /ssl/ca.pem
                  subPath: ca.crt
                - name: application-management-service-tls
                  readOnly: true
                  mountPath: /ssl/server.key
                  subPath: tls.key
                - name: application-management-service-tls
                  readOnly: true
                  mountPath: /ssl/server.crt
                  subPath: tls.crt
    ```

3. 保存退出并执行命令`kubectl describe deploy application-management-service -n openfuyao-system`查看是否挂载成功。
