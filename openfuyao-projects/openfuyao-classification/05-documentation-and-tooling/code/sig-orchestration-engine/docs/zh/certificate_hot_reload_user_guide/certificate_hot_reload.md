# 证书热更新

## 特性介绍

- Kubernetes、etcd、CoreDNS原生提供的证书热更新能力不完善，Kubernetes、etcd仅提供了部分证书热更新能力，CoreDNS完全不支持证书热更新。本特性旨在完善所有组件的证书热更新能力，以满足业务对于证书热加载的需求。

- 为简化描述，下文中所有“证书热更新”均指代处理证书、私钥和公钥等文件三者的热更新。

### 应用场景

- `Kubernetes`集群内通信的证书、私钥和CA证书更新后，所有组件自动完成证书热更新，所有组件间通信正常，业务组件访问`kube-apiserver`正常。
- SA密钥更新后，`kube-apiserver`、`kube-controller-manager`和`kubelet`自动完成证书热更新，业务组件访问`kube-apiserver`正常。

### 能力范围

**表1** Kubernetes组件证书热更新支持情况及其用途

| 组件                    | 证书/公钥/私钥对应参数                                       | 热更新支持情况 | 用途                                                         |
| ----------------------- | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| kube-apiserver          | --client-ca-file --requestheader-client-ca-file --tls-cert-file --tls-private-key-file --tls-sni-cert-key | 支持      | 服务端所需证书：用于与apiserver客户端通信。                  |
| kube-apiserver          | --etcd-cafile --etcd-certfile --etcd-keyfile                 | CA证书不支持 | 客户端所需证书：用于与etcd通信。                             |
| kube-apiserver          | --kubelet-certificate-authority --kubelet-client-certificate --kubelet-client-key | CA证书不支持 | 客户端所需证书：用于与kubelet通信。                          |
| kube-apiserver          | --peer-ca-file --proxy-client-cert-file --proxy-client-key-file | CA证书不支持 | 客户端所需证书：用于与peer通信（比如用户apiserver）。        |
| kube-apiserver          | --service-account-key-file --service-account-signing-key-file | 不支持     | 用于SA token签名及认证。                                     |
| kube-apiserver          | --oidc-ca-file                                               | 不支持     | OIDC客户端所需证书：用于对OpenID服务身份验证。               |
| kube-controller-manager | --client-ca-file --requestheader-client-ca-file --tls-cert-file --tls-private-key-file --tls-sni-cert-key | 支持      | 服务端所需证书：用于与controller-manger客户端通信。          |
| kube-controller-manager | --root-ca-file                                               | 不支持     | 根CA证书，会包含在kube-root-ca.crt configmap中以及在TokenSecret和注入token中。 |
| kube-controller-manager | --cluster-signing-key-file --cluster-signing-cert-file --cluster-signing-kube-apiserver-client-cert-file --cluster-signing-kube-apiserver-client-key-file --cluster-signing-kubelet-client-cert-file --cluster-signing-kubelet-client-key-file --cluster-signing-kubelet-serving-cert-file --cluster-signing-kubelet-serving-cert-file --cluster-signing-legacy-unknown-cert-file --cluster-signing-legacy-unknown-key-file | 支持      | 用于在证书轮转时为证书签名。                                 |
| kube-controller-manager | --service-account-private-key-file                           | 不支持     | 用于SA token签名。                                           |
| kube-controller-manager | --kubeconfig -certificate-authority -client-certificate -client-key | CA证书不支持 | 客户端所需证书：用于与apiserver通信。                        |
| kube-scheduler          | --client-ca-file --requestheader-client-ca-file --tls-cert-file --tls-private-key-file --tls-sni-cert-key | 支持      | 服务端所需证书：用于与客户端通信。                           |
| kube-scheduler          | --kubeconfig -certificate-authority -client-certificate -client-key | CA证书不支持 | 客户端所需证书：用于与apiserver通信。                        |
| kubelet                 | --client-ca-file --tls-cert-file --tls-private-key-file      | 不支持     | 服务端所需证书：用于与客户端通信。                           |
| kubelet                 | --kubeconfig -certificate-authority -client-certificate -client-key --bootstrap-kubeconfig -certificate-authority -client-certificate -client-key | CA证书不支持 | 客户端所需证书：用于与apiserver通信。                        |
| kube-proxy              | --kubeconfig: -certificate-authority -client-certificate -client-key | CA证书不支持 | 客户端所需证书：用于与apiserver通信。                        |
| etcd                    | --trusted-ca-file --cert-file --key-file --client-cert-file --client-key-file | CA证书不支持 | etcd客户端及服务端通信所需证书。                             |
| etcd                    | --peer-trusted-ca-file --peer-cert-file --peer-key-file --peer-client-cert-file --peer-client-key-file | CA证书不支持 | etcd实例间通信所需证书。                                     |
| etcd                    | --cacert --cert --key                                        | CA证书不支持 | etcd健康自检客户端所需证书。                                 |
| CoreDNS                 | kubernetes \{ tls: \}                                          | CA证书不支持 | 客户端所需证书：用于与apiserver安全通信。                    |
| CoreDNS                 | prometheus \{ tls: \}                                          | 不支持     | 服务端所需证书：用于与metrics接口调用方安全通信。            |
| CoreDNS                 | health \{ tls: \}                                              | 不支持     | 服务端所需证书：用于与health接口调用方安全通信。           |

### 亮点特征

- `kubernetes`集群内通信的证书、私钥和CA证书及SA密钥更新后，所有组件自动完成证书热更新。
- 业务不中断，所有组件间通信正常，业务组件访问`kube-apiserver`正常。

### 实现原理

- 实现方案包括服务端证书、私钥及客户端CA证书（可选）

  `tls.Config`支持配置`GetConfigForClient`方法，它会在每次新建tls连接时被调用以获取一个`tls.Config`。我们为`tls.Config`注册这个方法，并在方法里实现如下功能来实现热更新：

  创建一个`tls.Config`，并将内存中服务端证书、CA证书内容设置到`Certificates`和`ClientCAs`字段，服务端证书和CA证书内容由另外的协程进行侦听变化并动态刷新到内存。

  ```
  type Config struct {
      ... ...
      // 设备证书（用于服务端时保存服务端证书，用于客户端时保存客户端证书）
      Certificates []Certificate
      // 客户端CA证书
      ClientCAs *x509.CertPool
      // 动态提供tls.Config
  	  GetConfigForClient func(*ClientHelloInfo) (*Config, error)    
      ... ...
  }
  ```

- TLS客户端通信所需证书：包括客户端证书（可选）、私钥（可选）及服务端CA证书（可选）

  在`tls.Config`中注册方法`GetClientCertificate`，并在方法里实现如下功能：

  从内存中将客户端证书内容设置到`Certificates`字段，客户端证书内容由另外的协程进行侦听变化并动态刷新到内存。

  ```
  type Config struct {
      ... ...
      // 服务端CA证书
      RootCAs *x509.CertPool
      // 动态提供客户端证书
  	  GetClientCertificate func(*CertificateRequestInfo) (*Certificate, error)   
      ... ...
  }
  ```

  侦听CA文件变化，CA更新后重新创建`x509.CertPool`，并对`tls.Config`中原`x509.CertPool`对象进行替换。

- SA公钥和私钥(kube-apiserver/kube-controller-manager/kubelet)

  侦听SA公钥和私钥文件变化，文件内容更新后，更新使用SA公钥或私钥的对象，包括SA token生成器、SA token校验器等。同时需要刷新已分配的token：

1. token secret：由kube-controller-manager进行刷新。

2. 注入到Pod的token：由kubelet进行刷新，kubelet未配置SA密钥文件，无法感知其变化，需做如下处理：

   2.1 kube-apiserver侦听SA私钥文件变化，并将私钥hash发布为configmap。

   2.2 kubelet侦听密钥hash configmap变化，一旦密钥hash发生变化，则为当前节点所有Pod重新注入SA token。

- 根CA证书（kube-controller-manager/kubelet）

  侦听根证书文件变化，文件内容更新后，kube-controller-manager需刷新所有命名空间下的kube-root-ca.crt及所有token secret中的CA证书，kubelet为所有Pod重新注入包含最新CA证书的SA token。

### 与相关特性的关系

无特殊依赖。

## 安装

无需手动安装，自启用功能。

## 使用证书热更新

### 前提条件

无。

### 背景信息

- 用于`kubernetes`集群内通信的证书、私钥和CA证书更新后，所有组件自动完成证书热更新，所有组件间通信正常，业务组件访问`kube-apiserver`正常。

### 使用限制

- kubeconfig文件中内嵌的证书、私钥不支持热更新。
- CoreDNS仅考虑集群内交互所需的证书，forward、gRPC、etcd等插件的证书暂不支持热更新。
- 外部更新证书、私钥后，CaaSCore组件会在1min内完成证书热更新，5min以内完成SA token刷新。在证书热更新前，集群功能可能不可用。在SA token刷新完成前，业务Pod可能无法访问kube-apiserver；规避措施：各组件访问服务端失败时，有退避重试机制，不同组件和客户端的请求重试逻辑可能不一致，建议每30s重试一次，时长5min。

### 操作步骤

无需手动操作。SA密钥、CA证书或集群通信证书更新后自动完成Kubernetes组件证书更新。
