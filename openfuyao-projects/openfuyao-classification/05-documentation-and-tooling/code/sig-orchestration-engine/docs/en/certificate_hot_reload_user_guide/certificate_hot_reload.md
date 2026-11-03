# Certificate Hot Reload

## Feature Introduction

- The native certificate hot reload capabilities provided by Kubernetes, etcd, and CoreDNS are incomplete. Kubernetes and etcd only provide partial certificate hot reload capabilities, while CoreDNS does not support certificate hot reload at all. This feature aims to improve certificate hot reload capabilities for all components to meet business requirements for certificate hot reload.

- For simplicity, all references to "certificate hot reload" below refer to hot updates of certificates, private keys, and public key files.

### Use Cases

- After certificates, private keys, and CA certificates used for communication within the `Kubernetes` cluster are updated, all components automatically complete certificate hot reload, all inter-component communication works normally, and business components can access `kube-apiserver` normally.
- After SA keys are updated, `kube-apiserver`, `kube-controller-manager`, and `kubelet` automatically complete certificate hot reload, and business components can access `kube-apiserver` normally.

### Supported Scope

**Table 1** Kubernetes component certificate hot reload support and usage

| Component               | Certificate/Public Key/Private Key Parameters                 | Hot Reload Support | Usage                                                        |
| ----------------------- | ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| kube-apiserver          | --client-ca-file --requestheader-client-ca-file --tls-cert-file --tls-private-key-file --tls-sni-cert-key | Supported      | Server-side certificates: used for communication with apiserver clients.                  |
| kube-apiserver          | --etcd-cafile --etcd-certfile --etcd-keyfile                 | CA certificate not supported | Client-side certificates: used for communication with etcd.                             |
| kube-apiserver          | --kubelet-certificate-authority --kubelet-client-certificate --kubelet-client-key | CA certificate not supported | Client-side certificates: used for communication with kubelet.                          |
| kube-apiserver          | --peer-ca-file --proxy-client-cert-file --proxy-client-key-file | CA certificate not supported | Client-side certificates: used for communication with peers (e.g., user apiservers).        |
| kube-apiserver          | --service-account-key-file --service-account-signing-key-file | Not supported     | Used for SA token signing and authentication.                                     |
| kube-apiserver          | --oidc-ca-file                                               | Not supported     | OIDC client certificates: used for OpenID service authentication.               |
| kube-controller-manager | --client-ca-file --requestheader-client-ca-file --tls-cert-file --tls-private-key-file --tls-sni-cert-key | Supported      | Server-side certificates: used for communication with controller-manager clients.          |
| kube-controller-manager | --root-ca-file                                               | Not supported     | Root CA certificate, included in the kube-root-ca.crt configmap and in TokenSecrets and injected tokens. |
| kube-controller-manager | --cluster-signing-key-file --cluster-signing-cert-file --cluster-signing-kube-apiserver-client-cert-file --cluster-signing-kube-apiserver-client-key-file --cluster-signing-kubelet-client-cert-file --cluster-signing-kubelet-client-key-file --cluster-signing-kubelet-serving-cert-file --cluster-signing-kubelet-serving-cert-file --cluster-signing-legacy-unknown-cert-file --cluster-signing-legacy-unknown-key-file | Supported      | Used for signing certificates during certificate rotation.                                 |
| kube-controller-manager | --service-account-private-key-file                           | Not supported     | Used for SA token signing.                                           |
| kube-controller-manager | --kubeconfig -certificate-authority -client-certificate -client-key | CA certificate not supported | Client-side certificates: used for communication with apiserver.                        |
| kube-scheduler          | --client-ca-file --requestheader-client-ca-file --tls-cert-file --tls-private-key-file --tls-sni-cert-key | Supported      | Server-side certificates: used for communication with clients.                           |
| kube-scheduler          | --kubeconfig -certificate-authority -client-certificate -client-key | CA certificate not supported | Client-side certificates: used for communication with apiserver.                        |
| kubelet                 | --client-ca-file --tls-cert-file --tls-private-key-file      | Not supported     | Server-side certificates: used for communication with clients.                           |
| kubelet                 | --kubeconfig -certificate-authority -client-certificate -client-key --bootstrap-kubeconfig -certificate-authority -client-certificate -client-key | CA certificate not supported | Client-side certificates: used for communication with apiserver.                        |
| kube-proxy              | --kubeconfig: -certificate-authority -client-certificate -client-key | CA certificate not supported | Client-side certificates: used for communication with apiserver.                        |
| etcd                    | --trusted-ca-file --cert-file --key-file --client-cert-file --client-key-file | CA certificate not supported | Certificates required for etcd client and server communication.                             |
| etcd                    | --peer-trusted-ca-file --peer-cert-file --peer-key-file --peer-client-cert-file --peer-client-key-file | CA certificate not supported | Certificates required for communication between etcd instances.                                     |
| etcd                    | --cacert --cert --key                                        | CA certificate not supported | Certificates required for etcd health self-check clients.                                 |
| CoreDNS                 | kubernetes \{ tls: \}                                          | CA certificate not supported | Client-side certificates: used for secure communication with apiserver.                    |
| CoreDNS                 | prometheus \{ tls: \}                                          | Not supported     | Server-side certificates: used for secure communication with metrics interface callers.            |
| CoreDNS                 | health \{ tls: \}                                              | Not supported     | Server-side certificates: used for secure communication with health interface callers.           |

### Highlights

- After certificates, private keys, CA certificates, and SA keys used for communication within the `kubernetes` cluster are updated, all components automatically complete certificate hot reload.
- Business is not interrupted, all inter-component communication works normally, and business components can access `kube-apiserver` normally.

### Implementation Principle

- Implementation for server-side certificates, private keys, and client-side CA certificates (optional)

  `tls.Config` supports configuring the `GetConfigForClient` method, which is called each time a new TLS connection is established to obtain a `tls.Config`. We register this method in `tls.Config` and implement the following functionality to achieve hot reload:

  Create a `tls.Config` and set the in-memory server certificate and CA certificate content to the `Certificates` and `ClientCAs` fields; the server certificate and CA certificate content are monitored for changes by a separate goroutine and dynamically refreshed into memory.

  ```
  type Config struct {
      ... ...
      // Device certificates (server certificates when used as server, client certificates when used as client)
      Certificates []Certificate
      // Client CA certificates
      ClientCAs *x509.CertPool
      // Dynamically provides tls.Config
  	  GetConfigForClient func(*ClientHelloInfo) (*Config, error)    
      ... ...
  }
  ```

- TLS client communication certificates: including client certificates (optional), private keys (optional), and server CA certificates (optional)

  Register the `GetClientCertificate` method in `tls.Config` and implement the following functionality:

  Set the in-memory client certificate content to the `Certificates` field; the client certificate content is monitored for changes by a separate goroutine and dynamically refreshed into memory.

  ```
  type Config struct {
      ... ...
      // Server CA certificates
      RootCAs *x509.CertPool
      // Dynamically provides client certificates
  	  GetClientCertificate func(*CertificateRequestInfo) (*Certificate, error)   
      ... ...
  }
  ```

  Monitor CA file changes; after CA updates, recreate `x509.CertPool` and replace the original `x509.CertPool` object in `tls.Config`.

- SA public and private keys (kube-apiserver/kube-controller-manager/kubelet)

  Monitor SA public and private key file changes; after file content updates, update objects that use SA public or private keys, including SA token generators, SA token validators, etc. Simultaneously refresh already-assigned tokens:

1. Token secrets: refreshed by kube-controller-manager.

2. Tokens injected into Pods: refreshed by kubelet. Since kubelet is not configured with SA key files and cannot detect their changes, the following handling is needed:

   2.1 kube-apiserver monitors SA private key file changes and publishes the private key hash as a ConfigMap.

   2.2 kubelet monitors the key hash ConfigMap; once the key hash changes, it re-injects SA tokens for all Pods on the current node.

- Root CA certificate (kube-controller-manager/kubelet)

  Monitor root certificate file changes; after file content updates, kube-controller-manager needs to refresh kube-root-ca.crt in all namespaces and CA certificates in all token secrets; kubelet re-injects SA tokens containing the latest CA certificate for all Pods.

### Relationship with Related Features

No special dependencies.

## Installation

No manual installation required; the feature is auto-enabled.

## Using Certificate Hot Reload

### Prerequisites

None.

### Background Information

- After certificates, private keys, and CA certificates used for communication within the `kubernetes` cluster are updated, all components automatically complete certificate hot reload, all inter-component communication works normally, and business components can access `kube-apiserver` normally.

### Usage Restrictions

- Certificates and private keys embedded in kubeconfig files do not support hot reload.
- CoreDNS only considers certificates required for cluster-internal interaction; certificates for plugins such as forward, gRPC, and etcd do not currently support hot reload.
- After external certificate and private key updates, CaaSCore components complete certificate hot reload within 1 minute and SA token refresh within 5 minutes. Before certificate hot reload, cluster functions may be unavailable. Before SA token refresh is complete, business Pods may be unable to access kube-apiserver; mitigation: when components fail to access the server, a backoff retry mechanism is in place. The retry logic may differ across components and clients; it is recommended to retry every 30 seconds for a duration of 5 minutes.

### Procedure

No manual operation required. Kubernetes component certificate updates are completed automatically after SA keys, CA certificates, or cluster communication certificates are updated.
