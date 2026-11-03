# Installation Notes

## Supported Environments

This section introduces the operating systems supported by the openFuyao installation tool.

### Operating Systems

**Table 1** Supported operating system information

| Operating System  | Release Version | Kernel Version | Architecture       |
| --------- |-------| ---------- | ---------- |
| openEuler | 20.03 | 4.19.* | ARM64, x86_64 |
| openEuler | 22.03 | 5.10.* | ARM64, x86_64 |
| openEuler | 24.03 | 6.6.* | ARM64, x86_64 |
| Ubuntu    | 22.04 | 5.15.* | ARM64, x86_64 |
| UOS       | 20 1070e | 4.19.* | ARM64, x86_64 |
| UOS       | 20 1070e | 5.10.* | ARM64, x86_64 |

> ![icon](../figures/icon-note.gif) **Note:**<br />Other operating system versions have not been tested and may cause unexpected issues.
>
> The community performs feature development and self-verification based on the 5.10 kernel, and conducts compatibility testing in the openEuler environment. The kernel versions include 4.19, 5.10, 5.15, and 6.6.
>
> The openFuyao cluster currently supports only IPv4 scenarios and does not support IPv6.

## Precautions

- It is recommended to prepare two machines to serve as the bootstrap node and the service cluster node respectively. They can also be co-deployed on a single machine (in which case K3s and Kubernetes will run simultaneously and may affect each other, which is not recommended).
- When logging in to the openFuyao management plane on the bootstrap node for the first time, the username is admin and the password is test@1234. You need to change the username and password after the first login. The HTTPS protocol is used, and the default web service port is 30010.
- When logging in to the openFuyao management plane on the service cluster node for the first time, the username is admin and the password is test@1234. You need to change the username and password after the first login. The HTTPS protocol is used, and the default web service port is 31616.
- It is recommended that your node environment be a clean operating system environment without any docker or Kubernetes components installed, otherwise version conflicts may occur and cause installation failures. If you need to perform an experimental installation in an existing environment, you can download the [environment pre-check tool](environment_pre_check_tool_guide.md) for environment verification.
- Ensure that the time on the cluster node to be installed is consistent with the time on the bootstrap node, with a maximum time difference of 10s.
- Optional: If you need to use a custom CA certificate to issue certificates for all components within the Kubernetes cluster to meet enterprise-level security compliance requirements, you can configure the custom certificate before initializing the bootstrap node. For detailed operations, see [Custom Cluster Certificate Usage Guide](../bootstrap_cluster_configuration/using_custom_cluster_certificates.md).
- Optional: If you need to configure a custom repository for installation, see [Custom Repository Usage Guide](../bootstrap_cluster_configuration/custom_repository_configuration_guide.md).
- Optional: If you need to label nodes and edit label affinity in addons, see [Node Label Affinity Deployment Usage Guide](../service_management_cluster_configuration/node_label_affinity_deployment_guide.md).
- Optional: For custom configuration of containerd and kubelet, see [Binary File Customization Usage Guide](../service_management_cluster_configuration/binary_file_custom_configuration_guide.md).
- Optional: If you need to install custom extensions via addons, see [Addon Usage Guide](../service_management_cluster_configuration/using_add-ons.md).
