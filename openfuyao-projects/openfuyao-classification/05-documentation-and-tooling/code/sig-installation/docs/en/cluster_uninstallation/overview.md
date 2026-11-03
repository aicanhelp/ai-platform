# Cluster Uninstallation Overview

This page is the single entry for openFuyao cluster uninstallation. It covers **delete order**, **impact**, **where to run commands**, and **what each command removes**. Detailed steps are in the linked guides below.

> ![](./figures/icon-danger.gif) **Danger:**<br />
> Uninstall and delete are high-risk operations. Verify the target cluster, namespace, and resource names, and back up as needed. After a mistaken delete you may be unable to manage or uninstall other clusters from the management plane.

## Recommended delete order

Uninstall in this order. **Do not** remove the bootstrap cluster while service/management clusters still depend on it:

```text
Service cluster → Management cluster (if any) → Bootstrap cluster
```

**Table 1** Recommended delete order

| Order | Target | Why |
|:---|:---|:---|
| 1 | Service cluster | Deletion uses `BKECluster` (`bc`) and the management plane on the bootstrap or management cluster |
| 2 | Management cluster | Finish before bootstrap uninstall when a management cluster exists; handle `bc` on **both** the bootstrap cluster and the management cluster itself (after install, the listener switches from bootstrap to the management cluster) |
| 3 | Bootstrap cluster | Last step; the bootstrap management plane becomes unavailable afterward |

> ![](./figures/icon-caution.gif) **Caution:**<br />
> **Do not** delete the bootstrap cluster before service/management clusters are uninstalled. After bootstrap uninstall:
>
> - The openFuyao **bootstrap management plane cannot be logged in**;
> - You can no longer uninstall service/management clusters via the UI or `bc` on the bootstrap node;
> - Leftover nodes must be cleaned manually on each host.

## Where to operate

**Table 2** Where to operate and entry points

| Target | Web | CLI |
|:---|:---|:---|
| Service cluster | **Bootstrap-deployed**: `https://<bootstrap-IP>:30010`; **management-cluster-deployed**: `https://<management-cluster-IP>:31616` (**Cluster Lifecycle Management**) | On the cluster that hosts the management plane that created the service cluster, edit/delete the target `bc` |
| Management cluster | **Two Web steps**: (1) delete on management-cluster UI `https://<management-cluster-IP>:31616`; (2) delete on bootstrap UI `https://<bootstrap-IP>:30010` | **First** delete the local `bc` on the management cluster, **then** delete the corresponding `bc` on the bootstrap node |
| Bootstrap cluster | **Not supported** via Web | Only on the **bootstrap node** with `bke reset` |

## What gets deleted

### 1. Service / management cluster (via `BKECluster`)

Edit the target `bc` on the bootstrap or management cluster; configuration details are as follows:

**Table 3** BKECluster uninstall configuration

| Setting | Meaning | Effect |
|:---|:---|:---|
| `bke.bocloud.com/ignore-target-cluster-delete` | Whether to skip cleaning **target nodes** | Default `"true"`: remove API/`bc` objects only. Set `"false"` to reset/clean target nodes |
| `spec.reset: true` | Trigger delete | Triggers the deletion reconcile flow for the `bc` |
| `bke.bocloud.com/deep-restore-node` | Deep restore | Default `"true"`: also clean container runtimes |
| `bke.bocloud.com/ignore-namespace-delete` | Keep `bc` namespace | Default `"true"`; delete namespace only when `"false"` and no other `bc` remains |

For a **full** target uninstall, set `ignore-target-cluster-delete` to `"false"` and `spec.reset: true`. When only deleting API/`bc` objects without cleaning nodes, you can keep the default annotation as `"true"`.

> ![](./figures/icon-note.gif) **Note:**<br />
> For a **management cluster**, the node agent bkeagent first listens on the bootstrap cluster and later switches to the management cluster itself. Uninstall must handle **both** sides (Web: delete on both UIs; CLI: delete `bc` on both clusters); see the dedicated guides.

During uninstall, the controller cleans addons, Kubernetes components, etcd, and related resources in **reverse dependency order**. If cleanup fails, the `bc` may enter a terminating-failed state; use `kubectl describe bc` to troubleshoot.

### 2. Bootstrap cluster (`bke reset`)

**Table 4** bke reset command cleanup scope

| Command | Removes |
|:---|:---|
| `bke reset` | Local Kubernetes/k3s transition cluster containers and directories |
| `bke reset --mount` | Extract/mount dirs; also container service, NTP, yum/repo related config when used alone |
| `bke reset --all` | Broader cleanup of container services, NTP, runtime, network, leftovers (confirmation unless `--confirm`) |
| `bke reset --all --mount` | Combination of `--all` and `--mount`, including BKE-managed system config / timezone restore as implemented |

> ![](./figures/icon-note.gif) **Note:**<br />
> - `bke reset` does **not** unbind leftover HA VIPs; if a VIP remains on a node, unbind it manually with `ip addr del`, see [FAQ](../cluster_installation_guide/appendix/FAQ.md).
> - `rm -rf /bke` and `rm -f /usr/local/bin/bke` are **optional**, not required for a successful reset. They remove the install media directory and the local `bke` binary. Reset can still be considered successful without them, but you may need to re-download tools and media before the next install.

See [Command Reference](../cluster_installation_guide/appendix/command_reference.md) and [bkeadm Configuration](../cluster_installation_guide/appendix/bkeadm_configuration.md).

## Success criteria

### Frontend (Web)

- Gone from the **Cluster Lifecycle Management** list.
- For a **management cluster**: the management-cluster UI (`:31616`) can no longer be signed in or opened; the bootstrap UI (`:30010`) list shows no matching entry.

### Backend (CLI)

- Service cluster: `kubectl get bc -A` on the creating-side cluster no longer shows the `BKECluster`.
- Management cluster: former management-cluster API / management plane is unavailable; on the **bootstrap node**, `kubectl get bc -A` no longer lists that management cluster.
- Bootstrap: `bke reset --all --mount` finishes successfully; local k3s/Kubernetes processes and containers are gone; (after optional cleanup) `/bke` and the local `bke` binary are gone.

## Navigation

**Table 5** Cluster uninstallation documentation navigation

| Scenario | Doc |
|:---|:---|
| Delete service cluster (Web) | [Delete Service Cluster via Frontend (Web)](./delete_service_cluster_web.md) |
| Delete service cluster (CLI) | [Delete Service Cluster via Backend (CLI)](./delete_service_cluster_command.md) |
| Delete management cluster (Web) | [Delete Management Cluster via Frontend (Web)](./delete_management_cluster_web.md) |
| Delete management cluster (CLI) | [Delete Management Cluster via Backend (CLI)](./delete_management_cluster_command.md) |
| Delete bootstrap cluster (CLI) | [Delete Bootstrap Cluster via Backend (CLI)](./delete_bootstrap_cluster_command.md) |
