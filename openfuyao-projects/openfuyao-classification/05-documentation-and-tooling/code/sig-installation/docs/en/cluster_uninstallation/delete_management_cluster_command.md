# Delete Management Cluster via Backend (CLI)

Uninstall a management cluster. During installation, the node agent bkeagent **first listens to the bootstrap cluster**, then **switches to listening on the management cluster itself**. Therefore related `BKECluster` (`bc`) objects exist in both places:

1. **Bootstrap cluster** (objects created at install time)
2. **Management cluster itself** (reconcile entry after the listener switch)

For a full uninstall, **both** `bc` objects must be handled.

## Prerequisites

- Operate with an account that has `sudo` privileges and `kubectl` access to the target APIs.
- Access to both the **bootstrap node** and a **management control-plane node**.
- Service clusters under the management cluster should already be uninstalled.
- Read [Cluster Uninstallation Overview](./overview.md).

## Procedure

### Step 1: Delete the local `bc` on the management cluster (clean target nodes)

After the listener has switched, node reset/cleanup is driven by the `bc` on the **management cluster**. On a management control-plane node:

```bash
# kubeconfig points to the management cluster
kubectl get bc -A
kubectl edit bc -n <namespace> <name>
```

For full node cleanup:

```yaml
metadata:
  annotations:
    bke.bocloud.com/ignore-target-cluster-delete: "false"
spec:
  reset: true
```

Watch progress briefly if needed. As uninstall proceeds the management-cluster API often becomes unavailable—that is expected. Do not use `kubectl get bc` on the management cluster as the success check; continue with step 2.

### Step 2: Delete the corresponding `bc` on the bootstrap cluster

On the **bootstrap node** (kubeconfig for the bootstrap cluster), remove the matching management-cluster `bc` to clear leftover objects on the bootstrap side:

```bash
# kubeconfig points to the bootstrap cluster
kubectl get bc -A
kubectl edit bc -n <namespace> <name>
```

If nodes were already cleaned in step 1, you may keep `ignore-target-cluster-delete` at the default `"true"` and only set `spec.reset: true` (avoid repeating reset on an already cleaned environment):

```yaml
metadata:
  annotations:
    bke.bocloud.com/ignore-target-cluster-delete: "true"   # Can keep the default if nodes have been cleaned in step 1
spec:
  reset: true
```

Or delete directly:

```bash
kubectl delete bc -n <namespace> <name>
```

> ![](./figures/icon-note.gif) **Note:**<br />
> - Namespace/name on each cluster may differ; always check `kubectl get bc -A` on that cluster.
> - If the listener has not switched yet and reconcile is still on bootstrap only, you can first delete from bootstrap with `ignore-target-cluster-delete: "false"` and `spec.reset: true`; after switchover, still confirm and clean any local `bc` on the management cluster.

## What this deletes

| Location | Effect |
|:---|:---|
| Management-cluster `bc` (step 1) | Drives uninstall/reset of management nodes after listener switch (`ignore-target-cluster-delete: "false"` required for node cleanup) |
| Bootstrap `bc` (step 2) | Removes the corresponding CR on the bootstrap cluster to avoid leftovers |

## Success criteria

- Former management-cluster API / management plane is unavailable.
- On the **bootstrap node**: `kubectl get bc -A` no longer lists that management cluster.

## Follow-up

- Annotations / `bkeagent-listener`: [cluster-api-provider-bke Configuration](../cluster_installation_guide/appendix/cluster-api-provider-bke_configuration.md).
- Other uninstall paths and delete order: [Cluster Uninstallation Overview](./overview.md).
