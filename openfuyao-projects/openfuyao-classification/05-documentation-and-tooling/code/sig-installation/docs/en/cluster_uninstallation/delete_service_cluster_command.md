# Delete Service Cluster via Backend (CLI)

Uninstall a service cluster by editing `BKECluster` (`bc`) on the bootstrap or management cluster, matching controller behavior.

## Prerequisites

- Operate with an account that has `sudo` privileges and `kubectl` access to the target API.
- Sign in to the node that matches how the service cluster was deployed:
  - **Deployed from the bootstrap cluster**: log in to the **bootstrap node**; kubeconfig points to the bootstrap cluster.
  - **Deployed from the management cluster**: log in to a **management-cluster control-plane node**; kubeconfig points to the management cluster.
- Read [Cluster Uninstallation Overview](./overview.md).

## Procedure

1. List clusters:

   ```bash
   kubectl get bc -A
   ```

2. Edit the target (replace placeholders):

   ```bash
   kubectl edit bc -n <namespace> <name>
   ```

3. Apply (required for **cleaning target nodes**):

   ```yaml
   metadata:
     annotations:
       # Required as "false" to reset/clean target service nodes; default "true" only removes API objects
       bke.bocloud.com/ignore-target-cluster-delete: "false"
   spec:
     reset: true   # Trigger deletion, equivalent to directly deleting the BKECluster resource
   ```

   Optional related annotations:

   | Annotation | Recommended (full uninstall) | Description |
   |:---|:---|:---|
   | `bke.bocloud.com/deep-restore-node` | `"true"` (default) | Deep-clean the node, including container runtimes |
   | `bke.bocloud.com/ignore-namespace-delete` | As needed | Deletes the namespace only when set to `"false"` and no other `bc` remains in that namespace |

4. Watch progress:

   ```bash
   kubectl get bc -n <namespace> <name> -o yaml
   kubectl describe bc -n <namespace> <name>
   ```

## What this deletes

- The service cluster's `BKECluster` / related `BKENode` objects.
- With `ignore-target-cluster-delete: "false"`: Kubernetes components and addons on target nodes; runtime cleanup follows `deep-restore-node`.
- With default `"true"`: API objects only; nodes may retain the cluster.

## Success criteria

- `kubectl get bc -A` no longer lists the service cluster.

## Follow-up

- Annotations: [cluster-api-provider-bke Configuration](../cluster_installation_guide/appendix/cluster-api-provider-bke_configuration.md).
- Other uninstall paths and delete order: [Cluster Uninstallation Overview](./overview.md).
