# Delete Service Cluster via Frontend (Web)

Delete a service cluster from openFuyao **Cluster Lifecycle Management**. Choose the management-plane URL based on **where the service cluster was created**, see [Table 1](#table12452).

## Prerequisites

- Confirm the target is a **service cluster** and workloads are migrated or deletion is acceptable.
- Read [Cluster Uninstallation Overview](./overview.md).

## Choose the management plane by deploy entry

**Table 1** Relationship between service cluster deployment method and management plane<span id="table12452"></span>

| How the service cluster was deployed | Management plane to use | URL |
|:---|:---|:---|
| Created from the **bootstrap** management plane | Bootstrap openFuyao UI | `https://<bootstrap-IP>:30010` |
| Created from the **management cluster** management plane | Management-cluster openFuyao UI | `https://<management-cluster-IP>:31616` |

Menu: **Cluster Lifecycle Management**.

> ![](./figures/icon-note.gif) **Note:**<br />
> Use the same management plane that created the service cluster.

## Procedure

1. Sign in to the matching openFuyao UI from the table above.
2. Open **Cluster Lifecycle Management**.
3. Locate the service cluster and choose delete/uninstall.
4. Confirm the cluster name.
5. Wait until it disappears from the list.

## Success criteria

- The service cluster is gone from the **Cluster Lifecycle Management** list.

## Follow-up

- Scaling: [Cluster Scaling](../cluster_scaling).
- Other uninstall paths and delete order: [Cluster Uninstallation Overview](./overview.md).
