# Delete Management Cluster via Frontend (Web)

Uninstall a management cluster from the Web UI. Because the node agent bkeagent first listens on the bootstrap cluster and later switches to the management cluster, you must delete once on the **management-cluster UI** and once on the **bootstrap UI**. Both steps are required.

> ![](./figures/icon-note.gif) **Note:**<br />
> The management-cluster UI (`:31616`) runs on the management cluster and becomes unavailable as that cluster is uninstalled. After you start delete in step 1, you may no longer be able to sign in there—that is expected. Final Web confirmation is on the bootstrap UI (`:30010`).

## Prerequisites

- Before starting, you must be able to sign in to both the **management cluster** and **bootstrap** openFuyao UIs.
- Service clusters under the management cluster should already be uninstalled or migrated.
- Read [Cluster Uninstallation Overview](./overview.md).

## Procedure

1. Sign in to the management-cluster Web and delete the cluster.

   1.1 Open `https://<management-cluster-IP>:31616` and sign in to the management-cluster management interface.
   1.2 Select **Cluster Lifecycle Management** in the left navigation pane.
   1.3 Locate the **management cluster** to be deleted, and choose **Delete/Uninstall** in the operation column.
   1.4 Confirm the cluster name and confirm the deletion based on the risk prompt. After deletion starts, this management interface may become interrupted or inaccessible. There is no need to rely on this management interface to confirm the cluster list status.

2. Sign in to the bootstrap Web and delete the cluster.

   2.1 Open `https://<bootstrap-IP>:30010` and sign in to the bootstrap management interface.
   2.2 Select **Cluster Lifecycle Management** in the left navigation pane.
   2.3 Locate the entry corresponding to the same management cluster, and choose **Delete/Uninstall**.
   2.4 Confirm the name and wait until the cluster entry is removed from the cluster list.

> ![](./figures/icon-note.gif) **Note:**<br />
> - Both steps are required; doing only one side leaves leftovers on the other.
> - Display names on the two UIs may differ slightly; match the name used at create time.
> - Complete step 1 first; if `:31616` is already unreachable afterward, continue with step 2.

## Success criteria

- Management-cluster UI (`:31616`) can no longer be signed in or opened.
- Gone from the bootstrap UI (`:30010`) **Cluster Lifecycle Management** list.

## Follow-up

- Other uninstall paths and delete order: [Cluster Uninstallation Overview](./overview.md).
- Bootstrap uninstall must be last; see the bootstrap section in the overview.
