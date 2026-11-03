# Delete Bootstrap Cluster via Backend (CLI)

Uninstall the bootstrap cluster on the **bootstrap node**. Web delete is **not** supported.

## Prerequisites

- Account with `sudo` on the **bootstrap node**.
- Service and management clusters **must** already be uninstalled (or you accept losing the only unified uninstall path). See [Cluster Uninstallation Overview](./overview.md).

> ![](./figures/icon-danger.gif) **Danger:**<br />
> After bootstrap uninstall:
>
> - The bootstrap **management plane cannot be logged in**;
> - You **cannot** uninstall service/management clusters via UI or bootstrap `bc` anymore;
> - Remaining clusters require per-node manual cleanup.

## Procedure

1. Reset the bootstrap node:

   ```bash
   bke reset --all --mount
   ```

   > ![](./figures/icon-note.gif) **Note:**<br />
   > Without `--confirm`, `--all` prompts for confirmation. Use `bke reset --all --mount --confirm` to skip.

2. (Optional) Remove BKE media and the local binary:

   ```bash
   rm -rf /bke
   rm -f /usr/local/bin/bke
   ```

   > ![](./figures/icon-note.gif) **Note:**<br />
   > Step 2 only frees disk and removes the `bke` command. It is **not** required to consider the reset successful. Skip it if you want to keep media for a later `bke init`.

## What this deletes

| Step / command | Removes |
|:---|:---|
| `bke reset` (included in `--all`) | Local Kubernetes/k3s transition containers and dirs |
| `--all` | Container services, NTP, runtime, network, leftovers |
| `--mount` | Extract/mount dirs; with `--all`, BKE-managed system config / timezone as implemented |
| `rm -rf /bke` (optional) | Install/extract workspace |
| `rm -f /usr/local/bin/bke` (optional) | Local `bke` binary |

Does **not** auto-unbind HA VIPs (see [FAQ](../cluster_installation_guide/appendix/FAQ.md)).

## Success criteria

Whether step 2 ran does not affect judging that services were reset:

- `bke reset --all --mount` completed successfully.
- Bootstrap API is unreachable via `kubectl get pod -A`, or local k3s/Kubernetes processes and containers are gone.
- (If optional step done) `ls /bke` fails and `which bke` has no output.

## Follow-up

- Reinstall via [Online Bootstrap Installation](../cluster_installation_guide/bootstrap_cluster_installation/online_bootstrap_cluster_installation.md) or [Offline Bootstrap Installation](../cluster_installation_guide/bootstrap_cluster_installation/offline_bootstrap_cluster_installation.md).
- Parameters: [Command Reference](../cluster_installation_guide/appendix/command_reference.md), [bkeadm Configuration](../cluster_installation_guide/appendix/bkeadm_configuration.md).
