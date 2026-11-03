/**
 * 管理集群自管 BC 在 UI 上的显示名（sig-installation#224）。
 *
 * 合入 #224 后自管 BC 继承引导侧创建名，不再固定为 `bke-cluster`。
 * - 前端用例创建时 YAML `metadata.name` 已是完整资源名（常带 `bke-` 前缀）→ 原样返回。
 * - 环境变量给出逻辑名（无 `bke-`）→ 与后端 `ClusterName` 一致补前缀。
 */
export function managementClusterSelfDisplayName(bootstrapClusterName: string): string {
    const name = bootstrapClusterName.trim();
    if (!name) {
        throw new Error("bootstrapClusterName is required for self-managed BC UI display name (#224)");
    }
    return name.startsWith("bke-") ? name : `bke-${name}`;
}

/**
 * 从环境变量解析已有管理集群自管显示名。
 * 优先 `MGMT_CLUSTER_SELF_NAME`，其次 `MGMT_CLUSTER_NAME`；均未设置时回退 `cluster` → `bke-cluster`（兼容旧环境）。
 */
export function resolveManagementClusterSelfDisplayNameFromEnv(): string {
    const fromEnv = (
        process.env.MGMT_CLUSTER_SELF_NAME ||
        process.env.MGMT_CLUSTER_NAME ||
        "cluster"
    ).trim();
    return managementClusterSelfDisplayName(fromEnv);
}
