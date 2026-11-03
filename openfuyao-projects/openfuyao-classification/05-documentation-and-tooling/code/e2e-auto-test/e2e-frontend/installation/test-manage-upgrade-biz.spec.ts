import type { Page } from "@playwright/test";
import { expect, test } from "@playwright/test";
import { nodeInfo, timeouts, urls } from "./constants/common";
import {login} from "./utils/rbac";
import {
    verifyClusterHealthOrInstallFailed,
    verifyClusterHealthy, verifyClusterHealthy2,
    verifyClusterUpgrading,
    verifyClusterDeleted,
} from "./utils/verify";
import {
    deleteCluster,
    startClusterUpgrade,
} from "./utils/clusterManage";
import {withSshConnection} from "@/installation/utils/upgrade-pre";

const { guideNodeInfo } = nodeInfo;
const { totalTimeout } = timeouts;

function requireEnv(name: string): string {
    const value = process.env[name];
    if (!value) {
        throw new Error(`${name} is required`);
    }
    return value;
}

function getManagerUrl(): string {
    return `https://${requireEnv("TEST_NODE1_IP")}:31616`;
}

function shellQuote(value: string | number): string {
    return `'${String(value).replace(/'/g, `'"'"'`)}'`;
}

function yamlSingleQuote(value: string | number): string {
    return `'${String(value).replace(/'/g, "''")}'`;
}

function escapeRegExp(value: string): string {
    return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
async function verifyClusterInstallStartedOrSettled(page: Page, clusterName: string) {
    const statusPattern = new RegExp(`${escapeRegExp(clusterName)}.*(安装中|未知|健康|安装失败)`);
    await expect(page.getByRole('row', { name: statusPattern })).toBeVisible({ timeout: timeouts.validateTimeout });
}

async function createManageClusterBySsh(clusterName: string) {
    const guideNodeHost = requireEnv("TEST_NODE1_IP");
    const guideNodePassword = requireEnv("TEST_NODE1_PASSWORD");
    const masterNodeIp = requireEnv("TEST_NODE2_IP");
    const masterNodePassword = requireEnv("TEST_NODE2_PASSWORD");
    const workerNodeIp = requireEnv("TEST_NODE3_IP");
    const workerNodePassword = requireEnv("TEST_NODE3_PASSWORD");
    const guideNodeUsername = "root";

    return withSshConnection(
        {
            host: guideNodeHost,
            username: guideNodeUsername,
            password: guideNodePassword,
            port: Number(guideNodeInfo.port),
            readyTimeout: 15000,
        },
        async (ssh) => {
            const commands = [
                {
                    command: "mkdir -p /opt/muw",
                    errorMessage: "Failed to create manage config directory",
                },
                {
                    command: "cp /bke/cluster/1master1node-nodes.yaml /opt/muw/1master1worker-biz-nodes.yaml",
                    errorMessage: "Failed to copy node config",
                },
                {
                    command: `CLUSTER_NAME=${shellQuote(clusterName)} yq -i '.metadata.namespace = strenv(CLUSTER_NAME) | .metadata.labels."cluster.x-k8s.io/cluster-name" = strenv(CLUSTER_NAME)' /opt/muw/1master1worker-biz-nodes.yaml`,
                    errorMessage: "Failed to update node metadata",
                },
                {
                    command: `NODE_IP=${shellQuote(masterNodeIp)} NODE_PASSWORD=${shellQuote(masterNodePassword)} yq -i 'with(select(documentIndex == 0); .spec.ip = strenv(NODE_IP) | .spec.password = strenv(NODE_PASSWORD) | .spec.username = "root")' /opt/muw/1master1worker-biz-nodes.yaml`,
                    errorMessage: "Failed to update first node spec",
                },
                {
                    command: `NODE_IP=${shellQuote(workerNodeIp)} NODE_PASSWORD=${shellQuote(workerNodePassword)} yq -i 'with(select(documentIndex == 1); .spec.ip = strenv(NODE_IP) | .spec.password = strenv(NODE_PASSWORD) | .spec.username = "root")' /opt/muw/1master1worker-biz-nodes.yaml`,
                    errorMessage: "Failed to update second node spec",
                },
                {
                    command: "cp /bke/cluster/1master1node-cluster.yaml /opt/muw/1master1worker-biz-cluster.yaml",
                    errorMessage: "Failed to copy cluster config",
                },
                {
                    command: `CLUSTER_NAME=${shellQuote(clusterName)} yq -i '.metadata.name = strenv(CLUSTER_NAME)' /opt/muw/1master1worker-biz-cluster.yaml`,
                    errorMessage: "Failed to update cluster name",
                },
                {
                    command: `CLUSTER_NAMESPACE=${shellQuote(clusterName)} yq -i '.metadata.namespace = strenv(CLUSTER_NAMESPACE)' /opt/muw/1master1worker-biz-cluster.yaml`,
                    errorMessage: "Failed to update cluster namespace",
                },
                {
                    command: `yq -i '.spec.controlPlaneEndpoint = {}' /opt/muw/1master1worker-biz-cluster.yaml`,
                    errorMessage: "Failed to reset controlPlaneEndpoint",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and has("etcdVersion")).etcdVersion) = "v3.6.7-of.0"' /opt/muw/1master1worker-biz-cluster.yaml`,
                    errorMessage: "Failed to update etcdVersion",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and has("kubernetesVersion")).kubernetesVersion) = "v1.33.1-of.2"' /opt/muw/1master1worker-biz-cluster.yaml`,
                    errorMessage: "Failed to update kubernetesVersion",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and .name == "coredns").version) = "v1.12.2-of.1"' /opt/muw/1master1worker-biz-cluster.yaml`,
                    errorMessage: "Failed to update coredns version",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and .name == "kubeproxy").version) = "v1.33.1-of.2"' /opt/muw/1master1worker-biz-cluster.yaml`,
                    errorMessage: "Failed to update kubeproxy version",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and .name == "calico").version) = "v3.31.3"' /opt/muw/1master1worker-biz-cluster.yaml`,
                    errorMessage: "Failed to update calico version",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and has("containerdVersion")).containerdVersion) = "v2.1.1"' /opt/muw/1master1worker-biz-cluster.yaml`,
                    errorMessage: "Failed to update containerdVersion",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!seq")) |= map(select((.name // "") != "cluster-api" and (.name // "") != "openfuyao-system-controller"))' /opt/muw/1master1worker-biz-cluster.yaml`,
                    errorMessage: "Failed to remove addons from cluster config",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and has("openFuyaoVersion")).openFuyaoVersion) = "v26.05"' /opt/muw/1master1worker-biz-cluster.yaml`,
                    errorMessage: "Failed to update openFuyaoVersion",
                },
                {
                    command: "bke cluster create -f /opt/muw/1master1worker-biz-cluster.yaml -n /opt/muw/1master1worker-biz-nodes.yaml",
                    errorMessage: "Failed to create cluster by bke",
                },
            ];

            for (const { command, errorMessage } of commands) {
                const result = await ssh.execCommand(command);
                if (result.code !== 0) {
                    throw new Error(`${errorMessage}: ${result.stderr || result.stdout}`);
                }
            }
        },
    );
}

async function createManageCluster3M1WBySsh(clusterName: string) {
    const guideNodeHost = requireEnv("TEST_NODE1_IP");
    const guideNodePassword = requireEnv("TEST_NODE1_PASSWORD");
    const master1Ip = requireEnv("TEST_NODE2_IP");
    const master1Password = requireEnv("TEST_NODE2_PASSWORD");
    const master2Ip = requireEnv("TEST_NODE3_IP");
    const master2Password = requireEnv("TEST_NODE3_PASSWORD");
    const master3Ip = requireEnv("TEST_NODE4_IP");
    const master3Password = requireEnv("TEST_NODE4_PASSWORD");
    const workerIp = requireEnv("TEST_NODE5_IP");
    const workerPassword = requireEnv("TEST_NODE5_PASSWORD");
    const vipHost = process.env.MGMT_CLUSTER_VIP_HOST || process.env.MANAGEMENT_CLUSTER_VIP_HOST || "";
    const vipPort = process.env.MGMT_CLUSTER_VIP_PORT || process.env.MANAGEMENT_CLUSTER_VIP_PORT || "";
    const guideNodeUsername = "root";

    return withSshConnection(
        {
            host: guideNodeHost,
            username: guideNodeUsername,
            password: guideNodePassword,
            port: Number(guideNodeInfo.port),
            readyTimeout: 15000,
        },
        async (ssh) => {
            const appendWorkerNodeCommand = `cat >> /opt/muw/3master1worker-biz-nodes.yaml <<'EOF'
---
apiVersion: bke.bocloud.com/v1beta1
kind: BKENode
metadata:
  creationTimestamp: null
  labels:
    cluster.x-k8s.io/cluster-name: ${clusterName}
  name: bke-cluster-worker-1
  namespace: ${clusterName}
spec:
  hostname: worker-1
  ip: ${workerIp}
  password: ${yamlSingleQuote(workerPassword)}
  port: "22"
  role:
    - node
  username: root
status: {}
EOF`;

            const commands = [
                {
                    command: "mkdir -p /opt/muw",
                    errorMessage: "Failed to create manage config directory",
                },
                {
                    command: "cp /bke/cluster/1master1node-cluster.yaml /opt/muw/3master1worker-biz-cluster.yaml",
                    errorMessage: "Failed to copy management cluster config",
                },
                {
                    command: "cp /bke/cluster/3master-nodes.yaml /opt/muw/3master1worker-biz-nodes.yaml",
                    errorMessage: "Failed to copy 3master node config",
                },
                {
                    command: `CLUSTER_NAME=${shellQuote(clusterName)} yq -i '.metadata.name = strenv(CLUSTER_NAME) | .metadata.namespace = strenv(CLUSTER_NAME)' /opt/muw/3master1worker-biz-cluster.yaml`,
                    errorMessage: "Failed to update management cluster metadata",
                },
                {
                    command: `CLUSTER_NAME=${shellQuote(clusterName)} yq -i '.metadata.namespace = strenv(CLUSTER_NAME) | .metadata.labels."cluster.x-k8s.io/cluster-name" = strenv(CLUSTER_NAME)' /opt/muw/3master1worker-biz-nodes.yaml`,
                    errorMessage: "Failed to update management node metadata",
                },
                {
                    command: `NODE_IP=${shellQuote(master1Ip)} NODE_PASSWORD=${shellQuote(master1Password)} yq -i 'with(select(documentIndex == 0); .spec.ip = strenv(NODE_IP) | .spec.password = strenv(NODE_PASSWORD) | .spec.username = "root")' /opt/muw/3master1worker-biz-nodes.yaml`,
                    errorMessage: "Failed to update first master node spec",
                },
                {
                    command: `NODE_IP=${shellQuote(master2Ip)} NODE_PASSWORD=${shellQuote(master2Password)} yq -i 'with(select(documentIndex == 1); .spec.ip = strenv(NODE_IP) | .spec.password = strenv(NODE_PASSWORD) | .spec.username = "root")' /opt/muw/3master1worker-biz-nodes.yaml`,
                    errorMessage: "Failed to update second master node spec",
                },
                {
                    command: `NODE_IP=${shellQuote(master3Ip)} NODE_PASSWORD=${shellQuote(master3Password)} yq -i 'with(select(documentIndex == 2); .spec.ip = strenv(NODE_IP) | .spec.password = strenv(NODE_PASSWORD) | .spec.username = "root")' /opt/muw/3master1worker-biz-nodes.yaml`,
                    errorMessage: "Failed to update third master node spec",
                },
                {
                    command: appendWorkerNodeCommand,
                    errorMessage: "Failed to append worker node config",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and has("etcdVersion")).etcdVersion) = "v3.6.7-of.0"' /opt/muw/3master1worker-biz-cluster.yaml`,
                    errorMessage: "Failed to update etcdVersion",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and has("kubernetesVersion")).kubernetesVersion) = "v1.33.1-of.2"' /opt/muw/3master1worker-biz-cluster.yaml`,
                    errorMessage: "Failed to update kubernetesVersion",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and .name == "coredns").version) = "v1.12.2-of.1"' /opt/muw/3master1worker-biz-cluster.yaml`,
                    errorMessage: "Failed to update coredns version",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and .name == "kubeproxy").version) = "v1.33.1-of.2"' /opt/muw/3master1worker-biz-cluster.yaml`,
                    errorMessage: "Failed to update kubeproxy version",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and .name == "calico").version) = "v3.31.3"' /opt/muw/3master1worker-biz-cluster.yaml`,
                    errorMessage: "Failed to update calico version",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and has("containerdVersion")).containerdVersion) = "v2.1.1"' /opt/muw/3master1worker-biz-cluster.yaml`,
                    errorMessage: "Failed to update containerdVersion",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!seq")) |= map(select((.name // "") != "cluster-api" and (.name // "") != "openfuyao-system-controller"))' /opt/muw/3master1worker-biz-cluster.yaml`,
                    errorMessage: "Failed to remove addons from cluster config",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and has("openFuyaoVersion")).openFuyaoVersion) = "v26.05"' /opt/muw/3master1worker-biz-cluster.yaml`,
                    errorMessage: "Failed to update openFuyaoVersion",
                },
            ];

            if (vipHost && vipPort) {
                commands.splice(5, 0, {
                    command: `VIP_HOST=${shellQuote(vipHost)} VIP_PORT=${shellQuote(vipPort)} yq -i '.spec.controlPlaneEndpoint.host = strenv(VIP_HOST) | .spec.controlPlaneEndpoint.port = env(VIP_PORT)' /opt/muw/3master1worker-biz-cluster.yaml`,
                    errorMessage: "Failed to update controlPlaneEndpoint",
                });
            }

            commands.push({
                command: "bke cluster create -f /opt/muw/3master1worker-biz-cluster.yaml -n /opt/muw/3master1worker-biz-nodes.yaml",
                errorMessage: "Failed to create 3master1worker management cluster by bke",
            });

            for (const { command, errorMessage } of commands) {
                const result = await ssh.execCommand(command);
                if (result.code !== 0) {
                    throw new Error(`${errorMessage}: ${result.stderr || result.stdout}`);
                }
            }
        },
    );
}

async function loginManagementCluster(page: Page) {
    await page.goto(getManagerUrl() + urls.loginUrl);
    await login(page, process.env.TEST_USERNAME, process.env.TEST_PASSWORD);
}

const TEST_CLUSTER_NAME_1M1W = 'bke-cluster-biz-upgrade-1m1w';
const TEST_CLUSTER_NAME_3M1W = 'bke-cluster-biz-upgrade-3m1w';

test.describe('管理集群升级业务集群', { tag: ["@post-init", "@with-management-cluster","@custom-repo"] },  () => {
    test.describe.configure({ mode: 'serial' });


    test("从管理集群创建1master1worker业务集群,从管理集群页面点升级", async ({ page }) => {
        test.setTimeout(totalTimeout * 2);
        const clusterName = TEST_CLUSTER_NAME_1M1W;
        await test.step("创建1master1worker业务集群", async () => {
            await loginManagementCluster(page);
            await page.getByRole('link', { name: '集群生命周期管理' }).click();
            await createManageClusterBySsh(clusterName);
            await verifyClusterInstallStartedOrSettled(page, clusterName);
            await verifyClusterHealthy2(page, clusterName);
            await expect(page.getByRole('cell', { name: 'v26.05' }).first()).toBeVisible();
        });

        await test.step("集群升级", async () => {
            const upgradeVersion = "v26.06";
            await startClusterUpgrade(page, clusterName, upgradeVersion);
            await verifyClusterUpgrading(page, clusterName);
            await verifyClusterHealthy(page, clusterName);
            await expect(page.getByRole('cell', { name: 'v26.06' }).first()).toBeVisible();
            await expect(page.getByRole('cell', { name: '1.34.3-of.1' }).first()).toBeVisible();
        });

        await test.step("删除业务集群并清理引导页面记录", async () => {
            await verifyClusterHealthOrInstallFailed(page, TEST_CLUSTER_NAME_1M1W);
            await deleteCluster(page, TEST_CLUSTER_NAME_1M1W);
            await verifyClusterDeleted(page, TEST_CLUSTER_NAME_1M1W);
        });

    });

    test("从管理集群创建3master1worker业务集群,从管理集群页面点升级",  async ({ page }) => {
        test.setTimeout(totalTimeout * 2);
        const clusterName = TEST_CLUSTER_NAME_3M1W;

        await test.step("创建3master1worker管理集群", async () => {
            await loginManagementCluster(page);
            await page.getByRole('link', { name: '集群生命周期管理' }).click();
            await createManageCluster3M1WBySsh(clusterName);
            await verifyClusterInstallStartedOrSettled(page, clusterName);
            await verifyClusterHealthy2(page, clusterName);
            await expect(page.getByRole('cell', { name: 'v26.05' }).first()).toBeVisible();
        });

        await test.step("集群升级", async () => {
            const upgradeVersion = "v26.06";
            await startClusterUpgrade(page, clusterName, upgradeVersion);
            await verifyClusterUpgrading(page, clusterName);
            await verifyClusterHealthy(page, clusterName);
            await expect(page.getByRole('cell', { name: 'v26.06' }).first()).toBeVisible();
            await expect(page.getByRole('cell', { name: '1.34.3-of.1' }).first()).toBeVisible();
        });

        await test.step("删除业务集群并清理引导页面记录", async () => {
            await verifyClusterHealthOrInstallFailed(page, TEST_CLUSTER_NAME_3M1W);
            await deleteCluster(page, TEST_CLUSTER_NAME_3M1W);
            await verifyClusterDeleted(page, TEST_CLUSTER_NAME_3M1W);
        });
    });

});
