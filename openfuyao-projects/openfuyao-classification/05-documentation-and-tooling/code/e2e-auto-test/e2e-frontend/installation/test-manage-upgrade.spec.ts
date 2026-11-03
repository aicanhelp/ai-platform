import type { Page } from "@playwright/test";
import { expect, test } from "@playwright/test";
import { addonOptions, nodeInfo, timeouts, urls } from "./constants/common";
import {firstLoginChangePassword, login} from "./utils/rbac";
import {
    verifyClusterDeleted,
    verifyClusterHealthOrInstallFailed,
    verifyClusterUpgradeCompletedStable,
    verifyClusterHealthy, verifyClusterHealthy2,
    verifyClusterInstalling,
    verifyClusterUpgrading,
    waitForClusterUpgradeFinishedAfterStarted,
    waitForClusterUpgradeStarted,
} from "./utils/verify";
import type { NodeInfoFields } from "./utils/clusterManage";
import {
    createCluster,
    deleteCluster,
    goToCreateClusterPage,
    setAdvancedConfig,
    setClusterInfo,
    setNodeInfo,
    startClusterUpgrade,
} from "./utils/clusterManage";
import {withSshConnection} from "@/installation/utils/upgrade-pre";
import { managementClusterSelfDisplayName } from "./utils/managementClusterName";

const { node1Info, guideNodeInfo } = nodeInfo;
const { totalTimeout } = timeouts;

// 管理节点地址
const managerUrl = process.env.TEST_MANAGERURL || '';

function requireEnv(name: string): string {
    const value = process.env[name];
    if (!value) {
        throw new Error(`${name} is required`);
    }
    return value;
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
                    command: "mkdir -p /opt/manage",
                    errorMessage: "Failed to create manage config directory",
                },
                {
                    command: "cp /bke/cluster/1master1node-nodes.yaml /opt/manage/1master1worker-mgmt-nodes.yaml",
                    errorMessage: "Failed to copy node config",
                },
                {
                    command: `CLUSTER_NAME=${shellQuote(clusterName)} yq -i '.metadata.namespace = strenv(CLUSTER_NAME) | .metadata.labels."cluster.x-k8s.io/cluster-name" = strenv(CLUSTER_NAME)' /opt/manage/1master1worker-mgmt-nodes.yaml`,
                    errorMessage: "Failed to update node metadata",
                },
                {
                    command: `NODE_IP=${shellQuote(masterNodeIp)} NODE_PASSWORD=${shellQuote(masterNodePassword)} yq -i 'with(select(documentIndex == 0); .spec.ip = strenv(NODE_IP) | .spec.password = strenv(NODE_PASSWORD) | .spec.username = "root")' /opt/manage/1master1worker-mgmt-nodes.yaml`,
                    errorMessage: "Failed to update first node spec",
                },
                {
                    command: `NODE_IP=${shellQuote(workerNodeIp)} NODE_PASSWORD=${shellQuote(workerNodePassword)} yq -i 'with(select(documentIndex == 1); .spec.ip = strenv(NODE_IP) | .spec.password = strenv(NODE_PASSWORD) | .spec.username = "root")' /opt/manage/1master1worker-mgmt-nodes.yaml`,
                    errorMessage: "Failed to update second node spec",
                },
                {
                    command: "cp /bke/cluster/1master1node-cluster.yaml /opt/manage/1master1worker-mgmt-cluster.yaml",
                    errorMessage: "Failed to copy cluster config",
                },
                {
                    command: `CLUSTER_NAME=${shellQuote(clusterName)} yq -i '.metadata.name = strenv(CLUSTER_NAME)' /opt/manage/1master1worker-mgmt-cluster.yaml`,
                    errorMessage: "Failed to update cluster name",
                },
                {
                    command: `CLUSTER_NAMESPACE=${shellQuote(clusterName)} yq -i '.metadata.namespace = strenv(CLUSTER_NAMESPACE)' /opt/manage/1master1worker-mgmt-cluster.yaml`,
                    errorMessage: "Failed to update cluster namespace",
                },
                {
                    command: `yq -i '.spec.controlPlaneEndpoint = {}' /opt/manage/1master1worker-mgmt-cluster.yaml`,
                    errorMessage: "Failed to reset controlPlaneEndpoint",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and has("etcdVersion")).etcdVersion) = "v3.6.7-of.0"' /opt/manage/1master1worker-mgmt-cluster.yaml`,
                    errorMessage: "Failed to update etcdVersion",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and has("kubernetesVersion")).kubernetesVersion) = "v1.33.1-of.2"' /opt/manage/1master1worker-mgmt-cluster.yaml`,
                    errorMessage: "Failed to update kubernetesVersion",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and .name == "coredns").version) = "v1.12.2-of.1"' /opt/manage/1master1worker-mgmt-cluster.yaml`,
                    errorMessage: "Failed to update coredns version",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and .name == "kubeproxy").version) = "v1.33.1-of.2"' /opt/manage/1master1worker-mgmt-cluster.yaml`,
                    errorMessage: "Failed to update kubeproxy version",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and .name == "calico").version) = "v3.31.3"' /opt/manage/1master1worker-mgmt-cluster.yaml`,
                    errorMessage: "Failed to update calico version",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and .name == "cluster-api").param.containerdVersion) = "v2.1.1" | (.. | select(tag == "!!map" and .name == "cluster-api").param.openFuyaoVersion) = "v26.05" | (.. | select(tag == "!!map" and .name == "cluster-api").param.providerVersion) = "26.05" | (.. | select(tag == "!!map" and .name == "cluster-api").param.manifestsVersion) = "26.05"' /opt/manage/1master1worker-mgmt-cluster.yaml`,
                    errorMessage: "Failed to update cluster-api params",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and has("openFuyaoVersion")).openFuyaoVersion) = "v26.05"' /opt/manage/1master1worker-mgmt-cluster.yaml`,
                    errorMessage: "Failed to update openFuyaoVersion",
                },
                {
                    command: "bke cluster create -f /opt/manage/1master1worker-mgmt-cluster.yaml -n /opt/manage/1master1worker-mgmt-nodes.yaml",
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
            const appendWorkerNodeCommand = `cat >> /opt/manage/3master1worker-mgmt-nodes.yaml <<'EOF'
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
                    command: "mkdir -p /opt/manage",
                    errorMessage: "Failed to create manage config directory",
                },
                {
                    command: "cp /bke/cluster/1master1node-cluster.yaml /opt/manage/3master1worker-mgmt-cluster.yaml",
                    errorMessage: "Failed to copy management cluster config",
                },
                {
                    command: "cp /bke/cluster/3master-nodes.yaml /opt/manage/3master1worker-mgmt-nodes.yaml",
                    errorMessage: "Failed to copy 3master node config",
                },
                {
                    command: `CLUSTER_NAME=${shellQuote(clusterName)} yq -i '.metadata.name = strenv(CLUSTER_NAME) | .metadata.namespace = strenv(CLUSTER_NAME)' /opt/manage/3master1worker-mgmt-cluster.yaml`,
                    errorMessage: "Failed to update management cluster metadata",
                },
                {
                    command: `CLUSTER_NAME=${shellQuote(clusterName)} yq -i '.metadata.namespace = strenv(CLUSTER_NAME) | .metadata.labels."cluster.x-k8s.io/cluster-name" = strenv(CLUSTER_NAME)' /opt/manage/3master1worker-mgmt-nodes.yaml`,
                    errorMessage: "Failed to update management node metadata",
                },
                {
                    command: `NODE_IP=${shellQuote(master1Ip)} NODE_PASSWORD=${shellQuote(master1Password)} yq -i 'with(select(documentIndex == 0); .spec.ip = strenv(NODE_IP) | .spec.password = strenv(NODE_PASSWORD) | .spec.username = "root")' /opt/manage/3master1worker-mgmt-nodes.yaml`,
                    errorMessage: "Failed to update first master node spec",
                },
                {
                    command: `NODE_IP=${shellQuote(master2Ip)} NODE_PASSWORD=${shellQuote(master2Password)} yq -i 'with(select(documentIndex == 1); .spec.ip = strenv(NODE_IP) | .spec.password = strenv(NODE_PASSWORD) | .spec.username = "root")' /opt/manage/3master1worker-mgmt-nodes.yaml`,
                    errorMessage: "Failed to update second master node spec",
                },
                {
                    command: `NODE_IP=${shellQuote(master3Ip)} NODE_PASSWORD=${shellQuote(master3Password)} yq -i 'with(select(documentIndex == 2); .spec.ip = strenv(NODE_IP) | .spec.password = strenv(NODE_PASSWORD) | .spec.username = "root")' /opt/manage/3master1worker-mgmt-nodes.yaml`,
                    errorMessage: "Failed to update third master node spec",
                },
                {
                    command: appendWorkerNodeCommand,
                    errorMessage: "Failed to append worker node config",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and has("etcdVersion")).etcdVersion) = "v3.6.7-of.0"' /opt/manage/3master1worker-mgmt-cluster.yaml`,
                    errorMessage: "Failed to update etcdVersion",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and has("kubernetesVersion")).kubernetesVersion) = "v1.33.1-of.2"' /opt/manage/3master1worker-mgmt-cluster.yaml`,
                    errorMessage: "Failed to update kubernetesVersion",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and .name == "coredns").version) = "v1.12.2-of.1"' /opt/manage/3master1worker-mgmt-cluster.yaml`,
                    errorMessage: "Failed to update coredns version",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and .name == "kubeproxy").version) = "v1.33.1-of.2"' /opt/manage/3master1worker-mgmt-cluster.yaml`,
                    errorMessage: "Failed to update kubeproxy version",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and .name == "calico").version) = "v3.31.3"' /opt/manage/3master1worker-mgmt-cluster.yaml`,
                    errorMessage: "Failed to update calico version",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and .name == "cluster-api").param.containerdVersion) = "v2.1.1" | (.. | select(tag == "!!map" and .name == "cluster-api").param.openFuyaoVersion) = "v26.05" | (.. | select(tag == "!!map" and .name == "cluster-api").param.providerVersion) = "26.05" | (.. | select(tag == "!!map" and .name == "cluster-api").param.manifestsVersion) = "26.05"' /opt/manage/3master1worker-mgmt-cluster.yaml`,
                    errorMessage: "Failed to update cluster-api params",
                },
                {
                    command: `yq -i '(.. | select(tag == "!!map" and has("openFuyaoVersion")).openFuyaoVersion) = "v26.05"' /opt/manage/3master1worker-mgmt-cluster.yaml`,
                    errorMessage: "Failed to update openFuyaoVersion",
                },
            ];

            if (vipHost && vipPort) {
                commands.splice(5, 0, {
                    command: `VIP_HOST=${shellQuote(vipHost)} VIP_PORT=${shellQuote(vipPort)} yq -i '.spec.controlPlaneEndpoint.host = strenv(VIP_HOST) | .spec.controlPlaneEndpoint.port = env(VIP_PORT)' /opt/manage/3master1worker-mgmt-cluster.yaml`,
                    errorMessage: "Failed to update controlPlaneEndpoint",
                });
            }

            commands.push({
                command: "bke cluster create -f /opt/manage/3master1worker-mgmt-cluster.yaml -n /opt/manage/3master1worker-mgmt-nodes.yaml",
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
    await page.goto(process.env.TEST_MANAGERURL + urls.loginUrl);

    await page.getByRole('textbox', { name: '*用户名' }).fill(process.env.TEST_USERNAME!);
    await page.getByRole('textbox', { name: '*密码' }).fill(process.env.TEST_INITIAL_PASSWORD!);
    await page.getByRole('button', { name: '登录' }).click();

    const firstLoginHeading = page.getByRole('heading', { name: '初次修改密码' });
    const currentUserText = page.getByText(process.env.TEST_USERNAME!);

    await Promise.race([
        firstLoginHeading.waitFor({ state: 'visible', timeout: 15000 }).catch(() => undefined),
        currentUserText.waitFor({ state: 'visible', timeout: 15000 }).catch(() => undefined),
    ]);

    if (await firstLoginHeading.isVisible().catch(() => false)) {
        await firstLoginChangePassword(page);
        await expect(page.getByText(process.env.TEST_USERNAME)).toBeVisible();
        const count = await page.locator("text=admin").count();
        console.log("admin 出现次数:", count);
        return;
    }

    if (await currentUserText.isVisible().catch(() => false)) {
        return;
    }

    await page.goto(process.env.TEST_MANAGERURL + urls.loginUrl);
    await login(page, process.env.TEST_USERNAME, process.env.TEST_PASSWORD);
}

async function loginGuideClusterIfNeeded(page: Page) {
    await page.goto(process.env.TEST_GUIDEURL + urls.loginUrl);

    const usernameInput = page.getByRole('textbox', { name: '*用户名' });
    const lifecycleLink = page.getByRole('link', { name: '集群生命周期管理' });
    await Promise.race([
        usernameInput.waitFor({ state: 'visible', timeout: 15000 }).catch(() => undefined),
        lifecycleLink.waitFor({ state: 'visible', timeout: 15000 }).catch(() => undefined),
    ]);

    if (await usernameInput.isVisible({ timeout: 5000 }).catch(() => false)) {
        await login(page, process.env.TEST_USERNAME, process.env.TEST_PASSWORD);
    }

    await gotoLifecycleIfPossible(page);
}

/*
  - 每 15 秒尝试访问一次 managerUrl + urls.loginUrl
  - 每次访问最多等 10 秒
  - 如果访问抛错、超时，或 HTTP 状态码 >= 400，就认为管理集群页面已经不可访问，继续后续逻辑
  - 最多等 18 分钟，超过还可访问就抛错
 */
async function waitForPageUnavailable(
    page: Page,
    url: string,
    options?: {
        timeoutMs?: number;
        intervalMs?: number;
        navigationTimeoutMs?: number;
    },
) {
    const timeoutMs = options?.timeoutMs ?? 18 * 60 * 1000;
    const intervalMs = options?.intervalMs ?? 15 * 1000;
    const navigationTimeoutMs = options?.navigationTimeoutMs ?? 10 * 1000;
    const start = Date.now();

    while (Date.now() - start < timeoutMs) {
        try {
            const response = await page.goto(url, {
                waitUntil: 'domcontentloaded',
                timeout: navigationTimeoutMs,
            });
            if (response && response.status() >= 400) {
                return;
            }
        } catch {
            return;
        }

        await page.waitForTimeout(intervalMs);
    }

    throw new Error(`Page ${url} is still accessible after ${timeoutMs}ms`);
}


async function fillClusterInfo(
    page: Page,
    clusterName: string,
    masterInfo: NodeInfoFields,
    fuyaoVersion?: string,
    addon?: string,
) {
    await goToCreateClusterPage(page);
    await setClusterInfo(page, clusterName, fuyaoVersion);
    await setNodeInfo(page, 'master', 0, masterInfo);
    await setAdvancedConfig(page, addon ?? addonOptions.noAddon);
}
const TEST_CLUSTER_NAME_1M1W = 'bke-cluster-mgmt-upgrade-1m1w';
const TEST_CLUSTER_NAME_3M1W = 'bke-cluster-mgmt-upgrade-3m1w';
async function upgradeCluster(page: Page, clusterName: string, upgradeVersion: string) {
    await startClusterUpgrade(page, clusterName, upgradeVersion);
    await verifyClusterUpgrading(page, clusterName);
    await verifyClusterUpgradeCompletedStable(page, clusterName, upgradeVersion);
}

async function deleteClusterIfVisible(page: Page, clusterName: string): Promise<boolean> {
    const clusterCell = page.getByRole('cell', { name: clusterName }).first();
    try {
        await clusterCell.waitFor({ state: 'visible', timeout: 5000 });
    } catch {
        return false;
    }

    await verifyClusterHealthOrInstallFailed(page, clusterName);
    await deleteCluster(page, clusterName);
    await verifyClusterDeleted(page, clusterName);
    return true;
}

async function gotoLifecycleIfPossible(page: Page) {
    const lifecycleLink = page.getByRole('link', { name: '集群生命周期管理' });
    if (await lifecycleLink.isVisible().catch(() => false)) {
        await lifecycleLink.click();
    }
}

async function ensureLifecycleVisible(page: Page) {
    const lifecycleLink = page.getByRole('link', { name: '集群生命周期管理' });
    if (await lifecycleLink.isVisible({ timeout: 5000 }).catch(() => false)) {
        return;
    }

    await page.getByRole('img', { name: 'global' }).click();
    console.log("当前是英语,点击了一下所以变成了中文")
    await expect(lifecycleLink).toBeVisible({ timeout: timeouts.validateTimeout });
}

test.beforeEach(async ({ page }) => {
    // 登录管理集群节点
    test.setTimeout(totalTimeout);
    // await page.goto(managerUrl + urls.loginUrl);
    // await login(page, process.env.TEST_USERNAME, process.env.TEST_PASSWORD);
    // await page.getByRole('link', { name: '集群生命周期管理' }).click();

    // 登录 admin 账号
    await page.goto(process.env.TEST_GUIDEURL + urls.loginUrl)
    await login(page, process.env.TEST_USERNAME, process.env.TEST_PASSWORD);
});

// test.afterEach(async ({ page }) => {
//     const manageClusterName = managementClusterSelfDisplayName(TEST_CLUSTER_NAME_1M1W);
//
//     try {
//         await loginManagementCluster(page);
//         await gotoLifecycleIfPossible(page);
//         const deleted = await deleteClusterIfVisible(page, manageClusterName);
//         if (!deleted) {
//             console.warn(`afterEach cleanup skip management cluster delete: cluster ${manageClusterName} not found`);
//         }
//     } catch (error) {
//         console.warn(
//             'afterEach cleanup skip management cluster delete:',
//             error instanceof Error ? error.message : String(error),
//         );
//     }
//
//     try {
//         await page.goto(process.env.TEST_GUIDEURL + urls.loginUrl);
//         await login(page, process.env.TEST_USERNAME, process.env.TEST_PASSWORD);
//         const deleted1M1W = await deleteClusterIfVisible(page, TEST_CLUSTER_NAME_1M1W);
//         const deleted3M1W = await deleteClusterIfVisible(page, TEST_CLUSTER_NAME_3M1W);
//         if (!deleted1M1W && !deleted3M1W) {
//             console.warn(`afterEach cleanup skip guide cluster delete: clusters ${TEST_CLUSTER_NAME_1M1W}/${TEST_CLUSTER_NAME_3M1W} not found`);
//         }
//     } catch (error) {
//         console.warn(
//             'afterEach cleanup skip guide cluster delete:',
//             error instanceof Error ? error.message : String(error),
//         );
//     }
// });

test.describe('管理集群升级管理集群', { tag: ["@post-init","@custom-repo"] },  () => {
    test.describe.configure({ mode: 'serial' });

    // test('升级管理集群到指定版本', async ({ page }) => {
    //     const manageClusterName = managementClusterSelfDisplayName(TEST_CLUSTER_NAME_1M1W);
    //     const targetVersion = "v26.03.1";
    //     await page.getByRole('link', { name: '集群生命周期管理' }).click();
    //     await startClusterUpgrade(page, manageClusterName, targetVersion);
    //     // 等待升级完成，集群恢复健康状态
    //     await verifyClusterHealthy(page, manageClusterName);
    //     // await page.getByRole('cell', { name: manageClusterName }).waitFor({ state: 'visible' });
    //     await expect(page.getByRole('cell', { name: targetVersion }).first()).toBeVisible();
    //     await verifyClusterHealthOrInstallFailed(page, manageClusterName);
    //     await deleteCluster(page, manageClusterName);
    //     await verifyClusterDeleted(page, manageClusterName);
    // });


    // test("管理集群升级业务集群", async ({ page }) => {
    //     test.setTimeout(totalTimeout);
    //     const clusterName = 'test-cluster';
    //     await test.step("创建初始集群", async () => {
    //         const fuyaoVersion = "v26.03";
    //         // 设置 集群 信息
    //         await fillClusterInfo(page, clusterName, node1Info, fuyaoVersion);
    //         // 创建集群
    //         await page.getByRole('button', { name: '一键校验' }).click();
    //         await createCluster(page);
    //         // 验证
    //         await verifyClusterInstalling(page, clusterName);
    //         await verifyClusterHealthy(page, clusterName);
    //         await expect(page.getByRole('cell', { name: 'v26.03' }).first()).toBeVisible();
    //     });

    //     await test.step("集群升级", async () => {
    //         const upgradeVersion = "v26.03.1";
    //         await upgradeCluster(page, clusterName, upgradeVersion);
    //         await expect(page.getByRole('cell', { name: 'v26.03.1' }).first()).toBeVisible();
    //     });

    //     await test.step("删除业务集群", async () => {
    //         await page.getByRole('cell', { name: clusterName }).waitFor({ state: 'visible' });
    //         await verifyClusterHealthOrInstallFailed(page, clusterName);
    //         await deleteCluster(page, clusterName);
    //         await verifyClusterDeleted(page, clusterName);
    //     });
    // });


    test("从引导节点创建1master1worker管理集群,从管理集群页面自升级", async ({ page }) => {
        test.setTimeout(totalTimeout * 2);
        const clusterName = TEST_CLUSTER_NAME_1M1W;
        await test.step("创建1master1worker管理集群", async () => {
            await createManageClusterBySsh(clusterName);
            await verifyClusterInstallStartedOrSettled(page, clusterName);
            await verifyClusterHealthy2(page, clusterName);
            await expect(page.getByRole('cell', { name: 'v26.05' }).first()).toBeVisible();
        });

        await test.step("登录管理集群并处理首次改密", async () => {
            await loginManagementCluster(page);
        });

        await test.step("管理集群自升级到 v26.06", async () => {
            await ensureLifecycleVisible(page);
            await page.getByRole('link', { name: '集群生命周期管理' }).click();
            await page.getByRole('link', { name: '总览' }).click();
            await page.getByRole('link', { name: '集群生命周期管理' }).click();
            // #224: 自管 BC UI 名继承引导侧创建资源名，不再固定 bke-cluster
            const manageClusterName = managementClusterSelfDisplayName(clusterName);
            await verifyClusterHealthy(page, manageClusterName);
            await expect(page.getByRole('cell', { name: 'v26.05' }).first()).toBeVisible();

            const targetVersion = "v26.06";

            await startClusterUpgrade(page, manageClusterName, targetVersion);
            await waitForClusterUpgradeStarted(page, manageClusterName, { requiredOccurrences: 2, intervalMs: 3000 });
            await waitForClusterUpgradeFinishedAfterStarted(page, manageClusterName, targetVersion, { intervalMs: 20000 });
        });

        await test.step("删除管理集群自管 BC 并清理引导页面记录", async () => {
            const manageClusterName = managementClusterSelfDisplayName(clusterName);

            await verifyClusterHealthOrInstallFailed(page, manageClusterName);
            await deleteCluster(page, manageClusterName);
            await waitForPageUnavailable(page, managerUrl + urls.loginUrl);

            await loginGuideClusterIfNeeded(page);
            await deleteCluster(page, clusterName);
            await verifyClusterDeleted(page, clusterName);
        });

    });

    test("从引导节点创建3master1worker管理集群,从管理集群页面自升级",  async ({ page }) => {
        test.setTimeout(totalTimeout * 2);
        const clusterName = TEST_CLUSTER_NAME_3M1W;

        await test.step("创建3master1worker管理集群", async () => {
            await createManageCluster3M1WBySsh(clusterName);
            await verifyClusterInstallStartedOrSettled(page, clusterName);
            await verifyClusterHealthy2(page, clusterName);
            await expect(page.getByRole('cell', { name: 'v26.05' }).first()).toBeVisible();
        });

        await test.step("登录管理集群并处理首次改密", async () => {
            await loginManagementCluster(page);
            // await page.goto(process.env.TEST_MANAGERURL + urls.loginUrl)
            // await login(page, process.env.TEST_USERNAME, process.env.TEST_PASSWORD);
        });

        await test.step("管理集群自升级到 v26.06", async () => {

            await ensureLifecycleVisible(page);
            await page.getByRole('link', { name: '集群生命周期管理' }).click();
            await page.getByRole('link', { name: '总览' }).click();
            await page.getByRole('link', { name: '集群生命周期管理' }).click();
            // #224: 自管 BC UI 名继承引导侧创建资源名，不再固定 bke-cluster
            const manageClusterName = managementClusterSelfDisplayName(clusterName);
            await verifyClusterHealthy(page, manageClusterName);
            await expect(page.getByRole('cell', { name: 'v26.05' }).first()).toBeVisible();

            const targetVersion = "v26.06";

            await startClusterUpgrade(page, manageClusterName, targetVersion);
            await waitForClusterUpgradeStarted(page, manageClusterName, { requiredOccurrences: 2, intervalMs: 3000, fallbackMs: 4 * 60 * 1000 });
            await waitForClusterUpgradeFinishedAfterStarted(page, manageClusterName, targetVersion, { intervalMs: 20000 });
            // await waitForClusterUpgradeStarted(page, manageClusterName, { requiredOccurrences: 2, intervalMs: 5000 });
        });

        await test.step("删除管理集群自管 BC 并清理引导页面记录", async () => {
            const manageClusterName = managementClusterSelfDisplayName(clusterName);

            await verifyClusterHealthOrInstallFailed(page, manageClusterName);
            await deleteCluster(page, manageClusterName);
            await waitForPageUnavailable(page, managerUrl + urls.loginUrl);

            await loginGuideClusterIfNeeded(page);
            await deleteCluster(page, clusterName);
            await verifyClusterDeleted(page, clusterName);
        });
    });

});
