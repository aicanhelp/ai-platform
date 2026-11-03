import type { Page } from "@playwright/test";
import { expect, test } from "@playwright/test";
import { nodeInfo, timeouts, urls } from "./constants/common";
import { firstLoginChangePassword, login } from "./utils/rbac";
import {
    verifyClusterDeleted,
    verifyClusterHealthOrInstallFailed,
    verifyClusterUpgrading,
    waitForClusterUpgradeFinishedAfterStarted,
    waitForClusterUpgradeStarted,
} from "./utils/verify";
import { deleteCluster, startClusterUpgrade } from "./utils/clusterManage";
import { BKEAGENT_LOCAL_DIR, BKEAGENT_SOURCE_BASE_URL, withSshConnection } from "@/installation/utils/upgrade-pre";
import { managementClusterSelfDisplayName } from "./utils/managementClusterName";

const { guideNodeInfo } = nodeInfo;
const { totalTimeout } = timeouts;

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


async function initGuideNodeForRc2(): Promise<void> {
    const guideNodeHost = requireEnv("TEST_NODE1_IP");
    const guideNodePassword = requireEnv("TEST_NODE1_PASSWORD");

    await withSshConnection(
        {
            host: guideNodeHost,
            username: "root",
            password: guideNodePassword,
            port: Number(guideNodeInfo.port),
            readyTimeout: 15000,
        },
        async (ssh) => {
            const commands = [
                {
                    command: "curl -sfL https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/26.6.0-rc.2/download.sh | bash",
                    errorMessage: "Failed to install bkeadm bootstrap scripts",
                },
                {
                    command: "echo Y | bke init --onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:26.6.0-rc.2",
                    errorMessage: "Failed to initialize guide node for rc2",
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

async function installManageCluster3M1NByBkeCli(clusterName: string): Promise<void> {
    const guideNodeHost = requireEnv("TEST_NODE1_IP");
    const guideNodePassword = requireEnv("TEST_NODE1_PASSWORD");
    const masterNodes = [
        { ip: requireEnv("TEST_NODE2_IP"), password: requireEnv("TEST_NODE2_PASSWORD") },
        { ip: requireEnv("TEST_NODE3_IP"), password: requireEnv("TEST_NODE3_PASSWORD") },
        { ip: requireEnv("TEST_NODE4_IP"), password: requireEnv("TEST_NODE4_PASSWORD") },
    ];
    const workerNode = {
        ip: requireEnv("TEST_NODE5_IP"),
        password: requireEnv("TEST_NODE5_PASSWORD"),
    };
    const vipHost = process.env.MGMT_CLUSTER_VIP_HOST || process.env.MANAGEMENT_CLUSTER_VIP_HOST || "";
    const vipPort = process.env.MGMT_CLUSTER_VIP_PORT || process.env.MANAGEMENT_CLUSTER_VIP_PORT || "";
    const bcYamlPath = "/bke/cluster/3master-cluster.yaml";
    const bnYamlPath = "/bke/cluster/3master-nodes.yaml";
    const deploymentPath = "/bke/cluster/3master-cluster.yaml";

    await withSshConnection(
        {
            host: guideNodeHost,
            username: "root",
            password: guideNodePassword,
            port: Number(guideNodeInfo.port),
            readyTimeout: 15000,
        },
        async (ssh) => {
            const appendWorkerNodeCommand = `cat >> ${shellQuote(bnYamlPath)} <<'EOF'
---
apiVersion: bke.bocloud.com/v1beta1
kind: BKENode
metadata:
  creationTimestamp: null
  labels:
    cluster.x-k8s.io/cluster-name: ${yamlSingleQuote(clusterName)}
  name: bke-cluster-worker-1
  namespace: ${yamlSingleQuote(clusterName)}
spec:
  hostname: worker-1
  ip: ${yamlSingleQuote(workerNode.ip)}
  password: ${yamlSingleQuote(workerNode.password)}
  port: "22"
  role:
    - node
  username: root
status: {}
EOF`;

            const commands = [
                {
                    command: `cp ${shellQuote(bcYamlPath)} ${shellQuote(bcYamlPath)}.bak`,
                    errorMessage: "Failed to backup bc yaml",
                },
                {
                    command: `cp ${shellQuote(bnYamlPath)} ${shellQuote(bnYamlPath)}.bak`,
                    errorMessage: "Failed to backup bn yaml",
                },
                {
                    command: `cat ${shellQuote(bnYamlPath)}.bak > ${shellQuote(bnYamlPath)} && ${appendWorkerNodeCommand}`,
                    errorMessage: "Failed to reset and append worker node config",
                },
                {
                    command: `CLUSTER_NAME=${shellQuote(clusterName)} yq -i '.metadata.name = strenv(CLUSTER_NAME) | .metadata.namespace = strenv(CLUSTER_NAME)' ${shellQuote(bcYamlPath)}`,
                    errorMessage: "Failed to update bc yaml metadata",
                },
                {
                    command: `CLUSTER_NAME=${shellQuote(clusterName)} yq -i '.metadata.namespace = strenv(CLUSTER_NAME) | .metadata.labels."cluster.x-k8s.io/cluster-name" = strenv(CLUSTER_NAME)' ${shellQuote(bnYamlPath)}`,
                    errorMessage: "Failed to update bn yaml metadata",
                },
            ];

            for (const [index, node] of masterNodes.entries()) {
                commands.push({
                    command: `NODE_IP=${shellQuote(node.ip)} NODE_PASSWORD=${shellQuote(node.password)} yq -i 'with(select(documentIndex == ${index}); .spec.ip = strenv(NODE_IP) | .spec.password = strenv(NODE_PASSWORD) | .spec.username = "root")' ${shellQuote(bnYamlPath)}`,
                    errorMessage: `Failed to update master node ${index + 1} spec`,
                });
            }

            if (vipHost && vipPort) {
                commands.push({
                    command: `VIP_HOST=${shellQuote(vipHost)} VIP_PORT=${shellQuote(vipPort)} yq -i '.spec.controlPlaneEndpoint.host = strenv(VIP_HOST) | .spec.controlPlaneEndpoint.port = env(VIP_PORT)' ${shellQuote(bcYamlPath)}`,
                    errorMessage: "Failed to update controlPlaneEndpoint",
                });
            }

            commands.push(
                {
                    command: `bke cluster create -f ${shellQuote(deploymentPath)} -n ${shellQuote(bnYamlPath)}`,
                    errorMessage: "Failed to create cluster by bke",
                },
            );

            for (const { command, errorMessage } of commands) {
                const result = await ssh.execCommand(command);
                if (result.code !== 0) {
                    throw new Error(`${errorMessage}: ${result.stderr || result.stdout}`);
                }
            }
        },
    );
}

function bkeagentBinaryVersionForTarget(targetVersion: string): string {
    const versionMap: Record<string, string> = {
        "v26.06-rc.3": "26.6.0-rc.3",
        "v26.06": "26.6.0",
    };
    const binaryVersion = versionMap[targetVersion];
    if (!binaryVersion) {
        throw new Error(`Unsupported target version for bkeagent binary: ${targetVersion}`);
    }
    return binaryVersion;
}

function getGuideNodeBkeagentBaseUrl(): string {
    return `http://${requireEnv("TEST_NODE1_IP")}:40080/files`;
}

async function prepareGuideNodeBkeagentBinaries(targetBkeagentVersion: string): Promise<void> {
    const guideNodeHost = requireEnv("TEST_NODE1_IP");
    const guideNodePassword = requireEnv("TEST_NODE1_PASSWORD");

    await withSshConnection(
        {
            host: guideNodeHost,
            username: "root",
            password: guideNodePassword,
            port: Number(guideNodeInfo.port),
            readyTimeout: 15000,
        },
        async (ssh) => {
            const commands = ["arm64", "amd64"].map((arch) => {
                const sourceUrl = `${BKEAGENT_SOURCE_BASE_URL}/${targetBkeagentVersion}/bkeagent_linux_${arch}`;
                const targetPath = `${BKEAGENT_LOCAL_DIR}/bkeagent-${targetBkeagentVersion}-linux-${arch}`;
                return `wget -q -O ${shellQuote(targetPath)} ${shellQuote(sourceUrl)} && chmod 0644 ${shellQuote(targetPath)} && test -s ${shellQuote(targetPath)}`;
            });
            const result = await ssh.execCommand(`set -euo pipefail
mkdir -p ${shellQuote(BKEAGENT_LOCAL_DIR)}
${commands.join("\n")}
ls -lh ${shellQuote(BKEAGENT_LOCAL_DIR)}/bkeagent-${targetBkeagentVersion}-linux-arm64 ${shellQuote(BKEAGENT_LOCAL_DIR)}/bkeagent-${targetBkeagentVersion}-linux-amd64`);
            if (result.code !== 0) {
                throw new Error(`Failed to prepare bkeagent binaries on guide node ${guideNodeHost}: ${result.stderr || result.stdout}`);
            }
        },
    );
}

function getGuideBaseUrl(): string {
    return `https://${requireEnv("TEST_NODE1_IP")}:30010`;
}

function getManagerBaseUrl(): string {
    return `https://${requireEnv("TEST_NODE2_IP")}:31616`;
}


async function submitLoginForm(page: Page, username: string, password: string): Promise<void> {
    await page.getByRole("textbox").nth(0).fill(username);
    await page.getByRole("textbox").nth(1).fill(password);
    await page.getByRole("button", { name: /登录|Login/i }).click();
}

async function loginManagementCluster(page: Page) {
    const username = requireEnv("TEST_USERNAME");
    const initialPassword = process.env.TEST_INITIAL_PASSWORD || "test@1234";
    const password = process.env.TEST_PASSWORD || "test@123";
    process.env.TEST_PASSWORD = password;

    await page.goto(getManagerBaseUrl() + urls.loginUrl);
    await submitLoginForm(page, username, initialPassword);

    const firstLoginHeading = page.getByRole("heading", { name: /首次修改密码|修改密码/ });
    const currentUserText = page.getByText(username).first();

    await Promise.race([
        firstLoginHeading.waitFor({ state: "visible", timeout: 15_000 }).catch(() => undefined),
        currentUserText.waitFor({ state: "visible", timeout: 15_000 }).catch(() => undefined),
    ]);

    if (await firstLoginHeading.isVisible().catch(() => false)) {
        await firstLoginChangePassword(page);
        await expect(page.getByText(username).first()).toBeVisible({ timeout: timeouts.validateTimeout });
        return;
    }

    if (await currentUserText.isVisible().catch(() => false)) {
        return;
    }

    await page.goto(getManagerBaseUrl() + urls.loginUrl);
    await login(page, username, password);
}


async function loginGuideClusterIfNeeded(page: Page) {
    const username = requireEnv("TEST_USERNAME");
    const initialPassword = process.env.TEST_INITIAL_PASSWORD || "test@1234";
    const password = process.env.TEST_PASSWORD || "test@123";
    process.env.TEST_PASSWORD = password;

    await page.goto(getGuideBaseUrl() + urls.loginUrl);

    const usernameInput = page.getByRole("textbox").nth(0);
    const lifecycleLink = page.getByRole("link", { name: "集群生命周期管理" });
    await Promise.race([
        usernameInput.waitFor({ state: "visible", timeout: 15_000 }).catch(() => undefined),
        lifecycleLink.waitFor({ state: "visible", timeout: 15_000 }).catch(() => undefined),
    ]);

    if (await usernameInput.isVisible({ timeout: 5000 }).catch(() => false)) {
        await submitLoginForm(page, username, initialPassword);

        const firstLoginHeading = page.getByRole("heading", { name: /首次修改密码|修改密码/ });
        const currentUserText = page.getByText(username).first();

        await Promise.race([
            firstLoginHeading.waitFor({ state: "visible", timeout: 15_000 }).catch(() => undefined),
            currentUserText.waitFor({ state: "visible", timeout: 15_000 }).catch(() => undefined),
        ]);

        if (await firstLoginHeading.isVisible().catch(() => false)) {
            await firstLoginChangePassword(page);
            await expect(page.getByText(username).first()).toBeVisible({ timeout: timeouts.validateTimeout });
        } else if (!(await currentUserText.isVisible().catch(() => false))) {
            await page.goto(getGuideBaseUrl() + urls.loginUrl);
            await login(page, username, password);
        }
    }

    await gotoLifecycleIfPossible(page);
}

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
                waitUntil: "domcontentloaded",
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

async function gotoLifecycleIfPossible(page: Page) {
    const lifecycleLink = page.getByRole("link", { name: "集群生命周期管理" });
    if (await lifecycleLink.isVisible().catch(() => false)) {
        await lifecycleLink.click();
    }
}

async function ensureLifecycleVisible(page: Page) {
    const lifecycleLink = page.getByRole("link", { name: "集群生命周期管理" });
    if (await lifecycleLink.isVisible({ timeout: 5000 }).catch(() => false)) {
        return;
    }

    await page.getByRole("img", { name: "global" }).click();
    await expect(lifecycleLink).toBeVisible({ timeout: timeouts.validateTimeout });
}

async function openLifecyclePage(page: Page, clusterName?: string) {
    const clusterCell = clusterName
        ? page.getByRole("cell", { name: clusterName, exact: true }).first()
        : undefined;
    const deadline = Date.now() + timeouts.installTimeout;

    do {
        const overviewLink = page.getByRole("link", { name: "总览" });
        if (await overviewLink.isVisible().catch(() => false)) {
            await overviewLink.click();
            await page.waitForTimeout(500);
        }

        await ensureLifecycleVisible(page);
        await page.getByRole("link", { name: "集群生命周期管理" }).click();
        await page.waitForLoadState("domcontentloaded").catch(() => undefined);

        if (!clusterCell || await clusterCell.isVisible().catch(() => false)) {
            return;
        }

        await page.waitForTimeout(10_000);
    } while (Date.now() < deadline);

    if (clusterCell) {
        await expect(clusterCell).toBeVisible({ timeout: 0 });
    }
}
async function injectMonitoringPodImagePullError(): Promise<void> {
    const managerMasterHost = requireEnv("TEST_NODE2_IP");
    const managerMasterPassword = requireEnv("TEST_NODE2_PASSWORD");

    await withSshConnection(
        {
            host: managerMasterHost,
            username: "root",
            password: managerMasterPassword,
            port: Number(guideNodeInfo.port),
            readyTimeout: 15000,
        },
        async (ssh) => {
            const command = `set -e
KUBECONFIG=/etc/kubernetes/admin.conf
NS=monitoring
DEPLOY=blackbox-exporter
kubectl --kubeconfig "$KUBECONFIG" -n "$NS" get deploy "$DEPLOY"
kubectl --kubeconfig "$KUBECONFIG" -n "$NS" patch deploy "$DEPLOY" --type='json' -p='[{"op":"replace","path":"/spec/template/spec/containers/0/image","value":"cr.openfuyao.cn/library/this-image-does-not-exist:latest"}]'
kubectl --kubeconfig "$KUBECONFIG" -n "$NS" rollout restart deploy "$DEPLOY"
kubectl --kubeconfig "$KUBECONFIG" -n "$NS" rollout status deploy "$DEPLOY" --timeout=120s || true
kubectl --kubeconfig "$KUBECONFIG" -n "$NS" get pod -l app.kubernetes.io/name=blackbox-exporter -o wide`;
            const result = await ssh.execCommand(command);
            if (result.code !== 0) {
                throw new Error(`Failed to inject blackbox-exporter image pull error: ${result.stderr || result.stdout}`);
            }
        },
    );
}
async function openLifecycleAndVerifyHealthy(page: Page, manageClusterName: string) {
    await openLifecyclePage(page, manageClusterName);

    await page.getByRole("cell", { name: manageClusterName, exact: true }).first().click();
    const healthyButton = page.getByRole("button", { name: "健康", exact: true });
    await expect(healthyButton).toBeVisible({ timeout: timeouts.validateTimeout });
}

async function upgradeAndVerify(page: Page, manageClusterName: string, targetVersion: string) {
    await openLifecyclePage(page, manageClusterName);
    await startClusterUpgrade(page, manageClusterName, targetVersion);
    await verifyClusterUpgrading(page, manageClusterName);
    await waitForClusterUpgradeStarted(page, manageClusterName, { requiredOccurrences: 2, intervalMs: 3000, fallbackMs: 4 * 60 * 1000 });
    await waitForClusterUpgradeFinishedAfterStarted(page, manageClusterName, targetVersion, { intervalMs: 20000 });
}

async function cleanupCreatedClustersThroughFrontend(
    page: Page,
    guideClusterName: string,
    manageClusterName: string,
): Promise<void> {
    await openLifecyclePage(page, manageClusterName);
    await verifyClusterHealthOrInstallFailed(page, manageClusterName);
    await deleteCluster(page, manageClusterName);
    await waitForPageUnavailable(page, getManagerBaseUrl() + urls.loginUrl);

    await loginGuideClusterIfNeeded(page);
    await openLifecyclePage(page, guideClusterName);
    await deleteCluster(page, guideClusterName);
    await verifyClusterDeleted(page, guideClusterName);
}
async function resetGuideNode(): Promise<void> {
    const guideNodeHost = requireEnv("TEST_NODE1_IP");
    const guideNodePassword = requireEnv("TEST_NODE1_PASSWORD");

    await withSshConnection(
        {
            host: guideNodeHost,
            username: "root",
            password: guideNodePassword,
            port: Number(guideNodeInfo.port),
            readyTimeout: 15000,
        },
        async (ssh) => {
            const result = await ssh.execCommand(`set +e
bke reset --all --mount
rm -rf /bke
exit 0`);
            if (result.code !== 0) {
                throw new Error(`Failed to reset guide node: ${result.stderr || result.stdout}`);
            }
        },
    );
}
test.describe('集群状态正常但存在 pod 异常时触发升级', { tag: ["@v26.06", "@online", "@pre-init", "@skip-temporarily"] }, () => {
    test.describe.configure({ mode: 'serial' });

    let guideClusterNameForCleanup = "";
    let managementClusterCreated = false;

    test.afterEach(async () => {
        try {
            if (managementClusterCreated || guideClusterNameForCleanup) {
                await resetGuideNode();
            }
        } finally {
            guideClusterNameForCleanup = "";
            managementClusterCreated = false;
        }
    });

    test('pod状态异常时触发升级-前端-高可用-管理集群-3m1w', async ({ page }) => {
        test.setTimeout(totalTimeout * 2);
        const clusterName = 'bke-cluster-mgmt-upgrade-abnormal-pod-3m1w';
        // #224: 自管 BC UI 名继承引导侧创建资源名，不再固定 bke-cluster
        const manageClusterName = managementClusterSelfDisplayName(clusterName);
        const targetVersion = 'v26.06-rc.3';

        guideClusterNameForCleanup = clusterName;
        managementClusterCreated = false;

        await test.step('初始化引导节点到 rc2 版本并准备 rc3 bkeagent', async () => {
            await initGuideNodeForRc2();
            await prepareGuideNodeBkeagentBinaries(bkeagentBinaryVersionForTarget(targetVersion));
        });

        await test.step('通过后端命令安装高可用3m1n管理集群', async () => {
            managementClusterCreated = true;
            await installManageCluster3M1NByBkeCli(clusterName);
        });

        await test.step('登录 openFuyao 前端并进入集群生命周期管理', async () => {
            await loginManagementCluster(page);
            await openLifecycleAndVerifyHealthy(page, manageClusterName);
        });

        await test.step('制造 monitoring 命名空间 pod 镜像拉取异常', async () => {
            await injectMonitoringPodImagePullError();
        });

        await test.step('找到对应集群点击升级最近版本', async () => {
            await upgradeAndVerify(page, manageClusterName, targetVersion);
        });

        await test.step('查看集群状态并清理', async () => {
            await openLifecyclePage(page, manageClusterName);
            await verifyClusterHealthOrInstallFailed(page, manageClusterName);
            await expect(page.getByRole('cell', { name: targetVersion }).first()).toBeVisible();
            await cleanupCreatedClustersThroughFrontend(page, clusterName, manageClusterName);
            await resetGuideNode();
            guideClusterNameForCleanup = "";
            managementClusterCreated = false;
        });
    });

    test('pod状态异常时触发跨版本升级-前端-高可用-管理集群-3m1w', async ({ page }) => {
        test.setTimeout(totalTimeout * 2);
        const clusterName = 'bke-cluster-mgmt-upgrade-cross-abnormal-pod-3m1w';
        // #224: 自管 BC UI 名继承引导侧创建资源名，不再固定 bke-cluster
        const manageClusterName = managementClusterSelfDisplayName(clusterName);
        const targetVersion = 'v26.06';

        guideClusterNameForCleanup = clusterName;
        managementClusterCreated = false;

        await test.step('初始化引导节点到 rc2 版本并准备 26.6.0 bkeagent', async () => {
            await initGuideNodeForRc2();
            await prepareGuideNodeBkeagentBinaries(bkeagentBinaryVersionForTarget(targetVersion));
        });

        await test.step('通过后端命令安装高可用3m1n管理集群', async () => {
            managementClusterCreated = true;
            await installManageCluster3M1NByBkeCli(clusterName);
        });

        await test.step('登录 openFuyao 前端并进入集群生命周期管理', async () => {
            await loginManagementCluster(page);
            await openLifecycleAndVerifyHealthy(page, manageClusterName);
        });

        await test.step('制造 monitoring 命名空间 pod 镜像拉取异常', async () => {
            await injectMonitoringPodImagePullError();
        });

        await test.step('找到对应集群点击升级跨2个版本', async () => {
            await upgradeAndVerify(page, manageClusterName, targetVersion);
        });

        await test.step('查看集群状态并清理', async () => {
            await openLifecyclePage(page, manageClusterName);
            await verifyClusterHealthOrInstallFailed(page, manageClusterName);
            await expect(page.getByRole('cell', { name: targetVersion }).first()).toBeVisible();
            await cleanupCreatedClustersThroughFrontend(page, clusterName, manageClusterName);
            await resetGuideNode();
            guideClusterNameForCleanup = "";
            managementClusterCreated = false;
        });
    });
});

