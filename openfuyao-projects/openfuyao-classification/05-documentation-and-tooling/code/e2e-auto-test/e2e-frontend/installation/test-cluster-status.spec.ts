import type { Page } from "@playwright/test";
import { expect, test } from "@playwright/test";
import type { NodeInfoEntry } from "./constants/common";
import { addonOptions, createClusterData, nodeInfo, timeouts, urls } from "./constants/common";
import { firstLoginChangePassword, login } from "./utils/rbac";
import { verifyClusterDeleted, verifyClusterHealthy, verifyClusterScaledDown, verifyClusterScaledUp } from "./utils/verify";
import {
    getDeploymentContainerImage,
    triggerDeploymentImagePullError,
    updateDeploymentContainerImage,
} from "./utils/upgrade-pre";
import {
    closeLogModal,
    createCluster,
    deleteCluster,
    goToCreateClusterPage,
    scaleDownConfirm,
    scaleUpCheckNodeInfo,
    scaleUpConfirm,
    scaleUpSetNodeInfo,
    setAdvancedConfig,
    setClusterInfo,
    setNodeInfo,
} from "./utils/clusterManage";
import { resolveManagementClusterSelfDisplayNameFromEnv } from "./utils/managementClusterName";

const { node1Info, node2Info, guideNodeInfo } = nodeInfo;
const { totalTimeout, validateTimeout, installTimeout } = timeouts;
const TEST_CLUSTER_NAME = "test-cluster";
const MANAGEMENT_KUBECONFIG = "/etc/kubernetes/admin.conf";
const CALICO_KUBE_CONTROLLERS = {
    namespace: "kube-system",
    deploymentName: "calico-kube-controllers",
    containerName: "calico-kube-controllers",
    goodImage: "hub.oepkgs.net/openfuyao/calico/kube-controllers:v3.31.3",
};
const MONITORING_DASHBOARD = {
    componentName: "monitoring-dashboard",
    appName: "test-monitor",
    namespace: "monitoring-dashboard",
    deploymentName: "monitoring-dashboard-service",
    containerName: "monitoring-dashboard-service",
};
const BAD_IMAGE_TAG = "notfound-e2e";
const BUSINESS_CORE_COMPONENTS = [
    {
        namespace: "openfuyao-system",
        deploymentName: "application-management-service",
        containerName: "application-management-service",
    },
];

type BusinessComponent = (typeof BUSINESS_CORE_COMPONENTS)[number];

async function fillClusterInfo(
    page: Page,
    clusterName: string,
    masterInfo: NodeInfoEntry,
    fuyaoVersion?: string,
    addon?: string,
) {
    await goToCreateClusterPage(page);
    await setClusterInfo(page, clusterName, fuyaoVersion);
    await setNodeInfo(page, "master", 0, masterInfo);
    await setAdvancedConfig(page, addon ?? addonOptions.noAddon);
    await page.getByRole("button", { name: "一键校验" }).click();
    await expect(page.getByText("一键校验成功，请创建")).toBeVisible({ timeout: validateTimeout });
}

async function verifyClusterInstallingWithException(page: Page, clusterName: string) {
    await expect(page.getByText("集群信息")).toBeVisible();
    await page.waitForFunction(() => {
        const el = document.querySelector(".clusterLogContent");
        return el && el.textContent && el.textContent.trim().length > 0;
    });
    await page.getByRole("button", { name: "退出日志" }).click();
    await page.waitForURL(/\/container_platform\/cluster\/cluster$/);
    const installingPattern = new RegExp(`${clusterName}.*安装中`);
    await expect(page.getByRole("row", { name: installingPattern })).toBeVisible({ timeout: validateTimeout });
}

async function verifyClusterAbnormal(page: Page, clusterName: string, timeoutMs = installTimeout) {
    const abnormalPattern = new RegExp(`${clusterName}.*异常`);
    await expect(page.getByRole("row", { name: abnormalPattern })).toBeVisible({ timeout: timeoutMs });
}

async function mutateImageToBad(image: string) {
    const imageWithoutTag = image.includes(":") ? image.slice(0, image.lastIndexOf(":")) : image;
    return `${imageWithoutTag}:${BAD_IMAGE_TAG}`;
}

async function breakBusinessComponentImage(node: NodeInfoEntry, component: BusinessComponent) {
    const originImage = await getDeploymentContainerImage({
        host: node.ip,
        username: node.userName,
        password: node.password,
        namespace: component.namespace,
        deploymentName: component.deploymentName,
        containerName: component.containerName,
    });
    const badImage = await mutateImageToBad(originImage);
    await updateDeploymentContainerImage({
        host: node.ip,
        username: node.userName,
        password: node.password,
        namespace: component.namespace,
        deploymentName: component.deploymentName,
        containerName: component.containerName,
        image: badImage,
        waitRollout: false,
    });
    return originImage;
}

async function restoreBusinessComponentImage(node: NodeInfoEntry, component: BusinessComponent, image: string) {
    await updateDeploymentContainerImage({
        host: node.ip,
        username: node.userName,
        password: node.password,
        namespace: component.namespace,
        deploymentName: component.deploymentName,
        containerName: component.containerName,
        image,
    });
}

async function loginBusinessClusterConsole(page: Page, baseUrl: string) {
    const username = requireEnv("TEST_USERNAME");
    const initialPassword = process.env.TEST_INITIAL_PASSWORD || "test@1234";
    const password = process.env.TEST_PASSWORD || "test@123";
    process.env.TEST_PASSWORD = password;

    const submitLoginForm = async (loginPassword: string) => {
        await page.getByRole("textbox").nth(0).fill(username);
        await page.getByRole("textbox").nth(1).fill(loginPassword);
        const loginButton = page.getByRole("button", { name: /登录|Login/i });
        await expect(loginButton).toBeEnabled({ timeout: 15_000 });
        await loginButton.click();
    };

    await page.goto(baseUrl + urls.loginUrl);
    await submitLoginForm(initialPassword);

    const firstLoginHeading = page.getByRole("heading", { name: /首次修改密码|修改密码/ });
    const currentUserText = page.getByText(username).first();
    await Promise.race([
        firstLoginHeading.waitFor({ state: "visible", timeout: 15_000 }).catch(() => undefined),
        currentUserText.waitFor({ state: "visible", timeout: 15_000 }).catch(() => undefined),
    ]);

    if (await firstLoginHeading.isVisible().catch(() => false)) {
        await firstLoginChangePassword(page);
        await expect(currentUserText).toBeVisible({ timeout: validateTimeout });
        return;
    }

    if (await currentUserText.isVisible().catch(() => false)) {
        return;
    }

    await page.goto(baseUrl + urls.loginUrl);
    await login(page, username, password);
}

async function fillVisiblePasswordFields(page: Page, password = requireEnv("TEST_PASSWORD")) {
    const passwordInputs = page.locator('input[type="password"]');
    const count = await passwordInputs.count();
    for (let i = 0; i < count; i += 1) {
        const input = passwordInputs.nth(i);
        if (await input.isVisible().catch(() => false)) {
            await input.fill(password);
        }
    }
}

async function syncLocalRepositoryByBaseUrl(page: Page, baseUrl: string) {
    await page.goto(baseUrl + "/container_platform/appMarket/stash/wareHouse");
    const localRepositoryRow = page.locator("tbody tr")
        .filter({ hasText: "local" })
        .filter({ hasText: "http://local-harbor." })
        .first();
    await expect(localRepositoryRow).toBeVisible({ timeout: 30000 });
    await localRepositoryRow.getByRole("img", { name: "more" }).click();
    await page.getByText("同步仓库", { exact: true }).last().click();
    await expect(page.getByText(/local仓库同步成功[！!]/).last()).toBeVisible({ timeout: 60000 });
}
async function deployMonitoringDashboardFromBusinessConsole(page: Page, baseUrl: string) {
    await loginBusinessClusterConsole(page, baseUrl);
    await page.getByText("应用市场").click();
    await page.getByRole("link", { name: "应用列表" }).click();
    await syncLocalRepositoryByBaseUrl(page, baseUrl);
    await page.goto(baseUrl + "/container_platform/appMarket/marketCategory");
    await page.getByText(MONITORING_DASHBOARD.componentName).click();
    await page.getByRole("button", { name: "部 署" }).click();
    await page.getByRole("textbox", { name: "* 应用名称" }).fill(MONITORING_DASHBOARD.appName);
    await fillVisiblePasswordFields(page);
    await page.getByRole("button", { name: "确 定" }).click();
    await expect(page.getByText("部署成功，即将跳转扩展组件管理页面")).toBeVisible({ timeout: 120_000 });
    await expect(page.getByRole("cell", { name: "check-circle 部署成功" })).toBeVisible({ timeout: 120_000 });
}
async function waitClusterStatusWithRefresh(
    page: Page,
    clusterName: string,
    statusPattern: RegExp,
    timeoutMs = installTimeout,
) {
    await expect
        .poll(
            
            async () => {
                await page.getByRole("menuitem", { name: "集群生命周期管理" }).getByRole("link").click();
                await page.waitForURL(/\/container_platform\/cluster\/cluster/);
                const searchBox = page.getByRole("searchbox", { name: "搜索集群名称" });
                await expect(searchBox).toBeVisible({ timeout: 30_000 });
                await searchBox.fill(clusterName, { timeout: 30_000 });
                await searchBox.press("Enter");
                return (await page.getByText(statusPattern).count()) > 0;
            },
            { timeout: timeoutMs, intervals: [20_000, 30_000, 60_000] },
        )
        .toBe(true);
}

async function waitClusterAbnormalWithRefresh(page: Page, clusterName: string, timeoutMs = installTimeout) {
    await waitClusterStatusWithRefresh(page, clusterName, /异常/, timeoutMs);
}

async function waitClusterHealthyWithRefresh(page: Page, clusterName: string, timeoutMs = installTimeout) {
    await waitClusterStatusWithRefresh(page, clusterName, /健康/, timeoutMs);
}

async function waitClusterStatusText(page: Page, clusterName: string, pattern: RegExp, timeoutMs = installTimeout) {
    const escapedClusterName = clusterName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const rowPattern = new RegExp(`${escapedClusterName}.*${pattern.source}`);
    await expect(page.getByRole("row", { name: rowPattern })).toBeVisible({ timeout: timeoutMs });
}

async function verifyClusterDeleting(page: Page, clusterName: string) {
    const deletingPattern = new RegExp(`${clusterName}.*删除中`);
    await expect(page.getByRole("row", { name: deletingPattern })).toBeVisible({ timeout: validateTimeout });
}

async function scaleUpClusterByOneNode(page: Page, clusterName: string, scaleNodeInfo: NodeInfoEntry) {
    await page.getByRole("link", { name: clusterName, exact: true }).click();
    await page.getByRole("button", { name: "plus 新增节点" }).click();
    await scaleUpSetNodeInfo(page, 0, scaleNodeInfo);
    await scaleUpCheckNodeInfo(page);
    await expect(page.getByText(/一键校验成功，请创建/)).toBeVisible({ timeout: validateTimeout });
    await scaleUpConfirm(page);
    await closeLogModal(page);
}

async function scaleDownNode(page: Page, clusterName: string, nodeName: string) {
    await page.getByRole("link", { name: clusterName, exact: true }).click();
    await page.getByRole("cell", { name: nodeName }).click();
    await page.locator('.cluster-table-cell.cluster-table-cell-fix-right.cluster-table-cell-fix-right-first.cluster-table-cell-row-hover > .cluster-space > .cluster-space-item > .anticon > svg').click();
    await scaleDownConfirm(page);
    await expect(page.getByText(/缩容中，时间较长请耐心等待|删除节点成功/)).toBeVisible({ timeout: validateTimeout });
    await closeLogModal(page);

}

function requireEnv(name: string): string {
    const value = process.env[name];
    if (!value) {
        throw new Error(`${name} is required`);
    }
    return value;
}

function getGuideUrl(): string {
    return requireEnv("TEST_GUIDEURL");
}

function getManagerUrl(): string {
    return requireEnv("TEST_MANAGERURL");
}

let currentTestClusterName = '';
let currentTestBaseUrl = getManagerUrl();

async function goToClusterListPage(page: Page, baseUrl = getManagerUrl()) {
    await page.goto(baseUrl + "/container_platform/cluster/cluster");
    if (page.url().includes(urls.loginUrl)) {
        await login(page, process.env.TEST_USERNAME, process.env.TEST_PASSWORD);
    }
}

async function cleanupCurrentTestCluster(page: Page, fallback = false, baseUrl = currentTestBaseUrl) {
    if (!currentTestClusterName) {
        return;
    }

    const clusterName = currentTestClusterName;

    try {
        await goToClusterListPage(page, baseUrl);
        await deleteCluster(page, clusterName);
        await verifyClusterDeleting(page, clusterName);
        await verifyClusterDeleted(page, clusterName);
        currentTestClusterName = '';
    } catch (error) {
        if (!fallback) {
            throw error;
        }
        console.warn(`[cluster-status] afterEach 删除失败: ${clusterName}`);
    } finally {
        if (fallback) {
            currentTestClusterName = '';
        }
    }
}

test.beforeEach(async ({ page }, testInfo) => {
    currentTestBaseUrl = testInfo.tags.includes("@post-init") ? getGuideUrl() : getManagerUrl();
    await page.goto(currentTestBaseUrl + urls.loginUrl);
    await login(page, process.env.TEST_USERNAME, process.env.TEST_PASSWORD);
    await page.getByRole('menuitem', { name: '集群生命周期管理' }).getByRole('link').click();
});

test.afterEach(async ({ page }) => {
    await cleanupCurrentTestCluster(page, false);
});

test.describe("集群状态验证", { tag: ["@online"] }, () => {
    test("管理集群安装业务集群成功", { tag: ["@with-management-cluster", "@cluster-status"] }, async ({ page }) => {
        test.setTimeout(totalTimeout);
        currentTestClusterName = `${TEST_CLUSTER_NAME}-management-installed-service`;

        await test.step("管理集群安装业务集群，安装中状态持续可见，最终变为健康", async () => {
            await fillClusterInfo(page, currentTestClusterName, node1Info);
            await createCluster(page);
            await verifyClusterInstallingWithException(page, currentTestClusterName);
            await verifyClusterHealthy(page, currentTestClusterName);
        });

        await test.step("删除业务集群，删除中后最终集群不存在", async () => {
            await cleanupCurrentTestCluster(page);
        });
    });

    test("引导集群安装业务集群成功", { tag: ["@post-init", "@cluster-status"] }, async ({ page }) => {
        test.setTimeout(totalTimeout);
        currentTestClusterName = `${TEST_CLUSTER_NAME}-bootstrap-installed-service`;
        currentTestBaseUrl = getGuideUrl();

        await test.step("通过引导集群安装业务集群，安装中状态持续可见，最终变为健康", async () => {
            await fillClusterInfo(page, currentTestClusterName, node1Info);
            await createCluster(page);
            await verifyClusterInstallingWithException(page, currentTestClusterName);
            await verifyClusterHealthy(page, currentTestClusterName);
        });
        await test.step("删除业务集群，删除中后最终集群不存在", async () => {
            await cleanupCurrentTestCluster(page);
        });
    });

    test("业务集群扩容成功", { tag: ["@with-management-cluster", "@cluster-status"] }, async ({ page }) => {
        test.setTimeout(totalTimeout);
        currentTestClusterName = `${TEST_CLUSTER_NAME}-service-scaled-up`;
        await test.step("创建业务集群并等待健康", async () => {
            await fillClusterInfo(page, currentTestClusterName, node1Info);
            await createCluster(page);
            await verifyClusterInstallingWithException(page, currentTestClusterName);
            await verifyClusterHealthy(page, currentTestClusterName);
        });
        await test.step("对业务集群进行扩容，扩容成功后节点出现在集群中", async () => {
            await page.getByRole("link", { name: currentTestClusterName, exact: true }).click();
            await page.getByRole("button", { name: "plus 新增节点" }).click();
            await scaleUpSetNodeInfo(page, 0, node2Info);
            await scaleUpCheckNodeInfo(page);
            await expect(page.getByText("一键校验成功，请创建")).toBeVisible({ timeout: validateTimeout });
            await scaleUpConfirm(page);
            await expect(page.getByText(/扩容中，时间较长请耐心等待|新增节点成功/)).toBeVisible();
            await closeLogModal(page);
            await verifyClusterScaledUp(page, currentTestClusterName, [node2Info.name]);
        });
        await test.step("删除业务集群，删除中后最终集群不存在", async () => {
            await cleanupCurrentTestCluster(page);
        });
    });

    test("业务集群扩容失败后删除失败节点恢复健康", { tag: ["@with-management-cluster", "@cluster-status"] }, async ({ page }) => {
        test.setTimeout(totalTimeout);
        currentTestClusterName = `${TEST_CLUSTER_NAME}-service-scaled-up-failed`;
        await test.step("创建业务集群并等待健康", async () => {
            await fillClusterInfo(page, currentTestClusterName, node1Info);
            await createCluster(page);
            await verifyClusterInstallingWithException(page, currentTestClusterName);
            await verifyClusterHealthy(page, currentTestClusterName);
        });
        try {
            await test.step("先执行扩容，再注入 calico-kube-controllers 镜像异常", async () => {
                await scaleUpClusterByOneNode(page, currentTestClusterName, node2Info);
                await verifyClusterScaledUp(page, currentTestClusterName, [node2Info.name]);
                await updateDeploymentContainerImage({
                    host: node1Info.ip,
                    username: node1Info.userName,
                    password: node1Info.password,
                    namespace: CALICO_KUBE_CONTROLLERS.namespace,
                    deploymentName: CALICO_KUBE_CONTROLLERS.deploymentName,
                    containerName: CALICO_KUBE_CONTROLLERS.containerName,
                    image: `${CALICO_KUBE_CONTROLLERS.goodImage.split(":")[0]}:${BAD_IMAGE_TAG}`,
                    waitRollout: false,
                });
                await triggerDeploymentImagePullError({
                    host: node1Info.ip,
                    username: node1Info.userName,
                    password: node1Info.password,
                    namespace: CALICO_KUBE_CONTROLLERS.namespace,
                    deploymentName: CALICO_KUBE_CONTROLLERS.deploymentName,
                    kubeconfig: MANAGEMENT_KUBECONFIG,
                });
                await waitClusterAbnormalWithRefresh(page, currentTestClusterName);
            });
            await test.step("恢复 calico-kube-controllers 镜像并删除扩容失败节点，集群恢复健康", async () => {
                await updateDeploymentContainerImage({
                    host: node1Info.ip,
                    username: node1Info.userName,
                    password: node1Info.password,
                    namespace: CALICO_KUBE_CONTROLLERS.namespace,
                    deploymentName: CALICO_KUBE_CONTROLLERS.deploymentName,
                    containerName: CALICO_KUBE_CONTROLLERS.containerName,
                    image: CALICO_KUBE_CONTROLLERS.goodImage,
                });
                await scaleDownNode(page, currentTestClusterName, node2Info.name);
                await verifyClusterScaledDown(page, currentTestClusterName, [node2Info.name]);
            });
        } finally {
            await updateDeploymentContainerImage({
                host: node1Info.ip,
                username: node1Info.userName,
                password: node1Info.password,
                namespace: CALICO_KUBE_CONTROLLERS.namespace,
                deploymentName: CALICO_KUBE_CONTROLLERS.deploymentName,
                containerName: CALICO_KUBE_CONTROLLERS.containerName,
                image: CALICO_KUBE_CONTROLLERS.goodImage,
            });
        }
        await test.step("删除业务集群，删除中后最终集群不存在", async () => {
            await cleanupCurrentTestCluster(page);
        });
    });

    test("业务集群安装失败后修复pod镜像恢复健康", { tag: ["@with-management-cluster", "@cluster-status"] }, async ({ page }) => {
            test.setTimeout(totalTimeout);
            currentTestClusterName = `${TEST_CLUSTER_NAME}-service-installed-failed`;
            await test.step("管理集群安装业务集群并等待健康", async () => {
                await fillClusterInfo(page, currentTestClusterName, node1Info);
                await createCluster(page);
                await verifyClusterInstallingWithException(page, currentTestClusterName);
                await verifyClusterHealthy(page, currentTestClusterName);
            });
            await test.step("将 calico-kube-controllers 镜像改为 notfound，触发集群异常", async () => {
                await updateDeploymentContainerImage({
                    host: node1Info.ip,
                    username: node1Info.userName,
                    password: node1Info.password,
                    namespace: CALICO_KUBE_CONTROLLERS.namespace,
                    deploymentName: CALICO_KUBE_CONTROLLERS.deploymentName,
                    containerName: CALICO_KUBE_CONTROLLERS.containerName,
                    image: `${CALICO_KUBE_CONTROLLERS.goodImage.split(":")[0]}:${BAD_IMAGE_TAG}`,
                    waitRollout: false,
                });
                await waitClusterAbnormalWithRefresh(page, currentTestClusterName);
            });
            await test.step("修复 calico-kube-controllers 镜像后，集群状态恢复为健康", async () => {
                await updateDeploymentContainerImage({
                    host: node1Info.ip,
                    username: node1Info.userName,
                    password: node1Info.password,
                    namespace: CALICO_KUBE_CONTROLLERS.namespace,
                    deploymentName: CALICO_KUBE_CONTROLLERS.deploymentName,
                    containerName: CALICO_KUBE_CONTROLLERS.containerName,
                    image: CALICO_KUBE_CONTROLLERS.goodImage,
                });
                await verifyClusterHealthy(page, currentTestClusterName);
            });
            await test.step("删除业务集群，删除中后最终集群不存在", async () => {
                await cleanupCurrentTestCluster(page);
            });
    });

    test("业务集群核心组件异常后恢复健康", { tag: ["@with-management-cluster", "@cluster-status"] }, async ({ page }) => {
            test.setTimeout(totalTimeout);
            currentTestClusterName = `${TEST_CLUSTER_NAME}-service-core-component-failed`;
            await test.step("创建业务集群并选择 openFuyao Cores，最终状态健康", async () => {
                await fillClusterInfo(page, currentTestClusterName, node1Info, undefined, addonOptions.openfuyaoCores);
                await createCluster(page);
                await verifyClusterInstallingWithException(page, currentTestClusterName);
                await verifyClusterHealthy(page, currentTestClusterName);
            });
            const targetComponent = BUSINESS_CORE_COMPONENTS[0];
            let originImage: string;
            await test.step("注入核心组件镜像异常", async () => {
                originImage = await breakBusinessComponentImage(node1Info, targetComponent);

                await triggerDeploymentImagePullError({
                    host: node1Info.ip,
                    username: node1Info.userName,
                    password: node1Info.password,
                    namespace: targetComponent.namespace,
                    deploymentName: targetComponent.deploymentName,
                    kubeconfig: MANAGEMENT_KUBECONFIG,
                });
            });
            await test.step("等待并验证集群状态变为异常", async () => {
                await waitClusterAbnormalWithRefresh(page, currentTestClusterName);
                await verifyClusterAbnormal(page, currentTestClusterName);
            });
            await test.step("恢复核心组件镜像，验证集群状态恢复健康", async () => {
                try {
                    await restoreBusinessComponentImage(node1Info, targetComponent, originImage);
                } finally {
                    await verifyClusterHealthy(page, currentTestClusterName);
                }
            });
            await test.step("删除业务集群，删除中后最终集群不存在", async () => {
                await cleanupCurrentTestCluster(page);
            });
    });

    test("集群状态验证-业务集群缩容失败", { tag: ["@skip-temporarily", "@with-management-cluster", "@cluster-status"] }, async ({ page }) => {
        test.setTimeout(totalTimeout);
        const nodeName = node2Info.name;
        currentTestClusterName = `${TEST_CLUSTER_NAME}-service-scale-down-failed`;
        let originalImage = '';

        await test.step("创建业务集群并等待健康", async () => {
            await fillClusterInfo(page, currentTestClusterName, node1Info);
            await createCluster(page);
            await verifyClusterInstallingWithException(page, currentTestClusterName);
            await verifyClusterHealthy(page, currentTestClusterName);
        });

        await test.step("先扩容一个节点作为缩容目标", async () => {
            await scaleUpClusterByOneNode(page, currentTestClusterName, node2Info);
            await verifyClusterScaledUp(page, currentTestClusterName, [nodeName]);
        });

        await test.step("注入 calico-kube-controllers 组件异常，触发缩容失败", async () => {
            originalImage = await breakBusinessComponentImage(node1Info, CALICO_KUBE_CONTROLLERS);
            await triggerDeploymentImagePullError({
                host: node1Info.ip,
                username: node1Info.userName,
                password: node1Info.password,
                namespace: CALICO_KUBE_CONTROLLERS.namespace,
                deploymentName: CALICO_KUBE_CONTROLLERS.deploymentName,
                kubeconfig: MANAGEMENT_KUBECONFIG,
            });
            await scaleDownNode(page, currentTestClusterName, nodeName);
            await waitClusterStatusText(page, currentTestClusterName, /缩容失败/);
        });

        await test.step("恢复 calico-kube-controllers 组件镜像", async () => {
            await restoreBusinessComponentImage(node1Info, CALICO_KUBE_CONTROLLERS, originalImage);
        });

        await test.step("删除业务集群，删除中后最终集群不存在", async () => {
            await cleanupCurrentTestCluster(page);
        });
    });

    test("集群状态验证-业务集群删除失败", { tag: ["@skip-temporarily", "@with-management-cluster", "@cluster-status"] }, async ({ page }) => {
        test.setTimeout(totalTimeout);
        currentTestClusterName = `${TEST_CLUSTER_NAME}-service-delete-failed`;
        let originalImage = '';

        await test.step("创建业务集群并等待健康", async () => {
            await fillClusterInfo(page, currentTestClusterName, node1Info);
            await createCluster(page);
            await verifyClusterInstallingWithException(page, currentTestClusterName);
            await verifyClusterHealthy(page, currentTestClusterName);
        });

        await test.step("注入 calico-kube-controllers 组件异常，触发删除失败", async () => {
            originalImage = await breakBusinessComponentImage(node1Info, CALICO_KUBE_CONTROLLERS);
            await triggerDeploymentImagePullError({
                host: node1Info.ip,
                username: node1Info.userName,
                password: node1Info.password,
                namespace: CALICO_KUBE_CONTROLLERS.namespace,
                deploymentName: CALICO_KUBE_CONTROLLERS.deploymentName,
                kubeconfig: MANAGEMENT_KUBECONFIG,
            });
            await cleanupCurrentTestCluster(page, true);
            await waitClusterStatusText(page, currentTestClusterName, /删除失败/);
        });

        await test.step("恢复 calico-kube-controllers 组件镜像并完成清理", async () => {
            await restoreBusinessComponentImage(node1Info, CALICO_KUBE_CONTROLLERS, originalImage);
            await cleanupCurrentTestCluster(page);
        });
    });

    test("集群状态验证-扩展组件对集群状态不影响", { tag: ["@with-management-cluster", "@cluster-status"] }, async ({ page }) => {
        test.setTimeout(totalTimeout);
        currentTestClusterName = `${TEST_CLUSTER_NAME}-service-addon-installed`;
        const businessConsoleUrl = `https://${node1Info.ip}:${createClusterData.fuyaoPort}`;
        let addonImage = '';

        await test.step("创建业务集群", async () => {
            await fillClusterInfo(page, currentTestClusterName, node1Info, undefined, addonOptions.openfuyaoCores);
            await createCluster(page);
            await verifyClusterInstallingWithException(page, currentTestClusterName);
            await verifyClusterHealthy(page, currentTestClusterName);
        });

        try {
            await test.step("从业务集群前端部署 monitoring-dashboard 扩展组件", async () => {
                await deployMonitoringDashboardFromBusinessConsole(page, businessConsoleUrl);
            });

            await test.step("将扩展组件 pod 镜像修改为异常并等待 5 分钟", async () => {
                addonImage = await breakBusinessComponentImage(node1Info, MONITORING_DASHBOARD);
                await triggerDeploymentImagePullError({
                    host: node1Info.ip,
                    username: node1Info.userName,
                    password: node1Info.password,
                    namespace: MONITORING_DASHBOARD.namespace,
                    deploymentName: MONITORING_DASHBOARD.deploymentName,
                    kubeconfig: MANAGEMENT_KUBECONFIG,
                });
                await page.waitForTimeout(5 * 60 * 1000);
            });

            await test.step("通过管理集群前端确认业务集群状态仍保持健康", async () => {
                await goToClusterListPage(page);
                await waitClusterHealthyWithRefresh(page, currentTestClusterName);
                await verifyClusterHealthy(page, currentTestClusterName);
            });
        } finally {
            if (addonImage) {
                await restoreBusinessComponentImage(node1Info, MONITORING_DASHBOARD, addonImage).catch((error) => {
                    console.warn(`[cluster-status] 恢复 monitoring-dashboard 镜像失败: ${error}`);
                });
            }
        }

        await test.step("清理业务集群", async () => {
            await cleanupCurrentTestCluster(page);
        });
    });
    test("管理集群核心组件异常后恢复", { tag: ["@skip-temporarily", "@with-management-cluster", "@cluster-status"] }, async ({ page }) => {
        test.setTimeout(totalTimeout);
        // #224: 自管 BC UI 名继承引导侧名；已有管理集群环境用 MGMT_CLUSTER_NAME / MGMT_CLUSTER_SELF_NAME
        const managementClusterName = resolveManagementClusterSelfDisplayNameFromEnv();
        const targetComponent = CALICO_KUBE_CONTROLLERS;
        const managementNodeInfo = guideNodeInfo;
        let originImage: string;
        await test.step("确认管理集群当前健康", async () => {
            await waitClusterHealthyWithRefresh(page, managementClusterName);
            await verifyClusterHealthy(page, managementClusterName);
        });
        await test.step("注入管理集群 calico-kube-controllers 镜像异常", async () => {
            originImage = await breakBusinessComponentImage(managementNodeInfo, targetComponent);

            await triggerDeploymentImagePullError({
                host: managementNodeInfo.ip,
                username: managementNodeInfo.userName,
                password: managementNodeInfo.password,
                namespace: targetComponent.namespace,
                deploymentName: targetComponent.deploymentName,
                kubeconfig: MANAGEMENT_KUBECONFIG,
            });
        });
        await test.step("等待并验证集群状态变为异常", async () => {
            await waitClusterAbnormalWithRefresh(page, managementClusterName);
            await verifyClusterAbnormal(page, managementClusterName);
        });
        await test.step("修复管理集群 calico-kube-controllers pod，验证管理集群状态恢复健康", async () => {
            try {
                await restoreBusinessComponentImage(managementNodeInfo, targetComponent, originImage);
            } finally {
                await verifyClusterHealthy(page, managementClusterName);
            }
        });
    });
});
