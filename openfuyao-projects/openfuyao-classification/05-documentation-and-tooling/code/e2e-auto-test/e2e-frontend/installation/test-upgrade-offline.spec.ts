import { readFile } from "fs/promises";
import type { Page } from "@playwright/test";
import { expect, test } from "@playwright/test";
import { addonOptions, createClusterData, urls, nodeInfo, timeouts } from "./constants/common";
import type { NodeInfoFields } from "./utils/clusterManage";
import { login, logout } from "./utils/rbac";
import {
    verifyClusterHealthy,
    verifyClusterInstalling,
    verifyClusterUpgrading,
    verifyClusterDeleted,
    verifyClusterHealthOrInstallFailed,
    verifyDeploymentStatus,
    verifyDeploymentImage,
} from "./utils/verify";
import {
    deleteCluster,
    goToCreateClusterPage,
    setClusterInfo,
    setNodeInfo,
    setAdvancedConfig,
    createCluster,
    startClusterUpgrade,
} from "./utils/clusterManage";

const { node1Info, guideNodeInfo } = nodeInfo;
const { totalTimeout, validateTimeout } = timeouts;
const TEST_CLUSTER_NAME = "test-cluster";
const OFFLINE_IMAGE_DOMAIN = createClusterData.offlineImageDomain;
const bootstrapIp = guideNodeInfo.ip;

const OFFLINE_INIT_VERSION_YAML_PATH = "./installation/versionYaml/v26_03_rc_3.yaml";
const OFFLINE_UPGRADE_VERSION_YAML_PATH = "./installation/versionYaml/v26_03.yaml";
const DEPLOYMENT_NAME = "bke-controller-manager";
const DEPLOYMENT_STATUS = "Active";
const EXPECTED_PROVIDER_BKE_IMAGE_TAG = "1.2.2";

async function loadYamlAsString(filePath: string) {
    return await readFile(filePath, "utf8");
}

async function uploadVersionYaml(page: Page, yamlPath: string) {
    const versionYaml = await loadYamlAsString(yamlPath);
    await page.getByRole("button", { name: "补丁上传" }).click();
    await page.getByRole("textbox").locator("div").fill(versionYaml);
    await page.getByRole("button", { name: /确\s*定/ }).click();
    await expect(page.getByText("上传成功")).toBeVisible();
}

async function fillOfflineClusterInfo(
    page: Page,
    clusterName: string,
    masterInfo: NodeInfoFields,
    fuyaoVersion?: string,
    addon?: string,
    useBootstrap = true,
) {
    await goToCreateClusterPage(page);
    await setClusterInfo(page, clusterName, fuyaoVersion);
    await setNodeInfo(page, "master", 0, masterInfo);

    if (useBootstrap) {
        await page.getByRole("combobox", { name: "镜像仓库地址" }).click();
        await page.getByTitle(OFFLINE_IMAGE_DOMAIN).click();
        await page.getByRole("combobox", { name: "镜像仓库IP地址" }).click();
        await page.getByTitle(bootstrapIp).click();
        await page.getByRole("combobox", { name: "Chart仓库地址" }).click();
        const chartAddress = `http://${bootstrapIp}:${createClusterData.chartPort}`;
        await page.getByTitle(chartAddress).click();
    } else {
        await setAdvancedConfig(page, addon ?? addonOptions.noAddon);
    }

    await page.getByRole("button", { name: "一键校验" }).click();
    await expect(page.getByText("一键校验成功，请创建")).toBeVisible({ timeout: validateTimeout });
}

async function upgradeCluster(page: Page, clusterName: string, upgradeVersion: string) {
    await startClusterUpgrade(page, clusterName, upgradeVersion);
    await verifyClusterUpgrading(page, clusterName);
    await verifyClusterHealthy(page, clusterName);
}

async function verifyClusterApiProviderImageTag(page: Page, expectedTag: string) {
    await page.getByRole("row", { name: TEST_CLUSTER_NAME }).getByRole("link").first().click();
    const imagePattern = new RegExp(`cluster-api-provider-bke:${expectedTag}`);
    await expect(page.getByText(imagePattern)).toBeVisible({ timeout: 60 * 1000 });
    await page.goBack();
    await expect(page.getByRole("link", { name: "集群生命周期管理" })).toBeVisible();
}

test.beforeEach(async ({ page }) => {
    await page.goto(process.env.TEST_GUIDEURL + urls.loginUrl);
    await login(page, process.env.TEST_USERNAME, process.env.TEST_PASSWORD);
});

test.afterEach(async ({ page }) => {
    const clusterCell = page.getByRole("cell", { name: TEST_CLUSTER_NAME }).first();

    // 引导集群场景通常已在集群列表页；管理集群场景可能停留在详情页
    let clusterVisible = false;
    try {
        await clusterCell.waitFor({ state: "visible", timeout: 5000 });
        clusterVisible = true;
    } catch {
        try {
            const lifecycleLink = page.getByRole("link", { name: "集群生命周期管理" });
            if (await lifecycleLink.isVisible({ timeout: 3000 })) {
                await lifecycleLink.click();
                await clusterCell.waitFor({ state: "visible", timeout: 5000 });
                clusterVisible = true;
            }
        } catch {
            // 忽略跳转异常，统一在下方判断是否跳过清理
        }
    }

    try {
        if (!clusterVisible) {
            console.warn(`afterEach cleanup skip cluster delete: cluster ${TEST_CLUSTER_NAME} not found`);
            return;
        }

        await verifyClusterHealthOrInstallFailed(page, TEST_CLUSTER_NAME);
        await deleteCluster(page, TEST_CLUSTER_NAME);
        await verifyClusterDeleted(page, TEST_CLUSTER_NAME);
    } finally {
        await logout(page);
    }
});

test.describe("离线安装业务集群后升级", async () => {
    test.describe.configure({ mode: "serial" });
    /**
     * 测试用例1: 离线引导集群安装后升级
     * 步骤：上传v26.03-rc.3版本yaml -> 引导集群前端安装业务集群(v26.03-rc.3) -> 上传v26.03版本yaml -> 升级到v26.03
     * 预期结果：yaml上传成功 -> 业务集群安装成功 -> 升级成功
     */
    test("离线引导集群初始化并安装v26.03-rc.3版本业务集群，然后升级到v26.03版本", { tag: ["@offline", "@post-init", '@skip-temporarily'] }, async ({ page }) => {
        test.setTimeout(totalTimeout);
        const clusterName = TEST_CLUSTER_NAME;

        await test.step("上传v26.03-rc.3版本yaml", async () => {
            await uploadVersionYaml(page, OFFLINE_INIT_VERSION_YAML_PATH);
        });

        await test.step("引导集群前端界面安装安装v26.03-rc.3版本业务集群", async () => {
            const fuyaoVersion = "v26.03-rc.3";
            await fillOfflineClusterInfo(page, clusterName, node1Info, fuyaoVersion);
            await createCluster(page);
            await verifyClusterInstalling(page, clusterName);
            await verifyClusterHealthy(page, clusterName);
            await expect(page.getByRole("cell", { name: "v26.03-rc.3" }).first()).toBeVisible();
            await verifyClusterApiProviderImageTag(page, "latest");
        });

        await test.step("上传v26.03版本yaml", async () => {
            await uploadVersionYaml(page, OFFLINE_UPGRADE_VERSION_YAML_PATH);
        });

        await test.step("业务集群升级到v26.03版本", async () => {
            const upgradeVersion = "v26.03";
            await upgradeCluster(page, clusterName, upgradeVersion);
            await expect(page.getByRole("cell", { name: "v26.03" }).first()).toBeVisible();
            await verifyClusterApiProviderImageTag(page, "v26.03");
        });
    });
    /**
     * 测试用例2: 离线管理集群安装后升级
     * 步骤：上传v26.03-rc.3版本yaml -> 管理集群前端安装业务集群(v26.03-rc.3) -> 上传v26.03版本yaml -> 升级到v26.03 -> 验证bke-controller-manager状态和镜像版本
     * 预期结果：yaml上传成功 -> 业务集群安装成功 -> 升级成功 -> deployment状态为Active且镜像版本为1.2.2
     */
    test("离线管理集群安装v26.03-rc.3版本业务集群，然后升级到v26.03版本", { tag: ["@offline", "@with-management-cluster", '@skip-temporarily'] }, async ({ page }) => {
        test.setTimeout(totalTimeout);
        const clusterName = TEST_CLUSTER_NAME;

        await test.step("上传v26.03-rc.3版本yaml", async () => {
            await uploadVersionYaml(page, OFFLINE_INIT_VERSION_YAML_PATH);
        });

        await test.step("管理集群前端界面安装v26.03-rc.3版本业务集群", async () => {
            const fuyaoVersion = "v26.03-rc.3";
            await page.getByRole("link", { name: "集群生命周期管理" }).click();
            await fillOfflineClusterInfo(page, clusterName, node1Info, fuyaoVersion, addonOptions.openfuyaoCores, false);
            await createCluster(page);
            await verifyClusterInstalling(page, clusterName);
            await verifyClusterHealthy(page, clusterName);
            await expect(page.getByRole("cell", { name: "v26.03-rc.3" }).first()).toBeVisible();
            await verifyClusterApiProviderImageTag(page, "latest");
        });

        await test.step("上传v26.03版本yaml", async () => {
            await uploadVersionYaml(page, OFFLINE_UPGRADE_VERSION_YAML_PATH);
        });

        await test.step("业务集群升级到v26.03版本", async () => {
            const upgradeVersion = "v26.03";
            await upgradeCluster(page, clusterName, upgradeVersion);
            await expect(page.getByRole("cell", { name: "v26.03" }).first()).toBeVisible();
            await verifyClusterApiProviderImageTag(page, "v26.03");
        });

        await test.step("验证bke-controller-manager状态和镜像版本", async () => {
            await page.getByRole("link", { name: clusterName }).click();
            await verifyDeploymentStatus(page, DEPLOYMENT_NAME, DEPLOYMENT_STATUS);
            await verifyDeploymentImage(page, DEPLOYMENT_NAME, EXPECTED_PROVIDER_BKE_IMAGE_TAG);
        });
    });
});
