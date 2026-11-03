import type { Page } from "@playwright/test";
import { expect, test } from "@playwright/test";
import { addonOptions, createClusterData, urls, nodeInfo, timeouts } from "./constants/common";
import type { NodeInfoFields } from "./utils/clusterManage";
import { login} from "./utils/rbac";
import {
    verifyClusterHealthy,
    verifyClusterInstalling,
    verifyClusterUpgrading,
    verifyClusterDeleted,
    verifyClusterHealthOrInstallFailed,
    verifyDeploymentStatus,
    verifyDeploymentImage,
} from "./utils/verify";
import { deleteCluster, goToCreateClusterPage, setClusterInfo, setNodeInfo, setAdvancedConfig, createCluster, startClusterUpgrade } from "./utils/clusterManage";
import { updateBkeControllerManagerManifests, updateInstallerServiceRemotePaths } from "./utils/upgrade-pre";

const { node1Info, guideNodeInfo } = nodeInfo;
const { totalTimeout, validateTimeout } = timeouts;
const TEST_CLUSTER_NAME ='test-cluster';
const OFFLINE_IMAGE_DOMAIN = createClusterData.offlineImageDomain;
const bootstrapIp = guideNodeInfo.ip;
const TARGET_MANIFESTS_IMAGE = "cr.openfuyao.cn/openfuyao/bke-manifests:1.2.1";
const ORIGINAL_MANIFESTS_IMAGE = "cr.openfuyao.cn/openfuyao/bke-manifests:latest";
const TEMP_REMOTE_PATCH_PATH = "https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/";
const TEMP_REMOTE_DEPLOY_PATH = "https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/v26.03-rc.3/";
const ORIGINAL_REMOTE_PATCH_PATH = "https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/";
const ORIGINAL_REMOTE_DEPLOY_PATH = "https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/latest/";
const DEPLOYMENT_NAME = "bke-controller-manager";
const DEPLOYMENT_STATUS = "Active";
const EXPECTED_PROVIDER_BKE_IMAGE_TAG = "1.2.2";
const GUIDE_KUBECONFIG = "/etc/rancher/k3s/k3s.yaml";
const MANAGEMENT_KUBECONFIG = "/etc/kubernetes/admin.conf";

// 填写集群信息，可指定集群名称、fuyao版本等
async function fillClusterInfo(
    page: Page,
    clusterName: string,
    masterInfo: NodeInfoFields,
    fuyaoVersion?: string,
    addon?: string,
    useBootstrap = false,
) {
    await goToCreateClusterPage(page);
    await setClusterInfo(page, clusterName, fuyaoVersion);
    await setNodeInfo(page, 'master', 0, masterInfo);

    if (useBootstrap) {
        await page.getByRole('combobox', { name: '镜像仓库地址' }).click();
        await page.getByTitle(OFFLINE_IMAGE_DOMAIN).click();
        await page.getByRole('combobox', { name: '镜像仓库IP地址' }).click();
        await page.getByTitle(bootstrapIp).click();
        await page.getByRole('combobox', { name: 'Chart仓库地址' }).click();
        const chartAddress = `http://${bootstrapIp}:${createClusterData.chartPort}`;
        await page.getByTitle(chartAddress).click();
    } else {
        await setAdvancedConfig(page, addon ?? addonOptions.noAddon);
    }

    await page.getByRole('button', { name: '一键校验' }).click();
    await expect(page.getByText('一键校验成功，请创建')).toBeVisible({ timeout: validateTimeout });
};

// 将集群升级至指定版本
async function upgradeCluster(page: Page, clusterName: string, upgradeVersion: string) {
    await startClusterUpgrade(page, clusterName, upgradeVersion);
    await verifyClusterUpgrading(page, clusterName);
    await verifyClusterHealthy(page, clusterName);
};

test.beforeEach(async ({ page }) => {
    await page.goto(process.env.TEST_GUIDEURL + urls.loginUrl)
    await login(page, process.env.TEST_USERNAME, process.env.TEST_PASSWORD);
});

test.afterEach(async ({ page }) => {
    const clusterCell = page.getByRole('cell', { name: TEST_CLUSTER_NAME }).first();

    // 引导集群场景通常已在集群列表页；管理集群场景可能停留在详情页
    let clusterVisible = false;
    try {
        await clusterCell.waitFor({ state: 'visible', timeout: 5000 });
        clusterVisible = true;
    } catch {
        try {
            const lifecycleLink = page.getByRole('link', { name: '集群生命周期管理' });
            if (await lifecycleLink.isVisible({ timeout: 3000 })) {
                await lifecycleLink.click();
                await clusterCell.waitFor({ state: 'visible', timeout: 5000 });
                clusterVisible = true;
            }
        } catch {
            // 忽略跳转异常，统一在下方判断是否跳过清理
        }
    }
    if (!clusterVisible) {
        console.warn(`afterEach cleanup skip cluster delete: cluster ${TEST_CLUSTER_NAME} not found`);
        return;
    }
    try {
        await verifyClusterHealthOrInstallFailed(page, TEST_CLUSTER_NAME);
        await deleteCluster(page, TEST_CLUSTER_NAME);
        await verifyClusterDeleted(page, TEST_CLUSTER_NAME);
    } catch (error) {
        // 清理阶段不应再导致二次失败
        console.warn(
            'afterEach cleanup skip cluster delete:',
            error instanceof Error ? error.message : String(error),
        );
    }
});

test.describe('在线安装业务集群后升级', { tag: ['@online'] }, async () => {
    test.describe.configure({ mode: 'serial' });
    /**
     * 测试用例1: 在线引导集群安装后升级
     * 步骤：引导集群前端安装业务集群(v26.03-rc.3) -> 升级到v26.03
     * 预期结果：业务集群安装成功 -> 升级成功
     */
    test("引导集群初始化并安装v26.03-rc.3版本业务集群，然后升级到v26.03版本", { tag: ['@post-init', '@skip-temporarily'] }, async ({ page }) => {
        test.setTimeout(totalTimeout);

        const clusterName = TEST_CLUSTER_NAME;

        await updateBkeControllerManagerManifests({
            host: guideNodeInfo.ip,
            username: guideNodeInfo.userName,
            password: guideNodeInfo.password,
            manifestsImage: TARGET_MANIFESTS_IMAGE,
            kubeconfig: GUIDE_KUBECONFIG,
        });
        await updateInstallerServiceRemotePaths({
            host: guideNodeInfo.ip,
            username: guideNodeInfo.userName,
            password: guideNodeInfo.password,
            remotePatchPath: TEMP_REMOTE_PATCH_PATH,
            remoteDeployPath: TEMP_REMOTE_DEPLOY_PATH,
            kubeconfig: GUIDE_KUBECONFIG,
        });

        try {
            await test.step("引导集群前端界面安装v26.03-rc.3版本业务集群", async () => {
                const fuyaoVersion = "v26.03-rc.3";
                await fillClusterInfo(page, clusterName, node1Info, fuyaoVersion)
                await createCluster(page);
                await verifyClusterInstalling(page, clusterName);
                await verifyClusterHealthy(page, clusterName);
                await expect(page.getByRole('cell', { name: 'v26.03-rc.3' }).first()).toBeVisible();
            });

            await test.step("业务集群升级到v26.03版本", async () => {
                const upgradeVersion = "v26.03";
                await upgradeCluster(page, clusterName, upgradeVersion);
                await expect(page.getByRole('cell', { name: 'v26.03' }).first()).toBeVisible();
            });
        } finally {
            await updateInstallerServiceRemotePaths({
                host: guideNodeInfo.ip,
                username: guideNodeInfo.userName,
                password: guideNodeInfo.password,
                remotePatchPath: ORIGINAL_REMOTE_PATCH_PATH,
                remoteDeployPath: ORIGINAL_REMOTE_DEPLOY_PATH,
                kubeconfig: GUIDE_KUBECONFIG,
            });
            await updateBkeControllerManagerManifests({
                host: guideNodeInfo.ip,
                username: guideNodeInfo.userName,
                password: guideNodeInfo.password,
                manifestsImage: ORIGINAL_MANIFESTS_IMAGE,
                kubeconfig: GUIDE_KUBECONFIG,
            });
        }
    });
    /**
     * 测试用例2: 在线管理集群安装后升级
     * 步骤：管理集群前端安装业务集群(v26.03-rc.3) -> 升级到v26.03 -> 验证bke-controller-manager状态和镜像版本
     * 预期结果：业务集群安装成功 -> 升级成功 -> deployment状态为Active且镜像版本为1.2.2
     */
    test("管理集群存在情况下安装v26.03-rc.3版本业务集群，然后升级到v26.03版本", { tag: ['@with-management-cluster', '@skip-temporarily'] }, async ({ page }) => {
        test.setTimeout(totalTimeout);
        const clusterName = TEST_CLUSTER_NAME;

        await updateBkeControllerManagerManifests({
            host: guideNodeInfo.ip,
            username: guideNodeInfo.userName,
            password: guideNodeInfo.password,
            manifestsImage: TARGET_MANIFESTS_IMAGE,
            kubeconfig: MANAGEMENT_KUBECONFIG,
        });
        await updateInstallerServiceRemotePaths({
            host: guideNodeInfo.ip,
            username: guideNodeInfo.userName,
            password: guideNodeInfo.password,
            remotePatchPath: TEMP_REMOTE_PATCH_PATH,
            remoteDeployPath: TEMP_REMOTE_DEPLOY_PATH,
            kubeconfig: MANAGEMENT_KUBECONFIG,
        });

        try {
            await test.step("管理集群前端界面安装v26.03-rc.3版本业务集群", async () => {
                const fuyaoVersion = "v26.03-rc.3";
                await page.getByRole('link', { name: '集群生命周期管理' }).click();
                await fillClusterInfo(page, clusterName, node1Info, fuyaoVersion);
                await createCluster(page);
                await verifyClusterInstalling(page, clusterName);
                await verifyClusterHealthy(page, clusterName);
                await expect(page.getByRole('cell', { name: 'v26.03-rc.3' }).first()).toBeVisible();
            });

            await test.step("业务集群升级到v26.03版本", async () => {
                const upgradeVersion = "v26.03";
                await upgradeCluster(page, clusterName, upgradeVersion);
                await expect(page.getByRole('cell', { name: 'v26.03' }).first()).toBeVisible();
            });

            await test.step('验证bke-controller-manager状态和镜像版本', async () => {
                await page.getByRole('link', { name: clusterName }).click();
                await verifyDeploymentStatus(page, DEPLOYMENT_NAME, DEPLOYMENT_STATUS);
                await verifyDeploymentImage(page, DEPLOYMENT_NAME, EXPECTED_PROVIDER_BKE_IMAGE_TAG);
            });
        } finally {
            await updateInstallerServiceRemotePaths({
                host: guideNodeInfo.ip,
                username: guideNodeInfo.userName,
                password: guideNodeInfo.password,
                remotePatchPath: ORIGINAL_REMOTE_PATCH_PATH,
                remoteDeployPath: ORIGINAL_REMOTE_DEPLOY_PATH,
                kubeconfig: MANAGEMENT_KUBECONFIG,
            });
            await updateBkeControllerManagerManifests({
                host: guideNodeInfo.ip,
                username: guideNodeInfo.userName,
                password: guideNodeInfo.password,
                manifestsImage: ORIGINAL_MANIFESTS_IMAGE,
                kubeconfig: MANAGEMENT_KUBECONFIG,
            });
        }
    });
});
