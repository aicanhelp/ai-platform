import { expect, test } from "@playwright/test";
import { addonOptions, urls, nodeInfo, timeouts } from "./constants/common";
import { login } from "./utils/rbac";
import {
    addNode,
    setClusterInfo,
    setNodeInfo,
    setAdvancedConfig,
    createCluster,
    deleteCluster,
    startClusterUpgrade,
    resetDeploymentImage,
} from "./utils/clusterManage";
import {
    verifyClusterHealthy,
    verifyClusterInstalling,
    verifyClusterUpgrading,
    verifyDeploymentStatus,
    verifyDeploymentImage,
    verifyClusterDeleted,
} from "./utils/verify";

const { node1Info, node2Info, node3Info } = nodeInfo;
const { totalTimeout } = timeouts;
const TEST_CLUSTER_NAME_PRFIX = 'test-cluster';

// 管理节点地址
const managerUrl = process.env.TEST_MANAGERURL || '';

// 升级相关常量
const UPGRADE_VERSION = {
    BASE: 'v25.09',                     // 创建集群时的基础版本
    TARGET: 'v25.09.998',               // 升级目标版本（正常升级）
    TARGET_MISSING_CONFIG: 'v25.09.999' // 升级目标版本（配置缺失场景）
};

const DEPLOYMENT = {
    NAME: 'bke-controller-manager',
    NAMESPACE: 'bke-system',
    STATUS: 'Active',                     // Deployment 页面显示的状态
    EXPECTED_IMAGE_AFTER_UPGRADE: '1.0.1',  // 升级后期望的镜像版本
    EXPECTED_IMAGE_MISSING_CONFIG: 'latest' // 配置缺失时期望保持的镜像版本
};

test.beforeEach(async ({ page }) => {
    // 登录管理节点
    await page.goto(managerUrl + urls.loginUrl);
    await login(page, process.env.TEST_USERNAME, process.env.TEST_PASSWORD);
});

test.afterEach(async ({ page }) => {
    // 每个用例结束后，将管理节点的镜像重置为 latest，确保下一个用例能正常触发升级
    await test.step('重置管理节点镜像版本', async () => {
        await page.goto(managerUrl);
        await verifyDeploymentStatus(page, DEPLOYMENT.NAME, DEPLOYMENT.STATUS);
        await resetDeploymentImage(page, DEPLOYMENT.NAME, 'latest');
    });
});

/**
 * 测试用例1: 单Master业务集群升级
 * 步骤：创建1M集群(v25.09) -> 升级到v25.09.998 -> 验证bke-controller-manager状态和镜像版本 -> 删除集群
 * 预期结果：集群创建成功 -> 升级成功 -> deployment状态为Active且镜像版本为1.0.1 -> 删除成功
 */
test.describe('管理集群升级测试', { tag: ['@with-management-cluster', '@skip-temporarily', '@online'] }, () => {
    test.describe.configure({ mode: 'serial' });
    test.setTimeout(totalTimeout);

    test('单 Master 基础版本升级到补丁版本', async ({ page }) => {
        const clusterName = TEST_CLUSTER_NAME_PRFIX + '-upgrade-1m';

        await test.step('进入集群生命周期管理页面', async () => {
            await page.getByRole('link', { name: '集群生命周期管理' }).click();
            await expect(page.getByRole('button', { name: 'plus 新增集群' })).toBeVisible();
        });

        await test.step('创建1M集群 (版本v25.09)', async () => {
            await test.step('配置集群信息', async () => {
                await setClusterInfo(page, clusterName, UPGRADE_VERSION.BASE);
                await setNodeInfo(page, 'master', 0, node1Info);
            });
            await test.step('设置高级配置', async () => {
                await setAdvancedConfig(page, addonOptions.noAddon);
            });
            await createCluster(page);
            await verifyClusterInstalling(page, clusterName);
            await verifyClusterHealthy(page, clusterName);
        });

        await test.step('升级集群到v25.09.998', async () => {
            await startClusterUpgrade(page, clusterName, UPGRADE_VERSION.TARGET);
            await verifyClusterUpgrading(page, clusterName);
            // 等待升级完成，集群恢复健康状态
            await verifyClusterHealthy(page, clusterName);
        });

        await test.step('验证bke-controller-manager状态和镜像版本', async () => {
            // 进入集群详情
            await page.getByRole('link', { name: clusterName }).click();
            // 验证deployment状态为Active
            await verifyDeploymentStatus(page, DEPLOYMENT.NAME, DEPLOYMENT.STATUS);
            // 验证镜像版本已更新为1.0.1
            await verifyDeploymentImage(page, DEPLOYMENT.NAME, DEPLOYMENT.EXPECTED_IMAGE_AFTER_UPGRADE);
        });

        await test.step('删除集群', async () => {
            // 返回集群列表页
            await page.getByRole('link', { name: '集群生命周期管理' }).click();
            await deleteCluster(page, clusterName);
            await verifyClusterDeleted(page, clusterName);
        });
    });

    /**
     * 测试用例2: 多Master业务集群升级 (3M)
     * 步骤：创建3M集群(v25.09) -> 升级到v25.09.998 -> 验证bke-controller-manager状态和镜像版本 -> 删除集群
     */
    test('多Master(3节点)基础版本升级到补丁版本', async ({ page }) => {
        const clusterName = TEST_CLUSTER_NAME_PRFIX + '-upgrade-3m';

        await test.step('进入集群生命周期管理页面', async () => {
            await page.getByRole('link', { name: '集群生命周期管理' }).click();
            await expect(page.getByRole('button', { name: 'plus 新增集群' })).toBeVisible();
        });

        await test.step('创建3M集群 (版本v25.09)', async () => {
            await test.step('配置集群信息', async () => {
                await setClusterInfo(page, clusterName, UPGRADE_VERSION.BASE);
                await setNodeInfo(page, 'master', 0, node1Info);
                // 添加第二个Master
                await addNode(page, 'master');
                await setNodeInfo(page, 'master', 1, node2Info);
                // 添加第三个Master
                await addNode(page, 'master');
                await setNodeInfo(page, 'master', 2, node3Info);
            });
            await test.step('设置高级配置', async () => {
                await setAdvancedConfig(page, addonOptions.noAddon);
            });
            await createCluster(page);
            await verifyClusterInstalling(page, clusterName);
            await verifyClusterHealthy(page, clusterName);
        });

        await test.step('升级集群到v25.09.998', async () => {
            await startClusterUpgrade(page, clusterName, UPGRADE_VERSION.TARGET);
            await verifyClusterUpgrading(page, clusterName);
            await verifyClusterHealthy(page, clusterName);
        });

        await test.step('验证bke-controller-manager状态和镜像版本', async () => {
            await page.getByRole('link', { name: clusterName }).click();
            await verifyDeploymentStatus(page, DEPLOYMENT.NAME, DEPLOYMENT.STATUS);
            await verifyDeploymentImage(page, DEPLOYMENT.NAME, DEPLOYMENT.EXPECTED_IMAGE_AFTER_UPGRADE);
        });

        await test.step('删除集群', async () => {
            await page.getByRole('link', { name: '集群生命周期管理' }).click();
            await deleteCluster(page, clusterName);
            await verifyClusterDeleted(page, clusterName);
        });
    });

    /**
     * 测试用例3: 单Master集群升级配置缺失
     * 步骤：创建1M集群(v25.09) -> 升级到v25.09.999(无provider-bke配置) -> 验证deployment仍为旧版本且正常运行 -> 删除集群
     * 预期结果：升级流程正常完成，但provider-bke镜像保持为latest，deployment状态为Active
     */
    test('单 Master ConfigMap 缺失或解析失败', async ({ page }) => {
        const clusterName = TEST_CLUSTER_NAME_PRFIX + '-upgrade-missing';

        await test.step('进入集群生命周期管理页面', async () => {
            await page.getByRole('link', { name: '集群生命周期管理' }).click();
            await expect(page.getByRole('button', { name: 'plus 新增集群' })).toBeVisible();
        });

        await test.step('创建1M集群 (版本v25.09)', async () => {
            await test.step('配置集群信息', async () => {
                await setClusterInfo(page, clusterName, UPGRADE_VERSION.BASE);
                await setNodeInfo(page, 'master', 0, node1Info);
            });
            await test.step('设置高级配置', async () => {
                await setAdvancedConfig(page, addonOptions.noAddon);
            });
            await createCluster(page);
            await verifyClusterInstalling(page, clusterName);
            await verifyClusterHealthy(page, clusterName);
        });

        await test.step('升级集群到v25.09.999 (配置缺失版本)', async () => {
            await startClusterUpgrade(page, clusterName, UPGRADE_VERSION.TARGET_MISSING_CONFIG);
            await verifyClusterUpgrading(page, clusterName);
            // 等待升级完成
            await verifyClusterHealthy(page, clusterName);
        });

        await test.step('验证bke-controller-manager状态和镜像版本保持不变', async () => {
            await page.getByRole('link', { name: clusterName }).click();
            // 验证deployment状态仍为Active
            await verifyDeploymentStatus(page, DEPLOYMENT.NAME, DEPLOYMENT.STATUS);
            // 验证镜像版本仍为latest（未升级）
            await verifyDeploymentImage(page, DEPLOYMENT.NAME, DEPLOYMENT.EXPECTED_IMAGE_MISSING_CONFIG);
        });

        await test.step('删除集群', async () => {
            await page.getByRole('link', { name: '集群生命周期管理' }).click();
            await deleteCluster(page, clusterName);
            await verifyClusterDeleted(page, clusterName);
        });
    });
});