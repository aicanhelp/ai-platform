import { expect, test } from "@playwright/test";
import { addonOptions, urls, nodeInfo, timeouts } from "./constants/common";
import { login, logout } from "./utils/rbac";
import {
    verifyClusterHealthy,
    verifyClusterInstalling,
    verifyClusterScaledUp,
    verifyClusterScaledDown,
    verifyClusterDeleted,
    verifyClusterHealthOrInstallFailed,
} from "./utils/verify";
import {
    goToCreateClusterPage,
    setClusterInfo,
    addNode,
    setNodeInfo,
    setAdvancedConfig,
    validateCluster,
    createCluster,
    scaleUpSetNodeInfo,
    scaleUpCheckNodeInfo,
    scaleUpConfirm,
    scaleDownConfirm,
    closeLogModal,
    deleteCluster,
} from "./utils/clusterManage";

const { guideNodeInfo, node1Info, node2Info, node3Info } = nodeInfo;
const { totalTimeout, validateTimeout } = timeouts;
let clusterName: string;
const TEST_CLUSTER_NAME_PREFIX = 'test-cluster';

test.beforeEach(async ({ page }) => {
    await test.step('已登录admin', async () => {
        await page.goto(process.env.TEST_GUIDEURL + urls.loginUrl);
        await login(page, process.env.TEST_USERNAME, process.env.TEST_PASSWORD);
        await goToCreateClusterPage(page);
    });
});

test.afterEach(async ({ page }) => {
    await page.goto(process.env.TEST_GUIDEURL);
    try {
        await page.getByRole('cell', { name: clusterName }).waitFor({ state: 'visible' });
        await verifyClusterHealthOrInstallFailed(page, clusterName);
        await deleteCluster(page, clusterName);
        await verifyClusterDeleted(page, clusterName);
    } finally {
        await logout(page);
    }
})

test.describe('引导节点配置为集群节点测试', { tag: ['@post-init', '@online'] }, () => {
    test.setTimeout(totalTimeout);

    /**
     * 测试用例1: 支持引导节点配置为Master节点(1M1N)
     * 步骤：登录管理面-配置引导节点ip为Master节点-创建1M1N集群(引导节点为其中的master)-扩容新工作节点-删除工作节点-删除集群
     * 预期结果：集群创建成功且pod正常-扩容成功-节点删除弹出提示-节点删除成功-删除集群弹出提示-删除成功
     */
    test('支持引导节点配置为Master节点(1M1N)', async ({ page }) => {
        clusterName = TEST_CLUSTER_NAME_PREFIX + '-1m1n-boot-master';
        await test.step('创建1M1N集群 (引导节点为Master)', async () => {
            // 设置引导节点为Master，另一个节点为Node
            await setClusterInfo(page, clusterName);
            await setNodeInfo(page, 'master', 0, guideNodeInfo);
            await addNode(page, 'node');
            await setNodeInfo(page, 'node', 0, node1Info);
            await setAdvancedConfig(page, addonOptions.noAddon);
            await validateCluster(page);
            await expect(page.getByText('一键校验成功，请创建')).toBeVisible({ timeout: validateTimeout });
            await expect(page.getByRole('button', { name: '创 建' })).toBeEnabled();
            await createCluster(page);
            await verifyClusterInstalling(page, clusterName);
            await verifyClusterHealthy(page, clusterName);
        });

        await test.step('扩容新工作节点', async () => {
            await page.getByRole('link', { name: clusterName, exact: true }).click();
            await page.getByRole('button', { name: 'plus 新增节点' }).click();
            await scaleUpSetNodeInfo(page, 0, node2Info);
            await scaleUpCheckNodeInfo(page);
            await expect(page.getByText('一键校验成功，请创建')).toBeVisible({ timeout: validateTimeout });
            await scaleUpConfirm(page);
            await expect(page.getByText(/扩容中，时间较长请耐心等待|新增节点成功/)).toBeVisible();
            await closeLogModal(page);
            await verifyClusterScaledUp(page, clusterName, [node2Info.name]);
        });

        await test.step('删除工作节点', async () => {
            await page.getByRole('row', { name: node2Info.name }).locator('[data-icon="delete"]').click();
            await scaleDownConfirm(page);
            await expect(page.getByText(/缩容中，时间较长请耐心等待|删除节点成功/)).toBeVisible();
            await closeLogModal(page);
            await verifyClusterScaledDown(page, clusterName, [node2Info.name]);
        });
    });

    /**
     * 测试用例2: 支持引导节点配置为Master节点(3M)
     * 步骤：登录管理面-配置引导节点ip为Master节点-创建3M集群(引导节点为其中一个master)-扩容节点-删除工作节点-删除集群
     * 预期结果：集群创建成功且pod正常-扩容成功-非引导节点删除成功-删除集群弹出提示-删除成功
     */
    test('支持引导节点配置为Master节点(3M)', async ({ page }) => {
        clusterName = TEST_CLUSTER_NAME_PREFIX + '-3m-boot-master';
        await test.step('创建3M集群 (引导节点为Master之一)', async () => {
            // 设置引导节点为第一个Master，另外两个节点也为Master
            await setClusterInfo(page, clusterName);
            await setNodeInfo(page, 'master', 0, guideNodeInfo);
            await addNode(page, 'master');
            await setNodeInfo(page, 'master', 1, node1Info);
            await addNode(page, 'master');
            await setNodeInfo(page, 'master', 2, node2Info);
            await setAdvancedConfig(page, addonOptions.noAddon);
            await validateCluster(page);
            await expect(page.getByText('一键校验成功，请创建')).toBeVisible({ timeout: validateTimeout });
            await expect(page.getByRole('button', { name: '创 建' })).toBeEnabled();
            await createCluster(page);
            await verifyClusterInstalling(page, clusterName);
            await verifyClusterHealthy(page, clusterName);
        });

        await test.step('扩容工作节点', async () => {
            await page.getByRole('link', { name: clusterName, exact: true }).click();
            await page.getByRole('button', { name: 'plus 新增节点' }).click();
            await scaleUpSetNodeInfo(page, 0, node3Info);
            await scaleUpCheckNodeInfo(page);
            await expect(page.getByText('一键校验成功，请创建')).toBeVisible({ timeout: validateTimeout });
            await scaleUpConfirm(page);
            await expect(page.getByText(/扩容中，时间较长请耐心等待|新增节点成功/)).toBeVisible();
            await closeLogModal(page);
            await verifyClusterScaledUp(page, clusterName, [node3Info.name]);
        });

        await test.step('删除工作节点', async () => {
            await page.getByRole('row', { name: node3Info.name }).locator('[data-icon="delete"]').click();
            await scaleDownConfirm(page);
            await expect(page.getByText(/缩容中，时间较长请耐心等待|删除节点成功/)).toBeVisible();
            await closeLogModal(page);
            await verifyClusterScaledDown(page, clusterName, [node3Info.name]);
        });
    });

    /**
     * 测试用例3: 支持引导节点配置为Node节点(1M1N)
     * 步骤：登录管理面-创建1M1N集群-扩容引导节点为工作节点-删除引导节点工作节点-删除集群
     * 预期结果：集群创建成功且pod正常-扩容成功-节点删除弹出提示（包含引导节点）-节点删除成功-删除集群-删除成功
     */
    test('支持引导节点配置为Node节点(1M1N)', async ({ page }) => {
        clusterName = TEST_CLUSTER_NAME_PREFIX + '-1m1n-boot-node';
        await test.step('创建1M1N集群 (不含引导节点)', async () => {
            // 使用两个非引导节点
            await setClusterInfo(page, clusterName);
            await setNodeInfo(page, 'master', 0, node1Info);
            await addNode(page, 'node');
            await setNodeInfo(page, 'node', 0, node2Info);
            await setAdvancedConfig(page, addonOptions.noAddon);
            await validateCluster(page);
            await expect(page.getByText('一键校验成功，请创建')).toBeVisible({ timeout: validateTimeout });
            await expect(page.getByRole('button', { name: '创 建' })).toBeEnabled();
            await createCluster(page);
            await verifyClusterInstalling(page, clusterName);
            await verifyClusterHealthy(page, clusterName);
        });

        await test.step('扩容引导节点为工作节点', async () => {
            await page.getByRole('link', { name: clusterName }).click();
            await page.getByRole('button', { name: 'plus 新增节点' }).click();
            await scaleUpSetNodeInfo(page, 0, guideNodeInfo);
            await scaleUpCheckNodeInfo(page);
            await expect(page.getByText('一键校验成功，请创建')).toBeVisible({ timeout: validateTimeout });
            await scaleUpConfirm(page);
            await expect(page.getByText(/扩容中，时间较长请耐心等待|新增节点成功/)).toBeVisible();
            await closeLogModal(page);
            await verifyClusterScaledUp(page, clusterName, [guideNodeInfo.name]);
        });

        await test.step('删除引导节点工作节点', async () => {
            await page.getByRole('row', { name: guideNodeInfo.name }).locator('[data-icon="delete"]').click();
            await scaleDownConfirm(page);
            await expect(page.getByText(/缩容中，时间较长请耐心等待|删除节点成功/)).toBeVisible();
            await closeLogModal(page);
            await verifyClusterScaledDown(page, clusterName, [guideNodeInfo.name]);
        });
    });

    /**
     * 测试用例4: 支持引导节点配置为Node节点(3M)
     * 步骤：登录管理面-创建3M集群-扩容引导节点为工作节点-删除工作节点-删除集群
     * 预期结果：集群创建成功且pod正常-扩容成功-节点删除弹出提示（包含引导节点）-节点删除成功-删除集群-删除成功
     */
    test('支持引导节点配置为Node节点(3M)', async ({ page }) => {
        clusterName = TEST_CLUSTER_NAME_PREFIX + '-3m-boot-node';
        await test.step('创建3M集群 (不含引导节点)', async () => {
            // 使用三个非引导节点
            await setClusterInfo(page, clusterName);
            await setNodeInfo(page, 'master', 0, node1Info);
            await addNode(page, 'master');
            await setNodeInfo(page, 'master', 1, node2Info);
            await addNode(page, 'master');
            await setNodeInfo(page, 'master', 2, node3Info);
            await setAdvancedConfig(page, addonOptions.noAddon);
            await validateCluster(page);
            await createCluster(page);
            await verifyClusterInstalling(page, clusterName);
            await verifyClusterHealthy(page, clusterName);
        });

        await test.step('扩容引导节点为工作节点', async () => {
            await page.getByRole('link', { name: clusterName, exact: true }).click();
            await page.getByRole('button', { name: 'plus 新增节点' }).click();
            await scaleUpSetNodeInfo(page, 0, guideNodeInfo);
            await scaleUpCheckNodeInfo(page);
            await expect(page.getByText('一键校验成功，请创建')).toBeVisible({ timeout: validateTimeout });
            await scaleUpConfirm(page);
            await expect(page.getByText(/扩容中，时间较长请耐心等待|新增节点成功/)).toBeVisible();
            await closeLogModal(page);
            await verifyClusterScaledUp(page, clusterName, [guideNodeInfo.name]);
        });

        await test.step('删除引导节点工作节点', async () => {
            await page.getByRole('row', { name: guideNodeInfo.name }).locator('[data-icon="delete"]').click();
            await scaleDownConfirm(page);
            await expect(page.getByText(/缩容中，时间较长请耐心等待|删除节点成功/)).toBeVisible();
            await closeLogModal(page);
            await verifyClusterScaledDown(page, clusterName, [guideNodeInfo.name]);
        });
    });
});
