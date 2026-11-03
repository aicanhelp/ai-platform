import { expect, test } from "@playwright/test";
import { addonOptions, urls, nodeInfo, timeouts } from "./constants/common";
import { login, logout } from "./utils/rbac";
import {
    verifyClusterHealthy,
    verifyClusterInstalling,
    verifyClusterDeleted,
    verifyClusterScaledUp,
    verifyClusterScaledDown,
    checkFieldsAndValidateErrorMessages,
    verifyClusterHealthOrInstallFailed,
} from "./utils/verify";
import {
    goToCreateClusterPage,
    setClusterInfo,
    setNodeInfo,
    setAdvancedConfig,
    validateCluster,
    createCluster,
    scaleUpSetNodeInfo,
    scaleUpAddNode,
    scaleUpCheckNodeInfo,
    scaleUpConfirm,
    scaleDownConfirm,
    deleteCluster,
    closeLogModal,
} from "./utils/clusterManage";
import { newContext } from "@/utils/common";

const { node1Info, node2Info, node3Info, node4Info } = nodeInfo;
const { totalTimeout, validateTimeout } = timeouts;
const TEST_CLUSTER_NAME = 'test-cluster';

test.beforeAll(async ({ browser }) => {
    const context = await newContext(browser);
    const page = await context.newPage();
    await test.step('已登录admin', async () => {
        await page.goto(process.env.TEST_GUIDEURL + urls.loginUrl)
        await login(page, process.env.TEST_USERNAME, process.env.TEST_PASSWORD);
    });

    await test.step('创建用于扩缩容的测试集群"', async () => {
        await goToCreateClusterPage(page);
        await setClusterInfo(page, TEST_CLUSTER_NAME);
        await setNodeInfo(page, 'master', 0, node1Info);
        await setAdvancedConfig(page, addonOptions.noAddon);

        await validateCluster(page);
        await expect(page.getByText('一键校验成功，请创建')).toBeVisible({ timeout: validateTimeout });
        await expect(page.getByRole('button', { name: '创 建' })).toBeEnabled();
        await createCluster(page);
        await verifyClusterInstalling(page, TEST_CLUSTER_NAME);
        await verifyClusterHealthy(page, TEST_CLUSTER_NAME);
    });

    await logout(page)
    await context.close()
});

test.afterAll(async ({ browser }) => {
    const context = await newContext(browser);
    const page = await context.newPage();

    await page.goto(process.env.TEST_GUIDEURL + urls.loginUrl)
    await login(page, process.env.TEST_USERNAME, process.env.TEST_PASSWORD);
    try {
        await page.getByRole('cell', { name: TEST_CLUSTER_NAME }).waitFor({ state: 'visible' });
        await verifyClusterHealthOrInstallFailed(page, TEST_CLUSTER_NAME);
        await deleteCluster(page, TEST_CLUSTER_NAME);
        await verifyClusterDeleted(page, TEST_CLUSTER_NAME);
    } finally {
        await logout(page);
        await context.close();
    }
});

test.beforeEach(async ({ page }) => {
    await page.goto(process.env.TEST_GUIDEURL + urls.loginUrl);
    await login(page, process.env.TEST_USERNAME, process.env.TEST_PASSWORD);
    await page.getByRole('link', { name: TEST_CLUSTER_NAME, exact: true }).click();
    await page.getByRole('button', { name: 'plus 新增节点' }).click();
});

test.describe("集群扩缩容", { tag: ['@post-init', '@online'] }, async () => {

    test.describe('集群扩缩容操作', async () => {
        test.setTimeout(totalTimeout);

        // 创建集群时，管理节点为node1，扩容时新增节点为node2，缩容时删除node2节点
        test("【集群管理-088、089、092、093】 集群扩缩容（单节点）", async ({ page }) => {
            await test.step("集群扩容", async () => {
                await scaleUpSetNodeInfo(page, 0, node2Info);
                await scaleUpCheckNodeInfo(page);
                await expect(page.getByText('一键校验成功，请创建')).toBeVisible({ timeout: validateTimeout });
                await scaleUpConfirm(page);
                await expect(page.getByText(/扩容中，时间较长请耐心等待|新增节点成功/)).toBeVisible();
                await closeLogModal(page);
                await verifyClusterScaledUp(page, TEST_CLUSTER_NAME, [node2Info.name]);
            });

            await test.step("集群缩容", async () => {
                // 【集群管理-095】集群缩容master节点不支持缩容
                // 管理节点不支持缩容
                await page.getByRole('cell', { name: node2Info.name }).click();
                await page.locator('.cluster-table-cell.cluster-table-cell-fix-right.cluster-table-cell-fix-right-first.cluster-table-cell-row-hover > .cluster-space > .cluster-space-item > .anticon > svg').click();
                await scaleDownConfirm(page);
                await expect(page.getByText(/缩容中，时间较长请耐心等待|删除节点成功/)).toBeVisible();
                await closeLogModal(page);
                await verifyClusterScaledDown(page, TEST_CLUSTER_NAME, [node2Info.name]);
            });
        });

        // 创建集群时，管理节点为node1，扩容时新增节点为node2、node3，缩容时删除node2、node3节点
        test("【集群管理-090、094】 集群扩缩容（多节点批量）", async ({ page }) => {
            await test.step("集群扩容多节点", async () => {
                await scaleUpSetNodeInfo(page, 0, node2Info);
                await scaleUpAddNode(page);
                await scaleUpSetNodeInfo(page, 1, node3Info);
                await scaleUpCheckNodeInfo(page);
                await expect(page.getByText('一键校验成功，请创建')).toBeVisible({ timeout: validateTimeout });
                await scaleUpConfirm(page);
                await expect(page.getByText(/扩容中，时间较长请耐心等待|新增节点成功/)).toBeVisible();
                await closeLogModal(page);
                await verifyClusterScaledUp(page, TEST_CLUSTER_NAME, [node2Info.name, node3Info.name]);
            });

            await test.step("集群缩容多节点", async () => {
                await page.getByRole('row', { name: node2Info.name + ' check-circle 就绪' }).getByLabel('', { exact: true }).click();
                await page.getByRole('row', { name: node3Info.name + ' check-circle 就绪' }).getByLabel('', { exact: true }).click();
                await page.getByRole('button', { name: '删 除' }).click();
                await scaleDownConfirm(page);
                await expect(page.getByText(/缩容中，时间较长请耐心等待|删除节点成功/)).toBeVisible();
                await closeLogModal(page);
                await verifyClusterScaledDown(page, TEST_CLUSTER_NAME, [node2Info.name, node3Info.name]);
            });
        });
    });

    test.describe('集群扩容校验', async () => {
        const DUMMY_CLUSTER_NAME = 'dummy-cluster';

        test.beforeAll(async ({ browser }) => {
            // 额外创建集群，用于扩容校验中与已有集群节点重复的IP、节点名称等情况
            const context = await newContext(browser);
            const page = await context.newPage();
            await test.step('已登录admin', async () => {
                await page.goto(process.env.TEST_GUIDEURL + urls.loginUrl)
                await login(page, process.env.TEST_USERNAME, process.env.TEST_PASSWORD);
            });
            await test.step('创建 Dummy 集群"', async () => {
                // 使用 node4，和正式测试逻辑中的 node1、node2、node3 错开
                await goToCreateClusterPage(page);
                await setClusterInfo(page, DUMMY_CLUSTER_NAME);
                await setNodeInfo(page, 'master', 0, node4Info);
                await setAdvancedConfig(page, addonOptions.noAddon);

                await validateCluster(page);
                await expect(page.getByText('一键校验成功，请创建')).toBeVisible({ timeout: validateTimeout });
                await expect(page.getByRole('button', { name: '创 建' })).toBeEnabled();
                await createCluster(page);
                await verifyClusterInstalling(page, DUMMY_CLUSTER_NAME);
                await verifyClusterHealthy(page, DUMMY_CLUSTER_NAME);
            });
            await logout(page);
            await context.close();
        });

        test.afterAll(async ({ browser }) => {
            const context = await newContext(browser);
            const page = await context.newPage();
            await page.goto(process.env.TEST_GUIDEURL + urls.loginUrl)
            await login(page, process.env.TEST_USERNAME, process.env.TEST_PASSWORD);
            try {
                await page.getByRole('cell', { name: DUMMY_CLUSTER_NAME }).waitFor({ state: 'visible' });
                await verifyClusterHealthOrInstallFailed(page, DUMMY_CLUSTER_NAME);
                await deleteCluster(page, DUMMY_CLUSTER_NAME);
                await verifyClusterDeleted(page, DUMMY_CLUSTER_NAME);
            } finally {
                await logout(page);
                await context.close();
            }
        });

        test('集群扩容单个虚假IP', async ({ page }) => {
            const invalidNodeInfo = { ...node2Info, ip: '192.0.2.1' };
            await scaleUpSetNodeInfo(page, 0, invalidNodeInfo);
            await scaleUpCheckNodeInfo(page);
            await expect(page.getByText('一键校验失败')).toBeVisible({ timeout: validateTimeout });
            await expect(page.getByRole('button', { name: /确\s*定/ })).not.toBeEnabled();
        });

        test('集群扩容多个虚假IP', async ({ page }) => {
            const invalidNode1Info = { ...node2Info, ip: '192.0.2.1' };
            const invalidNode2Info = { ...node3Info, ip: '198.51.100.2' };
            await scaleUpSetNodeInfo(page, 0, invalidNode1Info);
            await scaleUpAddNode(page);
            await scaleUpSetNodeInfo(page, 1, invalidNode2Info);
            await scaleUpCheckNodeInfo(page);
            await expect(page.getByText('一键校验失败')).toBeVisible({ timeout: validateTimeout });
            await expect(page.getByRole('button', { name: /确\s*定/ })).not.toBeEnabled();
        });

        test('集群扩容扩容SSH登录失败（端口错误）', async ({ page }) => {
            const invalidNodeInfo = { ...node2Info, port: '33' };
            await scaleUpSetNodeInfo(page, 0, invalidNodeInfo);
            await scaleUpCheckNodeInfo(page);
            await expect(page.getByText('一键校验失败')).toBeVisible({ timeout: validateTimeout });
            await expect(page.getByRole('button', { name: /确\s*定/ })).not.toBeEnabled();
        });

        test('集群扩容扩容SSH登录失败（账号错误）', async ({ page }) => {
            const invalidNodeInfo = { ...node2Info, userName: 'admin1' };
            await scaleUpSetNodeInfo(page, 0, invalidNodeInfo);
            await scaleUpCheckNodeInfo(page);
            await expect(page.getByText('一键校验失败')).toBeVisible({ timeout: validateTimeout });
            await expect(page.getByRole('button', { name: /确\s*定/ })).not.toBeEnabled();
        });

        test('集群扩容扩容SSH登录失败（密码错误）', async ({ page }) => {
            const invalidNodeInfo = { ...node2Info, password: 'xxxxxx' };
            await scaleUpSetNodeInfo(page, 0, invalidNodeInfo);
            await scaleUpCheckNodeInfo(page);
            await expect(page.getByText('一键校验失败')).toBeVisible({ timeout: validateTimeout });
            await expect(page.getByRole('button', { name: /确\s*定/ })).not.toBeEnabled();
        });

        test('集群扩容管理集群下其他节点已使用该IP（单个）', async ({ page }) => {
            const invalidNodeInfo = { ...node2Info, ip: node4Info.ip };
            await scaleUpSetNodeInfo(page, 0, invalidNodeInfo);
            await scaleUpCheckNodeInfo(page);
            await expect(page.getByText('一键校验失败')).toBeVisible({ timeout: validateTimeout });
            await expect(page.getByRole('button', { name: /确\s*定/ })).not.toBeEnabled();
        });

        test('集群扩容管理集群下其他节点已使用该IP（多个）', async ({ page }) => {
            const invalidNode1Info = { ...node2Info, ip: node4Info.ip };
            const invalidNode2Info = { ...node3Info, ip: node4Info.ip };
            await scaleUpSetNodeInfo(page, 0, invalidNode1Info);
            await scaleUpAddNode(page);
            await scaleUpSetNodeInfo(page, 1, invalidNode2Info);
            await scaleUpCheckNodeInfo(page);
            await expect(page.getByText('一键校验失败')).toBeVisible({ timeout: validateTimeout });
            await expect(page.getByRole('button', { name: /确\s*定/ })).not.toBeEnabled();
        });

        test('集群扩容扩容节点之间存在相同IP时', async ({ page }) => {
            const invalidNode1Info = node2Info;
            const invalidNode2Info = { ...node3Info, ip: node2Info.ip };
            await scaleUpSetNodeInfo(page, 0, invalidNode1Info);
            await scaleUpAddNode(page);
            await scaleUpSetNodeInfo(page, 1, invalidNode2Info);
            await scaleUpCheckNodeInfo(page);
            await expect(page.getByText('一键校验失败')).toBeVisible({ timeout: validateTimeout });
            await expect(page.getByRole('button', { name: /确\s*定/ })).not.toBeEnabled();
        });

        test('集群扩容管理集群下其他节点已使用该IP与SSH登录IP失败', async ({ page }) => {
            const invalidNode1Info = { ...node2Info, ip: node4Info.ip };    // node4已被临时集群使用
            const invalidNode2Info = { ...node3Info, password: 'xxxxxx' };  // 密码错误，ssh登录失败
            await scaleUpSetNodeInfo(page, 0, invalidNode1Info);
            await scaleUpAddNode(page);
            await scaleUpSetNodeInfo(page, 1, invalidNode2Info);
            await scaleUpCheckNodeInfo(page);
            await expect(page.getByText('一键校验失败')).toBeVisible({ timeout: validateTimeout });
            await expect(page.getByRole('button', { name: /确\s*定/ })).not.toBeEnabled();
        });

        test('集群扩容扩容节点名称与已有节点相同', async ({ page }) => {
            const invalidNodeInfo = { ...node2Info, name: node1Info.name };
            await scaleUpSetNodeInfo(page, 0, invalidNodeInfo);
            await scaleUpCheckNodeInfo(page);
            await expect(page.getByText('一键校验失败')).toBeVisible({ timeout: validateTimeout });
            await expect(page.getByRole('button', { name: /确\s*定/ })).not.toBeEnabled();
        });

        test('集群扩容扩容节点名称与其他扩容节点相同', async ({ page }) => {
            const invalidNode1Info = node2Info;
            const invalidNode2Info = { ...node3Info, name: node2Info.name };
            await scaleUpSetNodeInfo(page, 0, invalidNode1Info);
            await scaleUpAddNode(page);
            await scaleUpSetNodeInfo(page, 1, invalidNode2Info);
            await scaleUpCheckNodeInfo(page);
            await expect(page.getByText('一键校验失败')).toBeVisible({ timeout: validateTimeout });
            await expect(page.getByRole('button', { name: /确\s*定/ })).not.toBeEnabled();
        });

        test('集群扩容扩容节点信息填写不完整时', async ({ page }) => {
            await scaleUpSetNodeInfo(page, 0, {});
            await scaleUpCheckNodeInfo(page);
            const rows = page.locator('form div').filter({ hasText: 'Master信息节点名称IP地址端口号用户名密码操作添加新节点' })
            await checkFieldsAndValidateErrorMessages(rows);

        });
    });
});