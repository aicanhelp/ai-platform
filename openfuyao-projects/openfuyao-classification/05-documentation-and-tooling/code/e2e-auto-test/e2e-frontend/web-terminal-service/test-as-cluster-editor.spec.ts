import { test, expect, Page, BrowserContext } from '@playwright/test';
import { TEST_FUYAOURL, TEST_USERNAME, TEST_PASSWORD, CLUSTER_EDITOR_USERNAME, CLUSTER_EDITOR_INITIAL_PASSWORD, CLUSTER_EDITOR_PASSWORD, INVITE_TEST_USERNAME, INVITE_TEST_PASSWORD } from './consts';
import { createUser, inviteUserWithRole } from '../auth/utils/user-management';
import { login, firstLoginChangePassword, logout } from '../utils/common';
import { getVisibleLocator, openContainerTerminal, waitForPodListReady } from './utils/terminal-helpers';
import { deleteUserFromUserManagement, removeClusterUserFromMembers } from './utils/cleanup-helpers';
import { newContext } from '@/utils/common';

const timestamp = Date.now().toString().slice(-8);
const randomStr = Math.random().toString(36).substring(2, 6);
const testPodName = `e2e-pod-${timestamp}-${randomStr}`;

test.describe('cluster-editor测试', () => {
    let page: Page;
    let context: BrowserContext;
    let createdClusterEditorUsername: string;

    test.beforeAll(async ({ browser }) => {
        // 创建浏览器上下文
        context = await newContext(browser);
        page = await context.newPage();


        // ========== 步骤1: 用平台管理员登录 ==========
        await page.goto(`${TEST_FUYAOURL}/oauth2/auth/login/fuyaoPasswordProvider`);
        await page.waitForLoadState('networkidle');
        
        if (page.url().includes('/oauth2/auth/login/fuyaoPasswordProvider')) {
            await page.locator('#username').fill(TEST_USERNAME!);
            await page.locator('#password').fill(TEST_PASSWORD!);
            await page.locator('#login-btn').click();
            await page.waitForLoadState('networkidle');
        }
        
        await expect(page.getByText(TEST_USERNAME!)).toBeVisible();
        await expect(page).toHaveURL(/\/container_platform\/overview/);

        // ========== 步骤2: 创建 cluster-editor 用户 ==========
        createdClusterEditorUsername = await createUser(
            page,
            CLUSTER_EDITOR_USERNAME!,
            CLUSTER_EDITOR_INITIAL_PASSWORD!,
            'platform-regular'
        );

        // ========== 步骤3: 在集群中绑定 cluster-editor 角色 ==========
        await page.goto(`${TEST_FUYAOURL}/container_platform/clusterMember`);
        await page.waitForLoadState('networkidle');

        await inviteUserWithRole(page, createdClusterEditorUsername, 'cluster-editor');
        await expect(page.getByText(`邀请${createdClusterEditorUsername}成员成功！`)).toBeVisible({ timeout: 5000 });

        await page.goto(`${TEST_FUYAOURL}/container_platform/workload/pod`);
        await page.waitForSelector('table', { timeout: 10000 });

        // ========== 步骤4: 退出当前用户 ==========
        await logout(page);
        await expect(page).toHaveURL(/\/auth\/login|\/oauth2\/auth\/login/, { timeout: 10000 });

        // ========== 步骤5: 用 cluster-editor 登录并修改初始密码 ==========
        await login(page, createdClusterEditorUsername, CLUSTER_EDITOR_INITIAL_PASSWORD!);
        await firstLoginChangePassword(page, CLUSTER_EDITOR_PASSWORD!);

        // 验证登录成功
        await expect(page.getByText(createdClusterEditorUsername)).toBeVisible({ timeout: 10000 });
    });

    test.afterAll(async () => {
        // ========== 清理: 删除 cluster-editor 用户 ==========

        // 退出 cluster-editor 账号
        await logout(page);
        await expect(page).toHaveURL(/\/auth\/login|\/oauth2\/auth\/login/, { timeout: 10000 });

        // 用平台管理员登录
        await page.goto(`${TEST_FUYAOURL}/oauth2/auth/login/fuyaoPasswordProvider`);
        await page.waitForLoadState('networkidle');
        
        if (page.url().includes('/oauth2/auth/login/fuyaoPasswordProvider')) {
            await page.locator('#username').fill(TEST_USERNAME!);
            await page.locator('#password').fill(TEST_PASSWORD!);
            await page.locator('#login-btn').click();
            await page.waitForLoadState('networkidle');
        }
        
        await expect(page.getByText(TEST_USERNAME!)).toBeVisible();
        await expect(page).toHaveURL(/\/container_platform\/overview/);

        // 步骤1: 从集群成员中移除用户
        await page.goto(`${TEST_FUYAOURL}/container_platform/clusterMember`);
        await page.waitForLoadState('networkidle');

        // 点击用户的操作按钮
        await removeClusterUserFromMembers(page, createdClusterEditorUsername);

        // 步骤2: 从用户管理中删除用户
        await page.goto(`${TEST_FUYAOURL}/user_manage/user`);
        await page.waitForLoadState('networkidle');

        await deleteUserFromUserManagement(page, createdClusterEditorUsername);

        // 等待成功消息消失并刷新页面
        await expect(page.getByText(`删除用户${createdClusterEditorUsername}成功！`)).not.toBeVisible({ timeout: 5000 });
        await page.reload();
        await page.waitForLoadState('networkidle');

        // 关闭页面和上下文
        await page.close();
        await context.close();
    })

    test.afterEach(async()=>{
        await page.goto(`${TEST_FUYAOURL}`);
    })

    async function hasTerminalAccessDeniedMessage() {
        const deniedSignals = [
            page.getByText('当前用户尚无访问权限'),
            page.getByText('请联系管理员添加用户权限'),
            page.getByText('无权限'),
            page.locator('.container-platform-modal-body').filter({ hasText: '终端' }).first(),
            page.getByRole('dialog').filter({ hasText: '终端' }).first(),
        ];

        for (const locator of deniedSignals) {
            if (await locator.isVisible().catch(() => false)) {
                return true;
            }
        }

        return false;
    }

    async function expectContainerTerminalAccessDenied() {
        const terminalPanel = page.locator('.xterm').first();

        await page.goto(`${TEST_FUYAOURL}/container_platform/workload/pod`);
        await page.waitForSelector('table', { timeout: 10000 });
        await page.waitForLoadState('networkidle');

        for (let attempt = 0; attempt < 3; attempt++) {
            await waitForPodListReady(page);
            const rows = page.locator('tbody tr');
            const rowCount = await rows.count();

            for (let i = 0; i < rowCount; i++) {
                const row = rows.nth(i);
                const rowText = await row.textContent();
                if (!rowText || !rowText.includes('Running')) {
                    continue;
                }

                const podLink = row.locator('td a').first();
                if (!await podLink.isVisible().catch(() => false)) {
                    continue;
                }

                await podLink.click();
                await page.waitForLoadState('networkidle');
                await expect(page).toHaveURL(/\/container_platform\/workload\/pod\/.+\/.+/);
                await page.waitForSelector('tr.container-platform-table-row', { timeout: 10000 });

                await openContainerTerminal(page);

                let terminalAttemptResult = 'pending';
                await expect.poll(async () => {
                    if (await hasTerminalAccessDeniedMessage()) {
                        terminalAttemptResult = 'denied';
                        return terminalAttemptResult;
                    }

                    if (await terminalPanel.isVisible().catch(() => false)) {
                        terminalAttemptResult = 'opened';
                        return terminalAttemptResult;
                    }

                    return terminalAttemptResult;
                }, {
                    timeout: 10000,
                }).not.toBe('pending').catch(() => {});

                if (terminalAttemptResult === 'denied') {
                    await expect(terminalPanel).toHaveCount(0);
                    return;
                }

                if (terminalAttemptResult === 'opened') {
                    throw new Error('cluster-editor 用户意外成功打开了容器终端');
                }

                await page.goto(`${TEST_FUYAOURL}/container_platform/workload/pod`);
                await page.waitForSelector('table', { timeout: 10000 });
                await page.waitForLoadState('networkidle');
            }

            await page.goto(`${TEST_FUYAOURL}/container_platform/workload/pod`);
            await page.waitForSelector('table', { timeout: 10000 });
            await page.waitForLoadState('networkidle');
        }

        throw new Error('未拿到明确的容器终端无权限证据');
    }

    test('【WebTerminal-019】platform-regular角色且分配cluster-editor角色的用户无权限访问容器终端', {
        tag: ['@v24.12', '@web-terminal-service'],
    }, async () => {
        await expectContainerTerminalAccessDenied();
    });

    test('【WebTerminal-030】platform-regular角色且分配cluster-editor角色的用户无权限访问集群终端', {
        tag: ['@v24.12', '@web-terminal-service'],
    }, async () => {
        const terminalPanel = page.locator('.xterm').first();

        await page.goto(`${TEST_FUYAOURL}/container_platform/overview`);
        await page.waitForLoadState('networkidle');

        const terminalButton = await getVisibleLocator([
            page.getByRole('button', { name: 'kubectl终端工具' }),
            page.getByText('kubectl终端工具', { exact: true }),
            page.getByRole('img', { name: 'code' }),
            page.locator('[aria-label="code"]').first(),
        ]);

        await terminalButton.scrollIntoViewIfNeeded().catch(() => {});
        await terminalButton.click();

        await expect.poll(async () => {
            if (await hasTerminalAccessDeniedMessage()) {
                return 'denied';
            }

            if (await terminalPanel.isVisible().catch(() => false)) {
                return 'opened';
            }

            return 'pending';
        }, {
            timeout: 10000,
        }).toBe('denied');

        await expect(terminalPanel).toHaveCount(0);
    });
})
