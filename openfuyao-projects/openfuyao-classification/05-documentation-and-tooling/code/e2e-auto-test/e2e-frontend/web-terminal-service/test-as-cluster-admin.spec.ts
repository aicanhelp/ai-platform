import { test, expect, Page, BrowserContext } from '@playwright/test';
import { TEST_FUYAOURL, TEST_USERNAME, TEST_PASSWORD, CLUSTER_ADMIN_USERNAME, CLUSTER_ADMIN_INITIAL_PASSWORD, CLUSTER_ADMIN_PASSWORD, INVITE_TEST_USERNAME, INVITE_TEST_PASSWORD } from './consts';
import { createUser, inviteUserWithRole, editClusterUser } from '../auth/utils/user-management';
import { login, firstLoginChangePassword, logout } from '../utils/common';
import {
    openClusterTerminal,
    waitForTerminalReady,
    openAvailableContainerTerminal,
} from './utils/terminal-helpers';
import { deleteUserFromUserManagement, removeClusterUserFromMembers } from './utils/cleanup-helpers';
import { newContext } from '@/utils/common';

test.describe('cluster-admin测试', () => {
    let page: Page;
    let context: BrowserContext;
    let createdClusterAdminUsername: string;

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

        // ========== 步骤2: 创建集群管理员用户 ==========
        createdClusterAdminUsername = await createUser(
            page,
            CLUSTER_ADMIN_USERNAME!,
            CLUSTER_ADMIN_INITIAL_PASSWORD!,
            'platform-regular'
        );

        // ========== 步骤3: 在集群中绑定集群管理员角色 ==========
        await page.goto(`${TEST_FUYAOURL}/container_platform/clusterMember`);
        await page.waitForLoadState('networkidle');
        
        await inviteUserWithRole(page, createdClusterAdminUsername, 'cluster-admin');
        await expect(page.getByText(`邀请${createdClusterAdminUsername}成员成功！`)).toBeVisible({ timeout: 5000 });

        // ========== 步骤4: 退出当前用户 ==========
        await logout(page);
        await expect(page).toHaveURL(/\/auth\/login|\/oauth2\/auth\/login/, { timeout: 10000 });

        // ========== 步骤5: 用集群管理员登录并修改初始密码 ==========
        await login(page, createdClusterAdminUsername, CLUSTER_ADMIN_INITIAL_PASSWORD!);
        await firstLoginChangePassword(page, CLUSTER_ADMIN_PASSWORD!);

        // 验证登录成功
        await expect(page.getByText(createdClusterAdminUsername)).toBeVisible({ timeout: 10000 });
    });

    test.afterAll(async () => {
        // ========== 清理: 删除集群管理员用户 ==========

        // 退出集群管理员账号
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
        await removeClusterUserFromMembers(page, createdClusterAdminUsername);

        // 步骤2: 从用户管理中删除用户
        await deleteUserFromUserManagement(page, createdClusterAdminUsername);

        // 等待成功消息消失并刷新页面
        await page.reload();
        await page.waitForLoadState('networkidle');

        // 关闭页面和上下文
        await page.close();
        await context.close();
    });

    test.afterEach(async()=>{
        await page.goto(`${TEST_FUYAOURL}`);
    })

    test('【WebTerminal-018】platform-regular角色且分配cluster-admin角色的用户能正常访问容器终端', {
        tag: ['@v24.12', '@web-terminal-service'],
    }, async () => {
        const terminalPanel = page.locator('.xterm').first();
        const terminalTitle = page.locator('.terminal_header').first();

        await page.goto(`${TEST_FUYAOURL}/container_platform/workload/pod`);
        await page.waitForSelector('table', { timeout: 10000 });
        await page.waitForLoadState('networkidle');

        await openAvailableContainerTerminal(page, terminalPanel);

        const containerRow = page.locator('tr.container-platform-table-row').first();
        const containerName = ((await containerRow.locator('td').first().textContent()) || '').replace(/\s+/g, ' ').trim();

        await expect(terminalPanel).toBeVisible({ timeout: 30000 });
        await expect(terminalTitle).toContainText('容器：');
        await expect(terminalTitle).toContainText(containerName);
    });

    test('【WebTerminal-029】platform-regular角色且分配cluster-admin角色的用户能正常访问集群终端', {
        tag: ['@v24.12', '@web-terminal-service'],
    }, async () => {
        const terminalPanel = page.locator('.xterm').first();
        const terminalTitle = page.locator('.terminal_header').first();
        const clusterSelectionItem = page.locator('.container-platform-select-selection-item[title]').first();

        const currentClusterName = ((await clusterSelectionItem.getAttribute('title')) || (await clusterSelectionItem.textContent()) || '').replace(/\s+/g, ' ').trim();

        await openClusterTerminal(page);
        await expect(terminalPanel).toBeVisible({ timeout: 30000 });
        await waitForTerminalReady(page);
        await expect(terminalTitle).toContainText(`集群：${currentClusterName}`);
    });
})
