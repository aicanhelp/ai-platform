import { test, expect, Page, BrowserContext } from '@playwright/test';
import { TEST_FUYAOURL, TEST_USERNAME, TEST_PASSWORD, CLUSTER_VIEWER_USERNAME, CLUSTER_VIEWER_INITIAL_PASSWORD, CLUSTER_VIEWER_PASSWORD, INVITE_TEST_USERNAME, INVITE_TEST_PASSWORD } from './consts';
import { createUser, inviteUserWithRole, removeClusterUser, deleteUser } from './utils/user-management';
import { newContext, login, firstLoginChangePassword, logout } from '../utils/common';

test.describe('cluster-viewer测试', () => {
    let page: Page;
    let context: BrowserContext;
    let createdClusterViewerUsername: string;
    let createdInviteTestUsername: string;

    test.beforeAll(async ({ browser }) => {
        // 创建浏览器上下文
        context = await newContext(browser);
        page = await context.newPage();

        // ========== 步骤1: 用平台管理员登录 ==========
        await page.goto(`${TEST_FUYAOURL}`);
        await page.waitForLoadState('networkidle');
        await login(page, TEST_USERNAME!, TEST_PASSWORD!);
        await expect(page.getByText(TEST_USERNAME!)).toBeVisible({ timeout: 10000 });

        // ========== 步骤2: 创建集群管理员用户 ==========
        createdClusterViewerUsername = await createUser(
            page,
            CLUSTER_VIEWER_USERNAME!,
            CLUSTER_VIEWER_INITIAL_PASSWORD!,
            'platform-regular'
        );

        // ========== 步骤2.5: 创建用于邀请测试的用户 ==========
        createdInviteTestUsername = await createUser(
            page,
            INVITE_TEST_USERNAME!,
            INVITE_TEST_PASSWORD!,
            'platform-regular'
        );

        // ========== 步骤3: 在集群中绑定集群管理员角色 ==========
        await page.goto(`${TEST_FUYAOURL}/container_platform/clusterMember`);
        await page.waitForLoadState('networkidle');

        // 点击邀请按钮
        await inviteUserWithRole(page, createdClusterViewerUsername, 'cluster-viewer');
        await expect(page.getByText(`邀请${createdClusterViewerUsername}成员成功！`)).toBeVisible({ timeout: 5000 });

        await page.goto(`${TEST_FUYAOURL}/container_platform/workload/pod`);
        await page.waitForSelector('table', { timeout: 10000 });

        // ========== 步骤4: 退出当前用户 ==========
        await logout(page);
        await expect(page).toHaveURL(/\/auth\/login|\/oauth2\/auth\/login/, { timeout: 10000 });

        // ========== 步骤5: 用集群管理员登录并修改初始密码 ==========
        await login(page, createdClusterViewerUsername, CLUSTER_VIEWER_INITIAL_PASSWORD!);

        // 首次登录需要修改密码
        await firstLoginChangePassword(page, CLUSTER_VIEWER_PASSWORD!);

        // 验证登录成功
        await expect(page.getByText(createdClusterViewerUsername)).toBeVisible({ timeout: 10000 });
    });

    test.afterAll(async () => {
        // ========== 清理: 删除集群管理员用户 ==========

        // 退出集群管理员账号
        await logout(page);
        await expect(page).toHaveURL(/\/auth\/login|\/oauth2\/auth\/login/, { timeout: 10000 });

        // 用平台管理员登录
        await login(page, TEST_USERNAME!, TEST_PASSWORD!);
        await expect(page.getByText(TEST_USERNAME!)).toBeVisible({ timeout: 10000 });

        // 步骤1: 从集群成员中移除用户
        await page.goto(`${TEST_FUYAOURL}/container_platform/clusterMember`);
        await page.waitForLoadState('networkidle');

        // 点击用户的操作按钮
        await removeClusterUser(page, createdClusterViewerUsername, 'cluster-viewer');

        // 步骤2: 从用户管理中删除用户
        await page.goto(`${TEST_FUYAOURL}/user_manage/user`);
        await page.waitForLoadState('networkidle');

        await deleteUser(page, createdClusterViewerUsername, 'platform-regular');

        // 等待成功消息消失并刷新页面
        await expect(page.getByText(`删除用户${createdClusterViewerUsername}成功！`)).not.toBeVisible({ timeout: 5000 });
        await page.reload();
        await page.waitForLoadState('networkidle');

        // 步骤3: 删除邀请测试用户
        await deleteUser(page, createdInviteTestUsername, 'platform-regular');

        // 关闭页面和上下文
        await page.close();
        await context.close();
    })

    test('【用户管理-050】cluster-viewer及以上角色的用户登录容器平台可以对资源进行增删改查-3', {
        tag: ['@v24.09', '@多集群多用户分权分域认证鉴权'],
        annotation: {
            type: '备注',
            description: '这个用例包platform-admin、cluster-admin、cluster-viewer三个角色的操作，这里是cluster-viewer角色的操作'+
            'cluster-viewer只能查、改，不能增、删'
        }
    }, async () => {
        // 设置测试超时时间为60秒
        test.setTimeout(60000);

        await test.step('资源（pod）增删查改', async () => {

            // ========== 第一部分：Pod 操作 ==========

            // 1. 导航到 Pod 列表页
            await page.goto(`${TEST_FUYAOURL}/container_platform/workload/pod`);
            await page.waitForSelector('table', { timeout: 10000 });
        });
    });

    test('【用户管理-050】cluster-viewer及以上角色的用户登录容器平台可以对资源进行增删改查-4', {
        tag: ['@v24.09', '@多集群多用户分权分域认证鉴权'],
        annotation: {
            type: '备注',
            description: 'cluster-viewer不能查看两个成员页面'
        }
    }, async () => {
        // ========== 验证不能访问平台角色管理页面 ==========
        await page.goto(`${TEST_FUYAOURL}/user_manage/role`);
        await page.waitForLoadState('networkidle');

        // 验证显示无权限提示
        await expect(page.getByText('当前用户尚无访问权限')).toBeVisible({ timeout: 5000 });
        await expect(page.getByText('请联系管理员添加用户权限')).toBeVisible();

        // ========== 验证可以访问集群角色页面 ==========
        await page.goto(`${TEST_FUYAOURL}/container_platform/clusterMember`);
        await page.waitForLoadState('networkidle');

        // 验证显示无权限提示
        await expect(page.getByText('当前用户尚无访问权限')).toBeVisible({ timeout: 5000 });
        await expect(page.getByText('请联系管理员添加用户权限')).toBeVisible();
    });
});

