import { test, expect, Page, BrowserContext } from '@playwright/test';
import { TEST_FUYAOURL, TEST_USERNAME, TEST_PASSWORD, CLUSTER_ADMIN_USERNAME, CLUSTER_ADMIN_INITIAL_PASSWORD, CLUSTER_ADMIN_PASSWORD, INVITE_TEST_USERNAME, INVITE_TEST_PASSWORD } from './consts';
import { createUser, inviteUserWithRole, removeClusterUser, deleteUser, editClusterUser } from './utils/user-management';
import { newContext, login, firstLoginChangePassword, logout } from '../utils/common';
import { expectResourceDeleted } from '@/console/utils/common';

test.describe('集群管理员测试', () => {
    let page: Page;
    let context: BrowserContext;
    let createdClusterAdminUsername: string;
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
        createdClusterAdminUsername = await createUser(
            page,
            CLUSTER_ADMIN_USERNAME!,
            CLUSTER_ADMIN_INITIAL_PASSWORD!,
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
        await login(page, TEST_USERNAME!, TEST_PASSWORD!);
        await expect(page.getByText(TEST_USERNAME!)).toBeVisible({ timeout: 10000 });

        // 步骤1: 从集群成员中移除用户
        await page.goto(`${TEST_FUYAOURL}/container_platform/clusterMember`);
        await page.waitForLoadState('networkidle');

        await removeClusterUser(page, createdClusterAdminUsername, 'cluster-admin');

        // 步骤2: 从用户管理中删除用户
        await page.goto(`${TEST_FUYAOURL}/user_manage/user`);
        await page.waitForLoadState('networkidle');

        await deleteUser(page, createdClusterAdminUsername, 'platform-regular');

        // 等待成功消息消失并刷新页面
        await page.reload();
        await page.waitForLoadState('networkidle');

        // 步骤3: 删除邀请测试用户
        await deleteUser(page, createdInviteTestUsername, 'platform-regular');

        // 关闭页面和上下文
        await page.close();
        await context.close();
    });

    // ========== 测试用例 ==========

    test('【用户管理-009】验证任意用户首次登录必须修改密码；'+
        '【用户管理-070】创建用户后首次登录时输入正确用户名和密码跳转正确；'+
        '【用户管理-071】创建用户后首次登录必须修改密码'+
        '【用户管理-072】创建用户后登录成功', {
        annotation: {
            type: '备注',
            description: '验证工作实际在beforeAll中完成'
        }
    }, async () => {
        // 验证当前页面显示集群管理员用户名
        await expect(page.getByText(createdClusterAdminUsername)).toBeVisible();
    });

    test('【用户管理-057】集群管理员权限的用户无法查看平台角色列表，可以查看集群角色列表', {
        tag: ['@v24.09', '@多集群多用户分权分域认证鉴权']
    }, async () => {
        // ========== 验证不能访问平台角色管理页面 ==========
        await page.goto(`${TEST_FUYAOURL}/user_manage/role`);
        await page.waitForLoadState('networkidle');

        // 验证显示无权限提示
        await expect(page.getByText('当前用户尚无访问权限')).toBeVisible({ timeout: 5000 });
        await expect(page.getByText('请联系管理员添加用户权限')).toBeVisible();

        // ========== 验证可以访问集群角色页面 ==========
        await page.goto(`${TEST_FUYAOURL}/container_platform/clusterUser`);
        await page.waitForLoadState('networkidle');

        // 验证页面导航显示"集群角色"
        await expect(page.getByRole('listitem').filter({ hasText: '集群角色' })).toBeVisible({ timeout: 5000 });

        // 验证可以看到集群角色列表
        await expect(page.getByRole('cell', { name: 'cluster-admin', exact: true })).toBeVisible();
        await expect(page.getByRole('cell', { name: 'cluster-editor', exact: true })).toBeVisible();
        await expect(page.getByRole('cell', { name: 'cluster-viewer', exact: true })).toBeVisible();
    });

    test('【用户管理-062】集群管理员权限的用户邀请用户并绑定集群角色' +
        '【用户管理-065】集群管理员权限的用户修改集群用户的集群角色' +
        '【用户管理-068】集群管理员权限的用户移除集群用户', {
        tag: ['@v24.09', '@多集群多用户分权分域认证鉴权']
    }, async () => {
        // ========== 进入集群成员页面 ==========
        await page.goto(`${TEST_FUYAOURL}/container_platform/clusterMember`);
        await page.waitForLoadState('networkidle');

        // 验证邀请按钮可见
        await expect(page.getByRole('button', { name: '邀 请' })).toBeVisible({ timeout: 5000 });

        // ========== 邀请测试用户 ==========
        await inviteUserWithRole(page, createdInviteTestUsername, 'cluster-viewer');

        // ========== 验证邀请成功 ==========
        await expect(page.getByText(`邀请${createdInviteTestUsername}成员成功！`)).toBeVisible({ timeout: 5000 });

        // 验证用户出现在集群成员列表中
        await expect(page.getByRole('row', { name: `${createdInviteTestUsername} cluster-viewer` })).toBeVisible();

        // ========== 修改用户角色 ==========
        await editClusterUser(page, createdInviteTestUsername, 'cluster-viewer', 'cluster-editor');

        // ========== 验证角色修改成功 ==========
        await expect(page.getByText(`编辑${createdInviteTestUsername}成员成功！`)).toBeVisible({ timeout: 5000 });

        // 验证用户角色已更新为 cluster-editor
        await expect(page.getByRole('row', { name: `${createdInviteTestUsername} cluster-editor` })).toBeVisible();

        // ========== 清理：移除刚邀请的用户 ==========
        await removeClusterUser(page, createdInviteTestUsername, 'cluster-editor');
    });

    test('【用户管理-050】cluster-editor及以上角色的用户登录容器平台可以对资源进行增删改查-2', {
        tag: ['@v24.09', '@多集群多用户分权分域认证鉴权'],
        annotation: {
            type: '备注',
            description: '这个用例包platform-admin、cluster-admin、cluster-editor三个角色的操作，这里是cluster-admin角色的操作'
        }
    }, async () => {
        // 设置测试超时时间为300秒
        test.setTimeout(300000);
        // 生成符合 Kubernetes 命名规范的唯一名称（DNS-1123 子域名规范）
        // 使用时间戳后8位 + 随机字符串，确保唯一性且符合规范
        const timestamp = Date.now().toString().slice(-8);
        const randomStr = Math.random().toString(36).substring(2, 6);
        const testPodName = `e2e-pod-${timestamp}-${randomStr}`;

        await test.step('资源（pod）增删查改', async () => {

            // ========== 第一部分：Pod 操作 ==========

            // 1. 导航到 Pod 列表页
            await page.goto(`${TEST_FUYAOURL}/container_platform/workload/pod`);
            await page.waitForSelector('table', { timeout: 10000 });

            // 2. 创建 Pod
            await page.getByRole('button', { name: '创 建' }).click();

            // 等待YAML编辑器加载（CodeMirror编辑器）
            await page.waitForSelector('.cm-content', { timeout: 10000 });

            const podYaml = `apiVersion: v1
kind: Pod
metadata:
  name: ${testPodName}
  namespace: default
  labels:
    app: ${testPodName}
spec:
  containers:
  - name: nginx
    image: nginx:1.19.3
    ports:
    - containerPort: 80`;

            // 定位CodeMirror编辑器区域，使用 fill 方法填充内容
            const createEditor = page.locator('.cm-content');
            await createEditor.click();
            await page.keyboard.press('Control+A');
            await page.keyboard.press('Delete');
            await createEditor.fill(podYaml);

            // 点击确定
            await page.getByRole('button', { name: /确\s*定|确\s*认/ }).click();

            // 验证创建成功提示（实际提示是"创建成功"）
            await expect(page.getByText('创建成功')).toBeVisible();

            // 搜索并等待 Pod 变为 Running 状态
            await page.getByRole('textbox', { name: /搜索\s*Pod\s*名称/ }).fill(testPodName);
            await page.getByRole('textbox', { name: /搜索\s*Pod\s*名称/ }).press('Enter');
            await page.waitForLoadState('networkidle');

            // 轮询检查状态
            let podRunning = false;
            for (let i = 0; i < 20; i++) {
                const podRow = page.locator('tr').filter({ hasText: testPodName }).filter({ hasText: 'Running' });
                if (await podRow.count() > 0) {
                    podRunning = true;
                    break;
                }
                await page.getByRole('button', { name: 'sync' }).click();
                await page.waitForTimeout(10000);
            }
            expect(podRunning).toBe(true);

            // 3. 修改 Pod
            const podRow = page.locator('tr').filter({ hasText: testPodName });
            await podRow.getByRole('img', { name: 'more' }).click();
            await page.getByRole('button', { name: '修改' }).click();

            // 等待CodeMirror编辑器
            await page.waitForSelector('.cm-content', { timeout: 10000 });

            // 点击编辑器获得焦点
            await page.locator('[role="textbox"][data-language="yaml"]').first().click();

            // 打开搜索替换对话框
            await page.keyboard.press('Control+f');
            await page.waitForTimeout(500);

            // 在查找框中输入要查找的文本
            await page.getByRole('textbox', { name: 'Find' }).fill(`app: ${testPodName}`);

            // 在替换框中输入要替换的文本
            await page.getByRole('textbox', { name: 'Replace' }).fill(`app: ${testPodName}-edited`);

            // 点击替换按钮（或 replace all）
            await page.getByRole('button', { name: 'replace all' }).click();
            await page.waitForTimeout(300);

            // 关闭搜索对话框
            await page.getByRole('button', { name: 'close' }).click();

            await page.getByRole('button', { name: /确\s*定|确\s*认/ }).click();
            await expect(page.getByText(/修改成功/)).toBeVisible();

            // 4. 删除 Pod
            // 重新搜索并定位到创建的 pod
            await page.goto(`${TEST_FUYAOURL}/container_platform/workload/pod`);
            await page.waitForLoadState('networkidle'); // 
            await page.getByRole('textbox', { name: /搜索\s*Pod\s*名称/ }).fill(testPodName);
            await page.getByRole('textbox', { name: /搜索\s*Pod\s*名称/ }).press('Enter');
            await page.waitForLoadState('networkidle');

            // 重新定位 pod 行（因为页面可能已刷新）
            const podRowToDelete = page.locator('tr').filter({ hasText: testPodName });
            await podRowToDelete.getByRole('img', { name: 'more' }).click();
            await page.getByRole('button', { name: '删除' }).click();

            // 确认删除
            await expect(page.getByRole('dialog', { name: '删除Pod' })).toBeVisible();
            await page.getByRole('checkbox', { name: '已了解删除操作无法恢复' }).check();
            await page.getByRole('button', { name: '删 除' }).click();

            // 验证删除成功提示
            await expect(page.getByText('删除成功')).toBeVisible({ timeout: 10000 });

            // 验证删除
            await expect(page.getByRole('dialog', { name: '删除Pod' })).not.toBeVisible();
            await expectResourceDeleted(page, testPodName);
        });
    });
});

