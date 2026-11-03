// In this file, the "beforeAll" will login in as platform admin

import { test, expect, Page } from '@playwright/test';
import { TEST_FUYAOURL, TEST_USERNAME, TEST_PASSWORD, CLUSTER_ADMIN_INITIAL_PASSWORD, TEST_INITIAL_PASSWORD } from './consts';
import { createUser, deleteUser, inviteUserWithRole, removeClusterUser, editClusterUser, verifyClusterUserRole } from './utils/user-management';
import { newContext } from '../utils/common';
import { expectResourceDeleted } from '@/console/utils/common';

test.describe('平台管理员登录', () => {
  let page: Page;

  test.beforeEach(async ({ browser }) => {
    // 创建新的浏览器上下文和页面
    const context = await newContext(browser);
    page = await context.newPage();

    // 打开登录页面
    await page.goto(`${TEST_FUYAOURL}`);

    // 输入用户名
    await page.getByRole('textbox', { name: '*用户名' }).fill(TEST_USERNAME!);

    // 输入密码
    await page.getByRole('textbox', { name: '*密码' }).fill(TEST_PASSWORD!);

    // 点击登录按钮
    await page.getByRole('button', { name: '登录' }).click();

    // 等待登录成功 - 等待URL变化和网络空闲
    await page.waitForURL(/container_platform\/overview/, { timeout: 10000 });
    await page.waitForLoadState('networkidle', { timeout: 10000 });

    // 验证登录成功 - 使用更精确的选择器
    await expect(page.locator('p').filter({ hasText: TEST_USERNAME! }).first()).toBeVisible({ timeout: 10000 });
  });

  test.afterEach(async () => {
    await page.goto(`${TEST_FUYAOURL}`);
    // 鼠标移动到右上角 admin 元素上
    await page.locator('p').filter({ hasText: 'admin' }).hover();

    // 等待弹窗出现并点击退出登录
    await page.getByText('退出登录').click();

    // 关闭页面
    await page.close();
  });

  test('【用户管理-054】平台安装时自动创建平台及的默认用户', {
    tag: ['@v24.09', '@多集群多用户分权分域认证鉴权']
  }, async () => {
    // 1. 点击顶部导航栏的"用户管理"链接
    await page.getByRole('link', { name: '用户管理' }).click();

    // 2. 等待表格加载完成
    await page.waitForSelector('table', { timeout: 5000 });

    // 3. 验证admin用户存在
    await expect(page.getByRole('cell', { name: 'admin', exact: true })).toBeVisible();

    // 4. 验证admin用户的角色是platform-admin
    await expect(page.getByRole('cell', { name: 'platform-admin', exact: true })).toBeVisible();

    // 5. 验证admin用户的描述是"内置默认平台管理员"
    await expect(page.getByRole('cell', { name: '内置默认平台管理员' })).toBeVisible();
  });

  test('【用户管理-055】平台安装时自动创建集群级默认角色', {
    tag: ['@v24.09', '@多集群多用户分权分域认证鉴权']
  }, async () => {
    // 点击侧边栏的"集群角色"菜单
    await page.getByText('集群角色').click();

    // 等待表格加载完成（等待第一个表格行出现）
    await page.waitForSelector('tr.container-platform-table-row', { timeout: 5000 });

    // 验证 cluster-admin 角色存在（使用 exact: true 精确匹配）
    await expect(page.getByRole('cell', { name: 'cluster-admin', exact: true })).toBeVisible();

    // 验证 cluster-editor 角色存在
    await expect(page.getByRole('cell', { name: 'cluster-editor', exact: true })).toBeVisible();

    // 验证 cluster-viewer 角色存在
    await expect(page.getByRole('cell', { name: 'cluster-viewer', exact: true })).toBeVisible();
  });

  test('【用户管理-056】平台管理员权限的用户可以查看平台角色列表和集群角色列表' +
    '【用户管理-028】角色管理页面展示正常', {
    tag: ['@v24.09', '@多集群多用户分权分域认证鉴权'],
    annotation: {
      type: "备注",
      description: "这里是查看平台角色列表，集群角色列表在【用户管理-056】中已覆盖"
    }
  }, async () => {
    // 1. 点击顶部导航栏的"用户管理"链接
    await page.getByRole('link', { name: '用户管理' }).click();

    // 2. 点击左侧菜单的"角色管理"
    await page.getByRole('menuitem', { name: '角色管理' }).click();

    // 3. 等待角色列表表格加载
    await page.waitForSelector('table', { timeout: 5000 });

    // 4. 验证表头包含"角色名称"列
    await expect(page.getByRole('columnheader', { name: '角色名称' })).toBeVisible();

    // 5. 验证表头包含"描述"列
    await expect(page.getByRole('columnheader', { name: '描述' })).toBeVisible();
  });

  test('【用户管理-061】平台管理员权限的用户邀请用户并绑定集群角色；【用户管理-067】平台管理员权限的用户移除集群用户', {
    tag: ['@v24.09', '@多集群多用户分权分域认证鉴权'],
  }, async () => {
    let testUser = "test-user";

    await test.step('创建测试用户', async () => {
      // 创建测试用户
      testUser = await createUser(page, testUser, TEST_INITIAL_PASSWORD!, 'platform-regular');
    });

    await test.step('邀请用户并绑定集群角色', async () => {
      // 进入集群成员页面
      await page.getByRole('img', { name: 'home' }).locator('path').click();
      await page.getByRole('menuitem', { name: '集群成员' }).click();

      // 等待表格加载完成
      await page.waitForSelector('table', { timeout: 5000 });
      await page.waitForLoadState('networkidle');

      // 邀请用户并绑定角色
      await inviteUserWithRole(page, testUser, 'cluster-viewer');
      await expect(page.getByText(`邀请${testUser}成员成功！`).first()).toBeVisible();

      // 验证集群成员表格中出现test成员，角色为cluster-viewer
      await expect(page.getByRole('cell', { name: testUser, exact: true })).toBeVisible();
      // 验证该用户的角色为cluster-viewer（检查该用户所在的行）
      await expect(page.getByRole('row', { name: `${testUser} cluster-viewer` })).toBeVisible();
    });

    await test.step('移除集群用户', async () => {
      // 移除集群成员
      await removeClusterUser(page, testUser, 'cluster-viewer');
    });

    await test.step('删除测试用户', async () => {
      // 进入用户管理
      await page.getByRole('link', { name: '用户管理' }).click();
      await expect(page.getByRole('navigation').getByText('用户管理')).toBeVisible();

      // 删除测试用户
      await deleteUser(page, testUser, 'platform-regular');
    });
  });

  test('平台管理员修改集群成员角色后再改回；【用户管理-012】用户管理页面，列表操作列-修改，修改用户页面数据回显正确', {
    tag: ['@v24.09', '@多集群多用户分权分域认证鉴权'],
  }, async () => {
    let testUser = "test-user";
    const initialRole = 'cluster-viewer';
    const targetRole = 'cluster-editor';

    await test.step('创建测试用户', async () => {
      // 创建测试用户
      testUser = await createUser(page, testUser, TEST_INITIAL_PASSWORD!, 'platform-regular');
    });

    await test.step('邀请用户并绑定初始角色', async () => {
      // 进入集群成员页面
      await page.getByRole('img', { name: 'home' }).locator('path').click();
      await page.getByRole('menuitem', { name: '集群成员' }).click();

      // 等待表格加载完成
      await page.waitForSelector('table', { timeout: 5000 });
      await page.waitForLoadState('networkidle');

      // 邀请用户并绑定初始角色
      await inviteUserWithRole(page, testUser, initialRole);
      await expect(page.getByText(`邀请${testUser}成员成功！`).first()).toBeVisible();
    });

    await test.step('修改集群成员角色为目标角色', async () => {
      // 使用editClusterUser helper函数修改角色
      await editClusterUser(page, testUser, initialRole, targetRole);
      await verifyClusterUserRole(page, testUser, targetRole);
    });

    await test.step('修改集群成员角色回原角色', async () => {
      // 使用editClusterUser helper函数修改回原角色
      await editClusterUser(page, testUser, targetRole, initialRole);
      await verifyClusterUserRole(page, testUser, initialRole);
    });

    await test.step('删除测试用户', async () => {
      // 先移除集群成员
      await removeClusterUser(page, testUser, initialRole);

      // 再删除用户
      await page.getByRole('link', { name: '用户管理' }).click();
      await expect(page.getByRole('navigation').getByText('用户管理')).toBeVisible();
      await deleteUser(page, testUser, 'platform-regular');
    });
  });

  test('【用户管理-050】cluster-editor及以上角色的用户登录容器平台可以对资源进行增删改查-1', {
    tag: ['@v24.09', '@多集群多用户分权分域认证鉴权'],
    annotation: {
      type: '备注',
      description: '这个用例包platform-admin、cluster-admin、cluster-editor三个角色的操作，这里是platform-admin角色的操作'
    }
  }, async () => {
    // 设置测试超时时间为60秒
    test.setTimeout(60000);
    // 生成符合 Kubernetes 命名规范的唯一名称（DNS-1123 子域名规范）
    // 使用时间戳后8位 + 随机字符串，确保唯一性且符合规范
    const timestamp = Date.now().toString().slice(-8);
    const randomStr = Math.random().toString(36).substring(2, 6);
    const testPodName = `e2e-pod-${timestamp}-${randomStr}`;
    const testUserName = `e2e-user-${timestamp}-${randomStr}`;

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
        await page.waitForTimeout(3000);
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


    await test.step('平台成员增删查改', async () => {
      // ========== 第二部分：平台成员操作 ==========

      // 1. 导航到用户管理
      await page.goto(`${TEST_FUYAOURL}/user_manage/user`);
      await page.waitForSelector('table', { timeout: 10000 });

      await page.goto(`${TEST_FUYAOURL}/user_manage/user`);
      await page.waitForLoadState('networkidle');

      // 点击创建按钮
      await page.getByRole('button', { name: /创\s*建/ }).click();
      await expect(page.getByRole('dialog', { name: '创建用户' })).toBeVisible();

      // 填写用户信息
      await page.getByRole('textbox', { name: '* 用户名称' }).fill(testUserName!);
      await page.getByRole('textbox', { name: '* 初始密码' }).fill(CLUSTER_ADMIN_INITIAL_PASSWORD!);

      // 选择平台角色为 platform-regular
      await page.locator('.container-platform-select-in-form-item').click();
      await page.getByTitle('platform-regular').click();

      // 点击确定创建用户
      await page.getByRole('button', { name: /确\s*定|确\s*认/ }).click();
      await expect(page.getByText('创建用户成功！').first()).toBeVisible({ timeout: 5000 });

      // 搜索用户
      await page.getByRole('textbox', { name: '搜索用户名称' }).fill(testUserName);
      await page.getByRole('button', { name: 'search' }).click();
      await page.waitForLoadState('networkidle');

      // 验证用户存在及角色正确
      const userRow = page.locator('tr').filter({ hasText: testUserName });
      await expect(userRow).toBeVisible();
      await expect(userRow).toContainText('platform-regular');

      // 3. 修改用户
      await userRow.getByRole('img', { name: 'more' }).click();
      await page.getByRole('button', { name: '修改' }).click();

      const editDialog = page.getByRole('dialog', { name: '修改用户' });
      await expect(editDialog).toBeVisible();

      // 修改描述
      await editDialog.getByRole('textbox', { name: '用户描述' }).fill('Updated description');

      await editDialog.getByRole('button', { name: /确\s*定|确\s*认/ }).click();
      await expect(page.getByText(`编辑用户成功`)).toBeVisible();

      // 验证修改
      await expect(userRow).toContainText('Updated description');

      // 4. 删除用户
      await userRow.getByRole('img', { name: 'more' }).click();
      await page.getByRole('button', { name: '删除' }).click();

      // 确认删除
      await expect(page.getByRole('dialog', { name: '删除用户' })).toBeVisible();
      await page.getByRole('checkbox', { name: '已了解删除操作无法恢复' }).click();
      await page.getByRole('button', { name: '删 除' }).click();
      await expect(page.getByText(`删除用户${testUserName!}成功！`).first()).toBeVisible({ timeout: 5000 });
      await expect(userRow).not.toBeVisible();
    });
  });
});
