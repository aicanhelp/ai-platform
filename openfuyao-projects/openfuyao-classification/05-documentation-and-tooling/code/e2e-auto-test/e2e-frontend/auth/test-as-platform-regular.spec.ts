// 测试用例：验证 platform-regular 用户在没有分配任何权限时，登录后看不到一般的系统页面

import { test, expect, Page, BrowserContext } from '@playwright/test';
import { TEST_FUYAOURL, TEST_USERNAME, TEST_PASSWORD } from './consts';
import { newContext } from '../utils/common';

test.describe('platform-regular 用户无权限测试', () => {
  let page: Page;
  let context: BrowserContext;
  const TEST_USER = 'test-regular';
  const TEST_USER_INITIAL_PASSWORD = 'Test@123456';
  const TEST_USER_NEW_PASSWORD = 'NewTest@123456';

  test.beforeAll(async ({ browser }) => {
    // 创建浏览器上下文
    context = await newContext(browser);
    page = await context.newPage();

    // ========== 步骤1: 使用平台管理员登录 ==========
    await page.goto(`${TEST_FUYAOURL}`);
    await page.waitForLoadState('networkidle');
    await page.getByRole('textbox', { name: '*用户名' }).fill(TEST_USERNAME!);
    await page.getByRole('textbox', { name: '*密码' }).fill(TEST_PASSWORD!);
    await page.getByRole('button', { name: '登录' }).click();
    await expect(page.getByText(TEST_USERNAME!)).toBeVisible({ timeout: 10000 });

    // ========== 步骤2: 创建角色为 platform-regular 的用户（不分配任何权限） ==========
    await page.goto(`${TEST_FUYAOURL}/user_manage/user`);
    await page.waitForLoadState('networkidle');

    // 检查用户是否已存在，如果存在则先删除
    const existingUser = page.getByRole('cell', { name: TEST_USER, exact: true });
    const userExists = await existingUser.isVisible({ timeout: 3000 }).catch(() => false);
    
    if (userExists) {
      // 找到用户所在的行
      const userRow = page.locator('tr.container-platform-table-row').filter({ hasText: TEST_USER });
      await userRow.getByRole('img', { name: 'more' }).click();
      
      // 点击删除按钮
      await page.getByRole('button', { name: '删除' }).click();
      
      // 等待删除确认对话框出现
      await expect(page.getByRole('dialog', { name: '删除用户' })).toBeVisible();
      
      // 勾选确认复选框
      await page.getByRole('checkbox', { name: '已了解删除操作无法恢复' }).check();
      
      // 点击删除确认按钮
      await page.getByRole('button', { name: '删 除' }).click();
      
      await expect(page.getByText(`删除用户${TEST_USER}成功！`)).toBeVisible({ timeout: 5000 });
    }

    // 点击创建用户按钮
    await page.getByRole('button', { name: /创\s*建/ }).click();
    await expect(page.getByRole('dialog', { name: '创建用户' })).toBeVisible();

    // 填写用户信息
    await page.getByRole('textbox', { name: '* 用户名称' }).fill(TEST_USER);
    await page.getByRole('textbox', { name: '* 初始密码' }).fill(TEST_USER_INITIAL_PASSWORD);

    // 选择平台角色为 platform-regular
    await page.locator('.container-platform-select-in-form-item').click();
    await page.getByTitle('platform-regular').click();

    // 点击确定创建用户（不分配任何权限，不填写用户描述）
    await page.getByRole('button', { name: /确\s*定|确\s*认/ }).click();
    await expect(page.getByText('创建用户成功！')).toBeVisible({ timeout: 5000 });

    // ========== 步骤3: 退出当前登录 ==========
    // 鼠标移动到右上角用户名元素上
    await page.locator('p').filter({ hasText: TEST_USERNAME! }).hover();
    
    // 等待弹窗出现并点击退出登录
    await page.getByText('退出登录').click();
    
    // 等待跳转到登录页面
    await expect(page).toHaveURL(/\/auth\/login/, { timeout: 10000 });
  });

  test.afterAll(async () => {
    const userElement = page.locator('p').filter({ hasText: TEST_USER });
    const isUserVisible = await userElement.isVisible({ timeout: 2000 }).catch(() => false);
    if (isUserVisible) {
        await userElement.hover();
        await page.getByText('退出登录').click({ timeout: 2000 });
        await page.waitForTimeout(1000);
    }

    // 退出登录后页面会自动跳转到登录页，无需再 goto 首页：
    await page.waitForURL(/\/auth\/login/, { timeout: 10000 });
    await page.getByRole('textbox', { name: '*用户名' }).fill(TEST_USERNAME!);
    await page.getByRole('textbox', { name: '*密码' }).fill(TEST_PASSWORD!);
    await page.getByRole('button', { name: '登录' }).click();
    await expect(page.getByText(TEST_USERNAME!)).toBeVisible({ timeout: 10000 });

    // 删除测试用户
    await page.goto(`${TEST_FUYAOURL}/user_manage/user`, { timeout: 1000 });
    await page.waitForLoadState('networkidle');
    
    const userRow = page.locator('tr.container-platform-table-row').filter({ hasText: TEST_USER });
    const userExists = await userRow.isVisible({ timeout: 3000 }).catch(() => false);
    
    if (userExists) {
        await userRow.getByRole('img', { name: 'more' }).click();
        await page.getByRole('button', { name: '删除' }).click();
        await expect(page.getByRole('dialog', { name: '删除用户' })).toBeVisible({ timeout: 5000 });
        await page.getByRole('checkbox', { name: '已了解删除操作无法恢复' }).check();
        await page.getByRole('button', { name: '删 除' }).click();
        await expect(page.getByText(`删除用户${TEST_USER}成功！`)).toBeVisible({ timeout: 5000 });
    }
    await page.close();
    await context.close();
  });

  test.afterEach(async()=>{
    await page.goto(`${TEST_FUYAOURL}`);
    await page.waitForLoadState('networkidle');
  })

  test('【用户管理-052】角色为platform-regular的用户登录后会进入单独的页面，看不了容器平台任何内容', {
    tag: ['@v24.09', '@多集群多用户分权分域认证鉴权']
  }, async () => {
    // ========== 步骤1: 使用新创建的 platform-regular 用户登录 ==========
    await page.goto(`${TEST_FUYAOURL}`);
    await page.waitForLoadState('networkidle');

    // 输入用户名
    await page.getByRole('textbox', { name: '*用户名' }).fill(TEST_USER);

    // 输入初始密码
    await page.getByRole('textbox', { name: '*密码' }).fill(TEST_USER_INITIAL_PASSWORD);

    // 点击登录按钮
    await page.getByRole('button', { name: '登录' }).click();

    // ========== 步骤2: 处理首次登录需要修改密码的情况 ==========
    // 等待页面加载，可能是登录页面或密码修改页面
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000); // 等待页面稳定
    
    // 检查是否显示密码修改对话框
    const passwordConfirmHeading = page.getByRole('heading', { name: '初次修改密码' });
    let needsPasswordChange = false;
    try {
      needsPasswordChange = await passwordConfirmHeading.isVisible({ timeout: 5000 });
    } catch (e) {
      needsPasswordChange = false;
    }

    if (needsPasswordChange) {
      // 等待对话框完全加载
      await page.waitForTimeout(1000);
      
      // 输入新密码
      const newPasswordInput = page.getByRole('textbox', { name: '*新密码' });
      await newPasswordInput.waitFor({ state: 'visible', timeout: 10000 });
      await newPasswordInput.clear();
      await newPasswordInput.fill(TEST_USER_NEW_PASSWORD);
      await page.waitForTimeout(500);
      
      // 输入确认密码
      const confirmPasswordInput = page.locator('#confirm-password');
      await confirmPasswordInput.waitFor({ state: 'visible', timeout: 10000 });
      await confirmPasswordInput.clear();
      await confirmPasswordInput.fill(TEST_USER_NEW_PASSWORD);
      await page.waitForTimeout(500);
      
      // 点击确认按钮
      const confirmButton = page.getByRole('button', { name: /确\s*定|确\s*认/ });
      await confirmButton.waitFor({ state: 'visible', timeout: 10000 });
      await confirmButton.click();
      
      // 等待密码修改对话框消失（最多等待20秒）
      await expect(passwordConfirmHeading).not.toBeVisible({ timeout: 20000 });
      
      // 等待页面跳转完成
      await page.waitForLoadState('networkidle', { timeout: 20000 });
      await page.waitForTimeout(3000); // 额外等待确保页面完全加载
    }

    // ========== 步骤3: 验证用户看不到一般的系统页面 ==========
    // 等待页面加载完成
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000); // 额外等待确保页面完全渲染

    // 验证显示"当前用户尚无访问权限"的提示信息
    await expect(page.getByText('当前用户尚无访问权限')).toBeVisible({ timeout: 1000 });
    
    // 验证显示"请联系管理员添加用户权限"的提示信息
    await expect(page.getByText('请联系管理员添加用户权限')).toBeVisible();

    // 验证页面上没有显示正常的系统功能内容
    // 正常情况下应该能看到总览页面的内容（如节点、Pod、应用等统计信息）
    // 但无权限用户应该只看到无权限提示，不应该看到这些内容
    
    // 检查是否没有显示"节点"统计信息（正常登录后应该显示）
    const nodeStats = page.getByText('节点', { exact: false });
    const hasNodeStats = await nodeStats.isVisible().catch(() => false);
    if (hasNodeStats) {
      // 如果能看到节点统计，检查是否在无权限提示区域内
      const nodeStatsText = await nodeStats.textContent();
      if (nodeStatsText && nodeStatsText.includes('节点') && !nodeStatsText.includes('总数')) {
        // 可能是无权限页面上的其他文本，继续检查
      } else {
        throw new Error('用户不应该能看到节点统计信息');
      }
    }

    // 验证页面URL（应该已经跳转到主页面，但显示无权限提示）
    await expect(page).toHaveURL(new RegExp(`${TEST_FUYAOURL!.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}/?$`), { timeout: 1000 });
    
    // 验证用户名显示在右上角（说明已登录）
    await expect(page.getByText(TEST_USER)).toBeVisible();
  });
});

