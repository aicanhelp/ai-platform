import { test, expect, Page, BrowserContext } from '@playwright/test';
import { TEST_FUYAOURL, TEST_USERNAME, TEST_PASSWORD } from './consts';
import { createUser } from '../auth/utils/user-management';
import { login, firstLoginChangePassword, logout } from '../utils/common';
import { deleteUserFromUserManagement } from './utils/cleanup-helpers';
import { newContext } from '@/utils/common';

test.describe('platform-regular 用户无权限访问终端测试', () => {
  let page: Page;
  let context: BrowserContext;
  const TEST_USER_PREFIX = 'test-regular';
  const TEST_USER_INITIAL_PASSWORD = 'Test@123456';
  const TEST_USER_NEW_PASSWORD = 'NewTest@123456';
  let currentRegularPassword = TEST_USER_INITIAL_PASSWORD;
  let createdRegularUsername: string;

  test.beforeAll(async ({ browser }) => {
    // 创建浏览器上下文
    context = await newContext(browser);
    page = await context.newPage();

    // ========== 步骤1: 使用平台管理员登录 ==========
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

    // ========== 步骤2: 创建角色为 platform-regular 的用户（不分配任何权限） ==========
    createdRegularUsername = await createUser(
      page,
      TEST_USER_PREFIX,
      TEST_USER_INITIAL_PASSWORD,
      'platform-regular'
    );

    // ========== 步骤3: 退出当前登录 ==========
    await logout(page);
    await expect(page).toHaveURL(/\/auth\/login/, { timeout: 10000 });

    // ========== 步骤4: 使用新创建的 platform-regular 用户登录 ==========
    await login(page, createdRegularUsername, currentRegularPassword);
    await firstLoginChangePassword(page, TEST_USER_NEW_PASSWORD);
    await page.waitForLoadState('networkidle', { timeout: 20000 });
    await page.waitForTimeout(3000);
    currentRegularPassword = TEST_USER_NEW_PASSWORD;
    await expect(page.getByText(createdRegularUsername)).toBeVisible({ timeout: 10000 });
  });

  test.afterAll(async () => {
    await logout(page);
    await expect(page).toHaveURL(/\/auth\/login|\/oauth2\/auth\/login/, { timeout: 10000 });

    // 重新登录为平台管理员
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

    // 删除测试用户
    await page.goto(`${TEST_FUYAOURL}/user_manage/user`);
    await page.waitForLoadState('networkidle');
    
    if (createdRegularUsername) {
      await deleteUserFromUserManagement(page, createdRegularUsername);
    }
    await page.close();
    await context.close();
  });

  test.afterEach(async()=>{
    await page.goto(`${TEST_FUYAOURL}`);
  })

  test('【WebTerminal-017】platform-regular角色且未分配容器角色的用户无权限访问容器终端', {
    tag: ['@v24.12', '@web-terminal-service'],
  }, async () => {
    await expect(page.getByText('当前用户尚无访问权限')).toBeVisible({ timeout: 10000 });
    await expect(page.getByText('请联系管理员添加用户权限')).toBeVisible({ timeout: 10000 });
    await expect(page.getByText('Pod 列表')).toHaveCount(0);
    await expect(page.locator('.xterm').first()).toHaveCount(0);
  });

  test('【WebTerminal-028】platform-regular角色且未分配集群角色的用户无权限访问集群终端', {
    tag: ['@v24.12', '@web-terminal-service'],
  }, async () => {
    await expect(page.getByText('当前用户尚无访问权限')).toBeVisible({ timeout: 10000 });
    await expect(page.getByText('请联系管理员添加用户权限')).toBeVisible({ timeout: 10000 });
    await expect(page.getByText('kubectl终端工具')).toHaveCount(0);
    await expect(page.locator('.xterm').first()).toHaveCount(0);
  });
})
