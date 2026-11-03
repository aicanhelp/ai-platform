import { test, expect, Page, BrowserContext } from '@playwright/test';
import {
  TEST_FUYAOURL,
  TEST_USERNAME,
  TEST_PASSWORD,
  CLUSTER_VIEWER_USERNAME,
  CLUSTER_VIEWER_INITIAL_PASSWORD,
  CLUSTER_VIEWER_PASSWORD,
} from './consts';
import { createUser, inviteUserWithRole } from '../auth/utils/user-management';
import { login, firstLoginChangePassword, logout } from '../utils/common';
import { getVisibleLocator, openContainerTerminal, waitForPodListReady } from './utils/terminal-helpers';
import { deleteUserFromUserManagement, removeClusterUserFromMembers } from './utils/cleanup-helpers';
import { newContext } from '@/utils/common';

test.describe('cluster-viewer测试', () => {
  let page: Page;
  let context: BrowserContext;
  let createdClusterViewerUsername: string;

  test.beforeAll(async ({ browser }) => {
    context = await newContext(browser);
    page = await context.newPage();

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

    createdClusterViewerUsername = await createUser(
      page,
      CLUSTER_VIEWER_USERNAME!,
      CLUSTER_VIEWER_INITIAL_PASSWORD!,
      'platform-regular'
    );

    await page.goto(`${TEST_FUYAOURL}/container_platform/clusterMember`);
    await page.waitForLoadState('networkidle');
    await inviteUserWithRole(page, createdClusterViewerUsername, 'cluster-viewer');

    await logout(page);
    await expect(page).toHaveURL(/\/auth\/login|\/oauth2\/auth\/login/, { timeout: 10000 });

    await login(page, createdClusterViewerUsername, CLUSTER_VIEWER_INITIAL_PASSWORD!);
    await firstLoginChangePassword(page, CLUSTER_VIEWER_PASSWORD!);
    await expect(page.getByText(createdClusterViewerUsername)).toBeVisible({ timeout: 10000 });
  });

  test.afterAll(async () => {
    await logout(page);
    await expect(page).toHaveURL(/\/auth\/login|\/oauth2\/auth\/login/, { timeout: 10000 });

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

    await removeClusterUserFromMembers(page, createdClusterViewerUsername);

    await deleteUserFromUserManagement(page, createdClusterViewerUsername);

    await page.close();
    await context.close();
  });

  test.afterEach(async () => {
    await page.goto(`${TEST_FUYAOURL}`);
  });

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
          throw new Error('cluster-viewer 用户意外成功打开了容器终端');
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

  test('【WebTerminal-020】platform-regular角色且分配cluster-viewer角色的用户无权限访问容器终端', {
    tag: ['@v24.12', '@web-terminal-service'],
  }, async () => {
    await expectContainerTerminalAccessDenied();
  });

  test('【WebTerminal-031】platform-regular角色且分配cluster-viewer角色的用户无权限访问集群终端', {
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
});
