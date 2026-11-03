import { BrowserContext, expect, Page, test } from '@playwright/test';
import { FUYAO_BASE_URL } from '@/utils/constants';
import {
  fillYamlContent,
  saveCreateOrEdit,
  confirmDeleteDialog,
  encodeCrdNameForUrl,
} from '@/console/utils/common';
import {
  clickInstanceTab,
  searchInstance,
  clickRowAction,
  clickStartVm,
  clickStopVm,
  clickRestartVm,
  clickDeleteInstance,
} from './utils/virtualmachine';
import { login, firstLoginChangePassword, newAuthorizedContext, newContext } from '@/utils/common';
import {
  createUser,
  inviteUserWithRole,
  removeClusterUser,
  deleteUser,
} from '@/auth/utils/user-management';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const staticDir = path.resolve(__dirname, '../static');

const CRD_DETAIL_URL = FUYAO_BASE_URL + '/container_platform/customResourceDefinition/' + encodeCrdNameForUrl('virtualmachines.kubevirt.io');
const RBAC_VM_NAME = 'test-vm-rbac';
const RBAC_USER_BASE = 'rbacvmuser';

async function verifyOperationDenied(
  page: Page,
  operationName: string,
  clickOperation: () => Promise<void>,
) {
  const deniedPatterns = [
    /无权限/,
    /权限不足/,
    /没有.*权限/,
    /操作失败/,
    /拒绝/,
    /禁止访问/,
    /forbidden/i,
    /denied/i,
    /当前用户尚无访问权限/,
  ];

  const modal = page.locator('.pop_modal');

  const btnLocator = modal.getByRole('button', { name: operationName });
  const actionVisible = await btnLocator
    .isVisible({ timeout: 2000 })
    .catch(() => false);

  if (!actionVisible) {
    return;
  }

  const btnEnabled = await btnLocator.isEnabled().catch(() => true);
  if (!btnEnabled) {
    return;
  }

  await clickOperation();
  await page.waitForTimeout(1000);

  for (const pattern of deniedPatterns) {
    const isVisible = await page.getByText(pattern).first().isVisible().catch(() => false);
    if (isVisible) {
      return;
    }
  }

  await expect(
    page.getByText(/无权限|权限不足|没有.*权限|操作失败|拒绝|禁止访问|forbidden|denied|当前用户尚无访问权限/i).first(),
    `${operationName} 操作未按预期被拒绝`,
  ).toBeVisible({ timeout: 3000 });
}

test.describe('VirtualMachine RBAC 权限验证', { tag: ['@skip-temporarily'] }, () => {
  test.describe.configure({ mode: 'serial' });

  let crdExists = false;
  let adminContext: BrowserContext;
  let adminPage: Page;
  let rbacUsername: string;

  test.beforeAll(async ({ browser }) => {
    adminContext = await newAuthorizedContext(browser);
    adminPage = await adminContext.newPage();
    await adminPage.goto(FUYAO_BASE_URL);
    await adminPage.waitForLoadState('networkidle');
    if (adminPage.url().includes('/oauth2/auth/login/fuyaoPasswordProvider')) {
      await login(adminPage, process.env.TEST_USERNAME!, process.env.TEST_PASSWORD!);
      await expect(adminPage.getByText(process.env.TEST_USERNAME!)).toBeVisible({ timeout: 10000 });
    }

    // CRD 存在性检测
    await adminPage.goto(CRD_DETAIL_URL);
    await adminPage.waitForLoadState('networkidle');
    crdExists = await adminPage
      .getByRole('tab', { name: '实例' })
      .waitFor({ state: 'visible', timeout: 8000 })
      .then(() => true)
      .catch(() => false);
    test.skip(!crdExists, 'virtualmachines.kubevirt.io CRD 不存在，跳过 kubevirt RBAC 用例');

    await clickInstanceTab(adminPage);
    await adminPage.getByRole('button', { name: /创\s*建/ }).click();
    await adminPage.waitForURL(/createCR/, { timeout: 10000 });
    const yamlContent = fs.readFileSync(path.join(staticDir, 'valid-virtualmachine-rbac.yaml'), 'utf-8');
    await fillYamlContent(adminPage, yamlContent);
    await saveCreateOrEdit(adminPage);
    await expect(adminPage.getByText(/创建成功/)).toBeVisible({ timeout: 10000 });

    await adminPage.goto(CRD_DETAIL_URL);
    await clickInstanceTab(adminPage);
    await searchInstance(adminPage, RBAC_VM_NAME);
    await expect(adminPage.getByRole('row', { name: new RegExp(RBAC_VM_NAME) }).first()).toBeVisible({ timeout: 10000 });

    rbacUsername = await createUser(
      adminPage,
      RBAC_USER_BASE,
      process.env.TEST_INITIAL_PASSWORD!,
      'platform-regular',
    );

    await adminPage.goto(`${FUYAO_BASE_URL}/container_platform/clusterMember`);
    await adminPage.waitForLoadState('networkidle');
    await inviteUserWithRole(adminPage, rbacUsername, 'cluster-viewer');
  });

  test.afterAll(async () => {
    if (!crdExists) return;
    await adminPage.goto(FUYAO_BASE_URL);
    await adminPage.waitForLoadState('networkidle');
    if (adminPage.url().includes('/oauth2/auth/login/fuyaoPasswordProvider')) {
      await login(adminPage, process.env.TEST_USERNAME!, process.env.TEST_PASSWORD!);
      await expect(adminPage.getByText(process.env.TEST_USERNAME!)).toBeVisible({ timeout: 10000 });
    }

    await adminPage.goto(CRD_DETAIL_URL);
    await clickInstanceTab(adminPage);
    await searchInstance(adminPage, RBAC_VM_NAME);
    if (await adminPage.getByRole('row', { name: new RegExp(RBAC_VM_NAME) }).first().isVisible({ timeout: 5000 })) {
      await clickRowAction(adminPage, RBAC_VM_NAME);
      await clickDeleteInstance(adminPage);
      await confirmDeleteDialog(adminPage);
      await expect(adminPage.getByText(/删除成功/)).toBeVisible({ timeout: 10000 });
    }

    await adminPage.goto(`${FUYAO_BASE_URL}/container_platform/clusterMember`);
    await adminPage.waitForLoadState('networkidle');
    await removeClusterUser(adminPage, rbacUsername, 'cluster-viewer');

    await adminPage.goto(`${FUYAO_BASE_URL}/user_manage/user`);
    await adminPage.waitForLoadState('networkidle');
    await deleteUser(adminPage, rbacUsername, 'platform-regular');

    await adminContext.close();
  });

  test('鲲鹏-kubevirt-IT-015 - 限制权限用户无法对 VM 执行操作', {
    tag: ['@v26.06', '@kubevirt', '@rbac'],
  }, async ({ browser }) => {
    const viewerContext = await newContext(browser, {
      storageState: { cookies: [], origins: [] },
    });
    const viewerPage = await viewerContext.newPage();
    await viewerPage.goto(FUYAO_BASE_URL);
    await viewerPage.waitForLoadState('networkidle');
    await login(viewerPage, rbacUsername, process.env.TEST_INITIAL_PASSWORD!);
    await firstLoginChangePassword(viewerPage, process.env.TEST_PASSWORD!);
    await expect(viewerPage.getByText(rbacUsername)).toBeVisible({ timeout: 10000 });

    await viewerPage.goto(CRD_DETAIL_URL);
    await clickInstanceTab(viewerPage);
    await searchInstance(viewerPage, RBAC_VM_NAME);
    await expect(viewerPage.getByRole('row', { name: new RegExp(RBAC_VM_NAME) }).first()).toBeVisible({ timeout: 10000 });

    await test.step('开机操作被拒绝', async () => {
      await clickRowAction(viewerPage, RBAC_VM_NAME);
      await verifyOperationDenied(viewerPage, '开机', async () => {
        await clickStartVm(viewerPage);
      });
    });

    await test.step('关机操作被拒绝', async () => {
      await clickRowAction(viewerPage, RBAC_VM_NAME);
      await verifyOperationDenied(viewerPage, '关机', async () => {
        await clickStopVm(viewerPage);
      });
    });

    await test.step('重启操作被拒绝', async () => {
      await clickRowAction(viewerPage, RBAC_VM_NAME);
      await verifyOperationDenied(viewerPage, '重启', async () => {
        await clickRestartVm(viewerPage);
      });
    });

    await test.step('删除操作被拒绝', async () => {
      await clickRowAction(viewerPage, RBAC_VM_NAME);
      await verifyOperationDenied(viewerPage, '删除', async () => {
        await clickDeleteInstance(viewerPage);
      });
    });

    await viewerContext.close();
  });
});
