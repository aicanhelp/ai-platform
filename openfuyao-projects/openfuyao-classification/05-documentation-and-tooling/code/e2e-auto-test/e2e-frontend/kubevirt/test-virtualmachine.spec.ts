import { Browser, expect, test } from '@playwright/test';
import { FUYAO_BASE_URL } from '@/utils/constants';
import {
  fillYamlContent,
  saveCreateOrEdit,
  cancelCreateOrEdit,
  confirmDeleteDialog,
  encodeCrdNameForUrl,
} from '@/console/utils/common';
import { newAuthorizedContext } from '@/utils/common';
import {
  clickInstanceTab,
  searchInstance,
  clickRowAction,
  clickStartVm,
  clickStopVm,
  clickRestartVm,
  clickDeleteInstance,
  getEditorContent,
} from './utils/virtualmachine';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const staticDir = path.resolve(__dirname, '../static');

const CRD_DETAIL_URL = FUYAO_BASE_URL + '/container_platform/customResourceDefinition/' + encodeCrdNameForUrl('virtualmachines.kubevirt.io');
const VM_NAME = 'test-vm';

const setYamlInEditor = async (
  page: import('@playwright/test').Page,
  editFunc?: (content: string) => string,
  yamlFileName: string = 'valid-virtualmachine.yaml',
): Promise<void> => {
  let content = fs.readFileSync(path.join(staticDir, yamlFileName), 'utf-8');
  if (editFunc) {
    content = editFunc(content);
  }
  await fillYamlContent(page, content);
};

const cleanupTestVirtualMachine = async (browser: Browser) => {
  const context = await newAuthorizedContext(browser);
  const page = await context.newPage();
  await page.goto(CRD_DETAIL_URL);
  await clickInstanceTab(page);
  await searchInstance(page, VM_NAME);
  if (await page.getByRole('row', { name: new RegExp(VM_NAME) }).first().isVisible({ timeout: 10000 })) {
    await clickRowAction(page, VM_NAME);
    await clickDeleteInstance(page);
    await confirmDeleteDialog(page);
    await expect(page.getByText(/删除成功/)).toBeVisible({ timeout: 10000 });
  }
  await context.close();
};

test.describe('VirtualMachine管理', { tag: ['@skip-temporarily'] }, () => {
  let crdExists = false;

  test.beforeEach(async ({ page }) => {
    await page.goto(CRD_DETAIL_URL);
    await clickInstanceTab(page);
  });

  test.afterAll(async ({ browser }) => {
    if (!crdExists) return;
    await cleanupTestVirtualMachine(browser);
  });

  test.describe.configure({ mode: 'serial' });

  test.beforeAll(async ({ browser }) => {
    const ctx = await newAuthorizedContext(browser);
    const page = await ctx.newPage();
    await page.goto(CRD_DETAIL_URL);
    await page.waitForLoadState('networkidle');
    crdExists = await page
      .getByRole('tab', { name: '实例' })
      .waitFor({ state: 'visible', timeout: 8000 })
      .then(() => true)
      .catch(() => false);
    await ctx.close();
    test.skip(!crdExists, 'virtualmachines.kubevirt.io CRD 不存在，跳过 kubevirt 用例');
  });

  test('鲲鹏-kubevirt-IT-012 - 创建 VirtualMachine 成功', {
    tag: ['@v26.06', '@kubevirt'],
  }, async ({ page }) => {
    await page.getByRole('button', { name: /创\s*建/ }).click();
    await page.waitForURL(/createCR/, { timeout: 10000 });
    await setYamlInEditor(page);
    await saveCreateOrEdit(page);

    await page.goto(CRD_DETAIL_URL);
    await clickInstanceTab(page);
    await searchInstance(page, VM_NAME);
    await expect(page.getByRole('row', { name: new RegExp(VM_NAME) }).first()).toBeVisible({ timeout: 10000 });
  });

  test('鲲鹏-kubevirt-IT-014 - 创建 VirtualMachine 时编辑器显示模板 YAML 内容', {
    tag: ['@v26.06', '@kubevirt'],
  }, async ({ page }) => {
    await page.getByRole('button', { name: /创\s*建/ }).click();
    await page.waitForURL(/createCR/, { timeout: 10000 });
    const content = await getEditorContent(page);
    expect(content).toContain('apiVersion: kubevirt.io/v1');
    expect(content).toContain('kind: VirtualMachine');
    await cancelCreateOrEdit(page);
  });

  test('鲲鹏-kubevirt-IT-013 - VirtualMachine 开机操作验证请求 URL 正确', {
    tag: ['@v26.06', '@kubevirt'],
  }, async ({ page }) => {
    await searchInstance(page, VM_NAME);
    await expect(page.getByRole('row', { name: new RegExp(VM_NAME) }).first()).toBeVisible({ timeout: 10000 });
    await clickRowAction(page, VM_NAME);
    const [request] = await Promise.all([
      page.waitForRequest(req =>
        req.url().includes(`/virtualmachines/${VM_NAME}/start`) && req.method() === 'PUT'
      ),
      clickStartVm(page),
    ]);
    expect(request).toBeTruthy();
  });

  test('鲲鹏-kubevirt-IT-013 - VirtualMachine 重启操作验证请求 URL 正确', {
    tag: ['@v26.06', '@kubevirt'],
  }, async ({ page }) => {
    await searchInstance(page, VM_NAME);
    await expect(page.getByRole('row', { name: new RegExp(VM_NAME) }).first()).toBeVisible({ timeout: 10000 });
    await clickRowAction(page, VM_NAME);
    const [request] = await Promise.all([
      page.waitForRequest(req =>
        req.url().includes(`/virtualmachines/${VM_NAME}/restart`) && req.method() === 'PUT'
      ),
      clickRestartVm(page),
    ]);
    expect(request).toBeTruthy();
  });

  test('鲲鹏-kubevirt-IT-013 - VirtualMachine 关机操作验证请求 URL 正确', {
    tag: ['@v26.06', '@kubevirt'],
  }, async ({ page }) => {
    await searchInstance(page, VM_NAME);
    await expect(page.getByRole('row', { name: new RegExp(VM_NAME) }).first()).toBeVisible({ timeout: 10000 });
    await clickRowAction(page, VM_NAME);
    const [request] = await Promise.all([
      page.waitForRequest(req =>
        req.url().includes(`/virtualmachines/${VM_NAME}/stop`) && req.method() === 'PUT'
      ),
      clickStopVm(page),
    ]);
    expect(request).toBeTruthy();
  });

  test('鲲鹏-kubevirt-IT-012 - 删除 VirtualMachine 成功', {
    tag: ['@v26.06', '@kubevirt'],
  }, async ({ page }) => {
    await searchInstance(page, VM_NAME);
    await expect(page.getByRole('row', { name: new RegExp(VM_NAME) }).first()).toBeVisible({ timeout: 10000 });
    await clickRowAction(page, VM_NAME);
    await clickDeleteInstance(page);
    await confirmDeleteDialog(page);
    await expect(page.getByText(/删除成功/)).toBeVisible({ timeout: 10000 });
  });
});
