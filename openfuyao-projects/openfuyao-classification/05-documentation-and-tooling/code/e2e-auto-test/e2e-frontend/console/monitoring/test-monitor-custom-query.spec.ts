import { expect, test } from '@playwright/test';
import { FUYAO_BASE_URL } from '@/utils/constants';

const CUSTOM_QUERY_PATH = '/container_platform/monitor/monitorDashboard/customize/query';

const TIMEOUT_PAGE = 10000;
const TIMEOUT_ASSERT = 5000;
const WAIT_AFTER_CLICK = 300;

test.describe('监控-自定义查询', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(FUYAO_BASE_URL + CUSTOM_QUERY_PATH);
    await page.waitForLoadState('networkidle');
  });

  test('【监控-078】自定义查询页面，默认无查询条件，页面展示正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await expect(page).toHaveURL(new RegExp(`${CUSTOM_QUERY_PATH.replace(/\//g, '\\/')}`));

    await expect(page.getByText('自定义查询').first()).toBeVisible({ timeout: TIMEOUT_PAGE });
    await expect(page.getByRole('button', { name: '添加查询' })).toBeVisible({ timeout: TIMEOUT_ASSERT });

    const queryBoxArea = page.locator('.monitor_query_box_all');
    await expect(queryBoxArea).toBeVisible({ timeout: TIMEOUT_ASSERT });

    const emptyPlaceholder = page.getByPlaceholder('请输入 prometheus 表达式（Enter 换行，Shift+Enter 查询）');
    await expect(emptyPlaceholder.first()).toBeVisible({ timeout: TIMEOUT_ASSERT });

    await expect(page.getByText(/共\s*\d+\s*条/)).not.toBeVisible();
  });

  test('【监控-079】自定义查询界面，添加查询按钮功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await expect(page.getByText('自定义查询').first()).toBeVisible({ timeout: TIMEOUT_PAGE });

    const addBtn = page.getByRole('button', { name: '添加查询' });
    await expect(addBtn).toBeVisible({ timeout: TIMEOUT_ASSERT });
    await addBtn.click();

    const queryBoxes = page.locator('.monitor_query_box');
    await expect(queryBoxes).toHaveCount(2);
    await expect(page.getByPlaceholder('请输入 prometheus 表达式（Enter 换行，Shift+Enter 查询）').last()).toBeVisible({ timeout: TIMEOUT_ASSERT });
  });

  test('【监控-080】自定义查询页面，添加查询条件输入框能正常输入内容', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await expect(page.getByText('自定义查询').first()).toBeVisible({ timeout: TIMEOUT_PAGE });
    const searchInput = page.getByPlaceholder('请输入 prometheus 表达式（Enter 换行，Shift+Enter 查询）').first();
    await expect(searchInput).toBeVisible({ timeout: TIMEOUT_ASSERT });
    await searchInput.fill('up');
    await expect(searchInput).toHaveValue('up');
  });

  test('【监控-081】自定义查询页面，添加查询条件输入框能正常清空内容', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await expect(page.getByText('自定义查询').first()).toBeVisible({ timeout: TIMEOUT_PAGE });
    const searchInput = page.getByPlaceholder('请输入 prometheus 表达式（Enter 换行，Shift+Enter 查询）').first();
    await expect(searchInput).toBeVisible({ timeout: TIMEOUT_ASSERT });
    await searchInput.fill('up');
    await expect(searchInput).toHaveValue('up');
    await searchInput.clear();
    await expect(searchInput).toHaveValue('');
  });

  test('【监控-082】自定义查询页面，查询条件删除功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await expect(page.getByText('自定义查询').first()).toBeVisible({ timeout: TIMEOUT_PAGE });

    await expect(page.getByRole('button', { name: '添加查询' })).toBeVisible({ timeout: TIMEOUT_ASSERT });
    await page.getByRole('button', { name: '添加查询' }).click();
    await page.waitForTimeout(WAIT_AFTER_CLICK);

    const queryBoxes = page.locator('.monitor_query_box');
    await expect(queryBoxes).toHaveCount(2);

    await queryBoxes.first().locator('.delete').click();
    await page.waitForTimeout(WAIT_AFTER_CLICK);

    await expect(queryBoxes).toHaveCount(1);
  });
});
