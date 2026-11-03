import { expect, test } from '@playwright/test';
import { FUYAO_BASE_URL } from '@/utils/constants';
import {
  MONITOR_GOAL_PATH,
  SERVICEMONITOR_PATH,
  SERVICEMONITOR_CREATE_PATH,
  SERVICE_MONITOR_EXAMPLE_YAML,
  assertMonitorGoalSearch,
  assertPagination,
  assertServiceMonitorSearch,
  assertTableColumnSort,
  compareTime,
  createNamedServiceMonitor,
  deleteNamedServiceMonitorIfExists,
  getTestServiceMonitorRow,
  openTestServiceMonitorDetail,
  uniqueServiceMonitorName,
} from './util';

const PAGINATION_OPTS = {
  defaultPageSize: 10,
  newPageSize: 20,
  pageSizeFromLabel: '10 条/页',
  pageSizeToLabel: '20 条/页',
  enableJumpWhenTotalGt: 20,
} as const;

test.describe('监控-监控目标', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(FUYAO_BASE_URL + MONITOR_GOAL_PATH);
    await page.waitForLoadState('networkidle');
  });

  test('【监控-004】监控目标页面，搜索框能正常输入文本', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const input = page.getByPlaceholder('搜索端点或者命名空间');
    await expect(input).toBeVisible();
    await input.fill('test');
    await expect(input).toHaveValue('test');
  });

  test('【监控-005】监控目标页面，搜索框能正常清空文本', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const input = page.getByPlaceholder('搜索端点或者命名空间');
    await expect(input).toBeVisible();
    await input.fill('test');
    await input.clear();
    await expect(input).toHaveValue('');
  });

  test('【监控-007】监控目标页面，搜索框输入端点查询监控目标，查询正确', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await assertMonitorGoalSearch(page, 0);
  });

  test('【监控-008】监控目标页面，搜索框输入命名空间查询监控目标，查询正确', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await assertMonitorGoalSearch(page, 3);
  });

  test('【监控-009】监控目标列表按健康状态筛选功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const statusHeader = page.getByRole('columnheader', { name: /健康状态/ });
    await expect(statusHeader).toBeVisible();
    await statusHeader.getByRole('button', { name: /filter/i }).click();
    await page.getByRole('menuitem', { name: 'up' }).click();
    await page.waitForTimeout(500);

    const rows = page.locator('table tbody tr');
    const rowCount = await rows.count();
    if (rowCount === 0) {
      test.skip(true, '当前监控目标列表无数据，跳过健康状态筛选校验');
      return;
    }
    for (let i = 1; i < rowCount; i++) {
      const text = await rows.nth(i).locator('td').nth(2).innerText();
      const value = text.trim().toLowerCase();
      if (!value) {
        continue;
      }
      expect(value).toBe('up');
    }
  });

  test('【监控-010】监控目标列表按健康状态排序正确', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await assertTableColumnSort(page, /健康状态/, 2, {
      valueTransform: s => s.toLowerCase(),
    });
  });

  test('【监控-011】监控目标列表按命名空间排序正确', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await assertTableColumnSort(page, /命名空间/, 3);
  });

  test('【监控-012】监控目标列表按最后刮削排序正确', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await assertTableColumnSort(page, /最后刮削/, 4, {
      compare: compareTime,
    });
  });

  test('【监控-013】监控目标分页功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await assertPagination(page, PAGINATION_OPTS);
  });

  test('【监控-014】监控目标页面，按servicemonitor实例按钮跳转正确', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await page.getByRole('button', { name: 'ServiceMonitor实例' }).click();
    await expect(page).toHaveURL(new RegExp(SERVICEMONITOR_PATH.replace(/\//g, '\\/')));
  });
});

test.describe('监控-ServiceMonitor实例', () => {
  test.describe('无需测试实例', () => {
    test.beforeEach(async ({ page }) => {
      await page.goto(FUYAO_BASE_URL + SERVICEMONITOR_PATH);
      await page.waitForLoadState('networkidle');
    });

  test('【监控-015】servicemonitor实例页面，点击创建实例按钮跳转正确', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await page.getByRole('button', { name: '创建实例' }).click();
    await expect(page).toHaveURL(new RegExp(SERVICEMONITOR_CREATE_PATH.replace(/\//g, '\\/')));
  });

  test('【监控-017】servicemonitor实例页面，预置的实例信息修改按钮不可以点击', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const presetRow = page.locator('table tbody tr').filter({ has: page.locator('.disabled_color') }).first();
    if (!(await presetRow.isVisible().catch(() => false))) {
      test.skip(true, '当前列表无预置实例，跳过');
      return;
    }
    await page.getByText('ServiceMonitor实例').first().click();
    await presetRow.locator('.disabled_color').click();
    await expect(page.getByRole('link', { name: '修改' })).not.toBeVisible();
  });

  test('【监控-018】servicemonitor实例页面，预置的实例信息删除按钮不可以点击', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const presetRow = page.locator('table tbody tr').filter({ has: page.locator('.disabled_color') }).first();
    if (!(await presetRow.isVisible().catch(() => false))) {
      test.skip(true, '当前列表无预置实例，跳过');
      return;
    }
    await page.getByText('ServiceMonitor实例').first().click();
    await presetRow.locator('.disabled_color').click();
    await expect(page.getByRole('link', { name: '删除' })).not.toBeVisible();
  });

  test('【监控-021】servicemonitor实例页面，搜索框能正常输入文本', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const input = page.getByPlaceholder('搜索实例名称');
    await expect(input).toBeVisible();
    await input.fill('test');
    await expect(input).toHaveValue('test');
  });

  test('【监控-022】servicemonitor实例页面，搜索框能正常清空文本', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const input = page.getByPlaceholder('搜索实例名称');
    await expect(input).toBeVisible();
    await input.fill('test');
    await input.clear();
    await expect(input).toHaveValue('');
  });

  test('【监控-026】servicemonitor实例列表按命名空间排序正确', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await assertTableColumnSort(page, /命名空间/, 1);
  });

  test('【监控-027】servicemonitor实例列表按创建时间排序正确', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await assertTableColumnSort(page, /创建时间/, 2, {
      compare: compareTime,
    });
  });

  test('【监控-028】servicemonitor实例分页功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await assertPagination(page, PAGINATION_OPTS);
  });

  test('【监控-036】servicemonitor实例创建页面，取消返回实例列表', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await page.getByRole('button', { name: '创建实例' }).click();
    await expect(page).toHaveURL(new RegExp(SERVICEMONITOR_CREATE_PATH.replace(/\//g, '\\/')));
    await page.waitForLoadState('networkidle');
    const footer = page.locator('.tab_container .btn_footer').first();
    await footer.waitFor({ state: 'visible' });
    const cancelBtn = footer.locator('button').first();
    await cancelBtn.scrollIntoViewIfNeeded();
    await Promise.all([
      page.waitForURL(new RegExp(SERVICEMONITOR_PATH.replace(/\//g, '\\/'))),
      cancelBtn.click(),
    ]);
    await expect(page.getByText('ServiceMonitor实例').first()).toBeVisible();
  });

  test('【监控-039】servicemonitor实例创建页面，复制成功且内容完整', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
    await page.getByRole('button', { name: '创建实例' }).click();
    await expect(page).toHaveURL(new RegExp(SERVICEMONITOR_CREATE_PATH.replace(/\//g, '\\/')));
    await page.waitForLoadState('networkidle');
    const copyArea = page.locator('.monitorCreateServiceCopy');
    await expect(copyArea).toBeVisible();
    await copyArea.scrollIntoViewIfNeeded();
    await copyArea.click();
    await expect(page.getByText('复制成功！')).toBeVisible();
    const fromClipboard = await page.evaluate(async () => await navigator.clipboard.readText());
    expect(fromClipboard.replace(/\r\n/g, '\n').trim()).toBe(SERVICE_MONITOR_EXAMPLE_YAML.replace(/\r\n/g, '\n').trim());
  });

  test('【监控-041】servicemonitor实例-详情页面，系统预置资源无操作按钮', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const presetRow = page.locator('table tbody tr').filter({ has: page.locator('.disabled_color') }).first();
    if (!(await presetRow.isVisible().catch(() => false))) {
      test.skip(true, '当前列表无预置实例，跳过');
      return;
    }
    await presetRow.getByRole('link').first().click();
    await page.waitForLoadState('networkidle');
    await expect(page).toHaveURL(/\/serviceMonitor\/.+/);
    await expect(page.getByRole('tab', { name: '详情' })).toBeVisible();
    await expect(page.getByRole('button', { name: /操作/ })).not.toBeVisible();
  });
  });

  test.describe('已有测试实例', () => {
    test.beforeEach(async ({ page }) => {
      const name = uniqueServiceMonitorName();
      await createNamedServiceMonitor(page, name);
      await page.goto(FUYAO_BASE_URL + SERVICEMONITOR_PATH);
      await page.waitForLoadState('networkidle');
    });

    test.afterEach(async ({ page }) => {
      await deleteNamedServiceMonitorIfExists(page, uniqueServiceMonitorName());
    });

    test('【监控-019】servicemonitor实例页面，非预置的实例信息修改按钮可正常点击', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      const nonPresetRow = await getTestServiceMonitorRow(page);
      await page.getByText('ServiceMonitor实例').first().click();
      await nonPresetRow.locator('.primary_color').click();
      await page.getByRole('link', { name: '修改' }).click();
      await expect(page).toHaveURL(/\/serviceMonitor\/.+\/yaml/);
    });

    test('【监控-024】servicemonitor实例页面，搜索框输入名称查询实例，查询正确', {
      tag: ['@v24.06', '@k8s'],
    }, async ({ page }) => {
      await assertServiceMonitorSearch(page, 0);
    });

  test('【监控-042】servicemonitor实例-详情页面，非系统预置资源操作下拉选项正确', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await openTestServiceMonitorDetail(page);
    await expect(page).toHaveURL(/\/serviceMonitor\/.+/);
    await expect(page.getByRole('tab', { name: '详情' })).toBeVisible();
    await expect(page.getByRole('button', { name: /操作/ })).toBeVisible();
    await page.getByRole('button', { name: /操作/ }).click();
    const opMenu = page.locator('.container-platform-popover').last().locator('.column_pop');
    await expect(opMenu).toBeVisible();
    await expect(opMenu.getByText('修改', { exact: true })).toBeVisible();
    await expect(opMenu.getByText('修改标签', { exact: true })).toBeVisible();
    await expect(opMenu.getByText('修改注解', { exact: true })).toBeVisible();
    await expect(opMenu.getByText('删除', { exact: true })).toBeVisible();
  });

  test('【监控-043】servicemonitor实例-详情页面，非系统预置资源点击修改进入YAML页签', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await openTestServiceMonitorDetail(page);
    await expect(page).toHaveURL(/\/serviceMonitor\/.+/);
    await expect(page.getByRole('tab', { name: '详情' })).toBeVisible();
    await page.getByRole('button', { name: /操作/ }).click();
    const opMenu = page.locator('.container-platform-popover').last().locator('.column_pop');
    await expect(opMenu.getByText('修改', { exact: true })).toBeVisible();
    await opMenu.getByText('修改', { exact: true }).click();
    await expect(page.getByRole('tab', { name: 'YAML' })).toHaveAttribute('aria-selected', 'true');
  });

  test('【监控-045】servicemonitor实例详情页面，非系统预置资源编辑标签取消后内容不变', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await openTestServiceMonitorDetail(page);
    await expect(page).toHaveURL(/\/serviceMonitor\/.+/);
    await expect(page.getByRole('tab', { name: '详情' })).toBeVisible();
    const labelBlock = page.locator('.detail_info_box .annotation').nth(0).locator('.key_value');
    const labelsBefore = (await labelBlock.innerText()).trim();
    await page.getByRole('button', { name: /操作/ }).click();
    const opMenu = page.locator('.container-platform-popover').last().locator('.column_pop');
    await opMenu.getByText('修改标签', { exact: true }).click();
    await expect(page.getByText('编辑标签', { exact: true }).first()).toBeVisible();
    await page
      .getByText('编辑标签', { exact: true })
      .first()
      .locator('xpath=ancestor::div[contains(@class,"modal-wrap")][1]')
      .locator('button.cancel_btn')
      .click();
    await expect(page.getByText('编辑标签', { exact: true }).first()).not.toBeVisible();
    expect((await labelBlock.innerText()).trim()).toBe(labelsBefore);
  });

  test('【监控-047】servicemonitor实例详情页面，非系统预置资源编辑注解取消后内容不变', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await openTestServiceMonitorDetail(page);
    await expect(page).toHaveURL(/\/serviceMonitor\/.+/);
    await expect(page.getByRole('tab', { name: '详情' })).toBeVisible();
    const annBlock = page.locator('.detail_info_box .annotation').nth(1).locator('.key_value');
    const annBefore = (await annBlock.innerText()).trim();
    await page.getByRole('button', { name: /操作/ }).click();
    const opMenu = page.locator('.container-platform-popover').last().locator('.column_pop');
    await opMenu.getByText('修改注解', { exact: true }).click();
    await expect(page.getByText('编辑注解', { exact: true }).first()).toBeVisible();
    await page
      .getByText('编辑注解', { exact: true })
      .first()
      .locator('xpath=ancestor::div[contains(@class,"modal-wrap")][1]')
      .locator('button.cancel_btn')
      .click();
    await expect(page.getByText('编辑注解', { exact: true }).first()).not.toBeVisible();
    expect((await annBlock.innerText()).trim()).toBe(annBefore);
  });

  test('【监控-049】servicemonitor实例详情页面，非系统预置资源删除弹窗取消后仍在详情页', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await openTestServiceMonitorDetail(page);
    await expect(page).toHaveURL(/\/serviceMonitor\/.+/);
    const detailUrl = page.url();
    await expect(page.getByRole('tab', { name: '详情' })).toBeVisible();
    await page.getByRole('button', { name: /操作/ }).click();
    const opMenu = page.locator('.container-platform-popover').last().locator('.column_pop');
    await opMenu.getByText('删除', { exact: true }).click();
    const delModal = page.locator('.modal_flex_delete').filter({ has: page.getByText('删除ServiceMonitor') });
    await expect(delModal).toBeVisible();
    await delModal.locator('button.cancel_btn').click();
    await expect(delModal).not.toBeVisible();
    expect(page.url()).toBe(detailUrl);
    await expect(page).toHaveURL(/\/serviceMonitor\/.+/);
    await expect(page.getByRole('tab', { name: '详情' })).toBeVisible();
  });

  test('【监控-056】servicemonitor实例详情YAML页签复制成功且剪贴板为完整 YAML', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await page.context().grantPermissions(['clipboard-read', 'clipboard-write']);
    await openTestServiceMonitorDetail(page);
    await expect(page).toHaveURL(/\/serviceMonitor\/.+/);
    await page.getByRole('tab', { name: 'YAML' }).click();
    await expect(page.locator('.yaml_space_box .cm-content').first()).toBeVisible();
    const copyBtn = page.locator('.yamlCopy');
    await expect(copyBtn).toBeVisible();
    await copyBtn.scrollIntoViewIfNeeded();
    await copyBtn.click();
    await expect(page.getByText('复制成功！')).toBeVisible();
    const fromClipboard = (await page.evaluate(async () => await navigator.clipboard.readText())).replace(/\r\n/g, '\n').trim();
    expect(fromClipboard.length).toBeGreaterThan(80);
    expect(fromClipboard).toMatch(/^apiVersion:\s/m);
    expect(fromClipboard).toMatch(/kind:\s*ServiceMonitor/m);
    expect(fromClipboard).toContain('metadata:');
    expect(fromClipboard).toContain('spec:');
    expect(fromClipboard).toMatch(/spec:[\s\S]+/m);
  });
  });
});
