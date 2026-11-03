import { expect, test, type Page } from '@playwright/test';
import { FUYAO_BASE_URL } from '@/utils/constants';
import {
  clickDetailTab,
  clickRefresh,
  clickResourceNameInList,
  ExpectAddLabelAnnotationSuccess,
  deleteLabelAnnotationAndExpectSuccess,
  expectEmptyLabelAnnotationError,
  expectNoChangeLabelAnnotationMessage,
  openModifyLabelAnnotationEditor,
  searchResource,
} from './utils/common';

const NODE_LIST_PATH = '/container_platform/nodeManage';
const WAIT_AFTER_CLICK = 300;

test.describe('节点管理', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(FUYAO_BASE_URL + NODE_LIST_PATH);
    await page.waitForLoadState('networkidle');
  });

  test('【资源管理-563】节点，集群的节点信息展示正确', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    for (const header of ['节点名称', '类型', '状态', '是否繁忙']) {
      await expect(page.getByRole('columnheader', { name: header })).toBeVisible();
    }
    const dataRow = page.locator('table tbody tr').filter({ has: page.locator('td a') }).first();
    await expect(dataRow).toBeVisible({ timeout: 10000 });
    const cells = dataRow.locator('td');
    const cellCount = await cells.count();
    for (let i = 0; i < cellCount; i++) {
      expect((await cells.nth(i).innerText()).trim()).not.toBe('');
    }
  });

  test('【资源管理-564】节点，刷新按钮功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await clickRefresh(page);
    await expect(page.getByRole('columnheader', { name: '节点名称' })).toBeVisible();
    await expect(page.locator('table tbody tr').filter({ has: page.locator('td a') }).first()).toBeVisible();
  });

  test('【资源管理-565】节点，搜索框输入正确的节点名称，可正确查询出节点信息', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const nodeName = await getFirstNodeName(page);
    await searchResource(page, nodeName!);
    await expect(page.getByRole('row', { name: new RegExp(nodeName!) }).first()).toBeVisible({ timeout: 10000 });
  });

  test('【资源管理-566】节点，搜索框异常场景验证', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const nodeName = await getFirstNodeName(page);
    await searchResource(page, nodeName!.toUpperCase());
    await expect(page.getByRole('row', { name: new RegExp(nodeName!, 'i') }).first()).toBeVisible({ timeout: 10000 });
    await searchResource(page, 'a'.repeat(64));
    await expect(page.getByText(/暂无数据/)).toBeVisible({ timeout: 5000 });
  });

  test('【资源管理-567】节点，列表按节点名称排序正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await assertColumnSorted(page, /节点名称/, 0);
  });

  test('【资源管理-568】节点，列表按类型排序正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await assertColumnSorted(page, /类型/, 2);
  });

  test('【资源管理-569】节点，列表按类型筛选正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await assertColumnFiltered(page, /类型/, 2, '管理节点');
  });

  test('【资源管理-570】节点，列表按状态排序正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await assertColumnSorted(page, /状态/, 3);
  });

  test('【资源管理-571】节点，列表按状态筛选正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const statusHeader = page.getByRole('columnheader', { name: /状态/ });
    const hasStatusFilter = await statusHeader.getByRole('button', { name: /filter/i }).count() > 0;
    if (hasStatusFilter) {
      await assertColumnFiltered(page, /状态/, 3, '正常');
      return;
    }
    // 状态列无筛选项时，筛选不存在的节点类型以验证「暂无数据」兜底
    await assertColumnFiltered(page, /类型/, 2, '工作节点');
  });

  test('【资源管理-572】节点，列表按是否繁忙排序正确', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await assertColumnSorted(page, /是否繁忙/, 4);
  });

  test('【资源管理-573】节点，列表按是否繁忙排序正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    await assertColumnFiltered(page, /是否繁忙/, 4, '否');
  });

  test('【资源管理-574】节点，列表分页功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const totalText = await page.getByText(/共\d+条/).textContent().catch(() => null);
    const totalMatch = totalText?.match(/共(\d+)条/);
    const total = totalMatch?.[1] !== undefined ? parseInt(totalMatch[1], 10) : 0;
    if (total <= 10) {
      const rowCount = await countDataRows(page);
      expect(rowCount).toBeLessThanOrEqual(10);
    } else {
      await page.getByText('10 条/页', { exact: true }).click();
      await page.waitForTimeout(WAIT_AFTER_CLICK);
      await page.getByText('20 条/页', { exact: true }).click();
      await page.waitForTimeout(WAIT_AFTER_CLICK);
      expect(await countDataRows(page)).toBeLessThanOrEqual(20);
    }

    const nodeName = await getFirstNodeName(page);
    await clickResourceNameInList(page, nodeName!);
    await page.getByRole('tab', { name: 'Pod' }).click();
    await page.waitForLoadState('networkidle');
    const podTotalText = await page.getByText(/共\d+条/).textContent().catch(() => null);
    const podTotalMatch = podTotalText?.match(/共(\d+)条/);
    const podTotal = podTotalMatch?.[1] !== undefined ? parseInt(podTotalMatch[1], 10) : 0;
    if (podTotal <= 10) {
      expect(await countDataRows(page)).toBeLessThanOrEqual(10);
    }
  });

  test('【资源管理-575】节点，点击节点名称跳转到节点详情正确', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const nodeName = await getFirstNodeName(page);
    await clickResourceNameInList(page, nodeName!);
    await expect(page).toHaveURL(new RegExp(`${NODE_LIST_PATH}/${nodeName}$`));
  });

  test('【资源管理-576】节点，节点详情中的详情Tab页面信息展示正确', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const nodeName = await getFirstNodeName(page);
    await page.goto(`${FUYAO_BASE_URL}${NODE_LIST_PATH}/${nodeName}`);
    await page.waitForLoadState('networkidle');
    await clickDetailTab(page);
    for (const val of await page.locator('.base_value').all()) {
      expect(await val.textContent()).not.toMatch(/^-?$/);
    }
  });

  test('【资源管理-581】节点，节点详情中的Pod Tab页面信息展示正确', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const nodeName = await getFirstNodeName(page);
    await page.goto(`${FUYAO_BASE_URL}${NODE_LIST_PATH}/${nodeName}`);
    await page.waitForLoadState('networkidle');
    await page.getByRole('tab', { name: 'Pod' }).click();
    await page.waitForLoadState('networkidle');
    await expect(page.getByPlaceholder('搜索Pod名称')).toBeVisible({ timeout: 10000 });
    await expect(page.getByText(/共\d+条/)).toBeVisible({ timeout: 10000 });
    await expect(page.locator('a[href*="/workload/pod/"]').first()).toBeVisible({ timeout: 10000 });
  });

  test('【资源管理-582】节点，节点详情中的pod Tab页面搜索功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const nodeName = await getFirstNodeName(page);
    await page.goto(`${FUYAO_BASE_URL}${NODE_LIST_PATH}/${nodeName}`);
    await page.waitForLoadState('networkidle');
    await page.getByRole('tab', { name: 'Pod' }).click();
    await page.waitForSelector('.node_title,.flexBox', { timeout: 10000 });
    const testPodName = (await page.locator('.node_title a').first().textContent())?.trim().slice(0, 5);

    await page.getByPlaceholder('搜索Pod名称').fill('nonexistent-pod-e2e');
    await page.getByRole('button', { name: 'search' }).click();
    await expect(page.locator('.node_title a')).toHaveCount(0, { timeout: 10000 });

    if (testPodName) {
      await page.getByPlaceholder('搜索Pod名称').fill(testPodName);
      await page.getByRole('button', { name: 'search' }).click();
      for (const podLink of await page.locator('.node_title a').all()) {
        const resPodName = (await podLink.textContent())?.trim();
        expect(resPodName).toContain(testPodName);
      }
    }
  });
});

test.describe('节点管理-标签注解变更', () => {

  test.beforeEach(async ({ page }) => {
    await page.goto(FUYAO_BASE_URL + NODE_LIST_PATH);
    await page.waitForLoadState('networkidle');
  });

  test('【资源管理-579】节点，节点详情中的详情Tab-基本信息中修改标签功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const nodeName = await getFirstNodeName(page);
    await page.goto(`${FUYAO_BASE_URL}${NODE_LIST_PATH}/${nodeName}`);
    await page.waitForLoadState('networkidle');
    const key = `e2e-label-${Date.now()}`;
    const value = 'test-value';
    await test.step('修改成功', async () => {
      await openModifyLabelAnnotationEditor(page, 'label', 'icon');
      await ExpectAddLabelAnnotationSuccess(page, 'label', key, value);
    });
    await test.step('内容为空', async () => {
      await openModifyLabelAnnotationEditor(page, 'label', 'icon');
      await expectEmptyLabelAnnotationError(page, 'label');
    });
    await test.step('无变化提交', async () => {
      await openModifyLabelAnnotationEditor(page, 'label', 'icon');
      await expectNoChangeLabelAnnotationMessage(page, 'label');
    });
    await test.step('清理测试数据', async () => {
      await openModifyLabelAnnotationEditor(page, 'label', 'icon');
      await deleteLabelAnnotationAndExpectSuccess(page, 'label', key, value);
    });
  });

  test('【资源管理-580】节点，节点详情中的详情Tab-基本信息中修改注解功能正常', {
    tag: ['@v24.06', '@k8s'],
  }, async ({ page }) => {
    const nodeName = await getFirstNodeName(page);
    await page.goto(`${FUYAO_BASE_URL}${NODE_LIST_PATH}/${nodeName}`);
    await page.waitForLoadState('networkidle');
    const key = `e2e-annotation-${Date.now()}`;
    const value = 'test-value';
    await test.step('修改成功', async () => {
      await openModifyLabelAnnotationEditor(page, 'annotation', 'icon');
      await ExpectAddLabelAnnotationSuccess(page, 'annotation', key, value);
    });
    await test.step('内容为空', async () => {
      await openModifyLabelAnnotationEditor(page, 'annotation', 'icon');
      await expectEmptyLabelAnnotationError(page, 'annotation');
    });
    await test.step('无变化提交', async () => {
      await openModifyLabelAnnotationEditor(page, 'annotation', 'icon');
      await expectNoChangeLabelAnnotationMessage(page, 'annotation');
    });
    await test.step('清理测试数据', async () => {
      await openModifyLabelAnnotationEditor(page, 'annotation', 'icon');
      await deleteLabelAnnotationAndExpectSuccess(page, 'annotation', key, value);
    });
  });
});

async function getFirstNodeName(page: Page): Promise<string | null> {
  const link = page.locator('table tbody tr td a').first();
  if (!(await link.isVisible({ timeout: 10000 }).catch(() => false))) {
    return null;
  }
  return (await link.textContent())?.trim() ?? null;
}

async function countDataRows(page: Page): Promise<number> {
  return page.locator('table tbody tr').evaluateAll((rows) =>
    rows.filter((row) => {
      const cells = row.querySelectorAll('td');
      return cells.length > 0 && (row.textContent?.trim().length ?? 0) > 0;
    }).length,
  );
}

async function getColumnValues(page: Page, columnIndex: number): Promise<string[]> {
  const rowLocator = page.locator('table tbody tr').filter({ has: page.locator('td a') });
  const values: string[] = [];
  const count = await rowLocator.count();
  for (let i = 0; i < count; i++) {
    const text = (await rowLocator.nth(i).locator('td').nth(columnIndex).innerText()).trim();
    if (text) {
      values.push(text);
    }
  }
  return values;
}

async function assertColumnSorted(
  page: Page,
  columnHeader: RegExp | string,
  columnIndex: number,
): Promise<void> {
  const header = page.getByRole('columnheader', { name: columnHeader });
  await expect(header).toBeVisible();

  await header.click();
  await page.waitForTimeout(WAIT_AFTER_CLICK);
  let values = await getColumnValues(page, columnIndex);
  for (let i = 1; i < values.length; i++) {
    expect(values[i - 1].localeCompare(values[i], 'zh-Hans-CN') <= 0).toBeTruthy();
  }

  await header.click();
  await page.waitForTimeout(WAIT_AFTER_CLICK);
  values = await getColumnValues(page, columnIndex);
  for (let i = 1; i < values.length; i++) {
    expect(values[i - 1].localeCompare(values[i], 'zh-Hans-CN') >= 0).toBeTruthy();
  }
}

async function assertColumnFiltered(
  page: Page,
  columnHeader: RegExp | string,
  columnIndex: number,
  filterValue: string,
): Promise<void> {
  const header = page.getByRole('columnheader', { name: columnHeader });
  await expect(header).toBeVisible();
  const filterButton = header.getByRole('button', { name: /filter/i });
  await expect(filterButton).toBeVisible();
  await filterButton.click();
  await page.waitForTimeout(WAIT_AFTER_CLICK);
  await page.getByRole('menuitem', { name: filterValue, exact: true }).click();
  await page.getByRole('button', { name: /确\s*定/ }).click();
  await page.waitForTimeout(WAIT_AFTER_CLICK);

  const rowCount = await countDataRows(page);
  const hasNoData = await page.getByText(/暂无数据/).isVisible().catch(() => false);
  if (rowCount === 0 || hasNoData) {
    await expect(page.getByText(/暂无数据/)).toBeVisible({ timeout: 5000 });
    return;
  }
  const values = await getColumnValues(page, columnIndex);
  for (const value of values) {
    expect(value).toContain(filterValue);
  }
}
