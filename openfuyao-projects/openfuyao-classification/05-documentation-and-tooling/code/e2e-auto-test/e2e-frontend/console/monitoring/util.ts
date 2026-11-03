import { expect, test, type Locator, type Page } from '@playwright/test';
import { FUYAO_BASE_URL } from '@/utils/constants';
import {
  uniqueResourceName,
  readStaticYaml,
  createFromYamlViaApi,
  deleteResourceViaApi,
} from '@/console/utils/common'

export const TEST_SERVICEMONITOR_NAME = 'test-servicemonitor';
const SERVICEMONITOR_YAML = readStaticYaml('valid-servicemonitor.yaml');
const SERVICEMONITOR_PLURAL = 'servicemonitors';
const SERVICEMONITOR_NS = 'default';
const SERVICEMONITOR_API_GROUP = 'monitoring.coreos.com';

export const uniqueServiceMonitorName = (): string => uniqueResourceName(TEST_SERVICEMONITOR_NAME);

/** 搜索后定位指定名称所在行，避免误取列表第一行 */
export function locateServiceMonitorRow(page: Page, name: string): Locator {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return page.getByRole('row', { name: new RegExp(escaped) }).first();
}

export function locateTestServiceMonitorRow(page: Page): Locator {
  return locateServiceMonitorRow(page, uniqueServiceMonitorName());
}

export const MONITOR_GOAL_PATH = '/container_platform/monitor/monitorGoalManage';
export const SERVICEMONITOR_PATH = MONITOR_GOAL_PATH + '/serviceMonitor';
export const SERVICEMONITOR_CREATE_PATH = SERVICEMONITOR_PATH + '/create';

/** 与 console `exampleYaml.serviceMonitorYamlExample` 一致，用于断言复制内容完整 */
export const SERVICE_MONITOR_EXAMPLE_YAML = `apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: example-servicemonitor
  namespace: example-namespace
spec:
  endpoints:
    - interval: 30s
      port: web
      path: /metrics
  selector:
    matchLabels:
      app: example-app
  namespaceSelector:
    matchNames:
    - example-namespace
`;

/** 搜索、筛选后等待列表刷新 */
const WAIT_LIST_UPDATE = 500;
/** 点击列头、菜单后等待 UI 更新 */
const WAIT_UI_UPDATE = 300;

export async function gotoServiceMonitorList(page: Page): Promise<void> {
  await page.goto(FUYAO_BASE_URL + SERVICEMONITOR_PATH);
  await page.waitForLoadState('networkidle');
}

export async function isTestServiceMonitorInList(page: Page): Promise<boolean> {
  const name = uniqueServiceMonitorName();
  await searchServiceMonitorByName(page, name);
  return locateServiceMonitorRow(page, name).isVisible().catch(() => false);
}

export async function createNamedServiceMonitor(page: Page, name: string): Promise<void> {
  await createFromYamlViaApi(page.request, {
    yaml: SERVICEMONITOR_YAML,
    name,
    plural: SERVICEMONITOR_PLURAL,
    namespace: SERVICEMONITOR_NS,
    apiGroup: SERVICEMONITOR_API_GROUP,
  });
}

export async function createTestServiceMonitor(page: Page): Promise<void> {
  await createNamedServiceMonitor(page, uniqueServiceMonitorName());
}

/** 列表中确保存在非预置测试实例；不存在则通过 API 创建 */
export async function ensureTestServiceMonitor(page: Page): Promise<void> {
  const name = uniqueServiceMonitorName();
  await gotoServiceMonitorList(page);
  await searchServiceMonitorByName(page, name);
  if (await locateServiceMonitorRow(page, name).isVisible().catch(() => false)) {
    return;
  }
  await createNamedServiceMonitor(page, name);
  await gotoServiceMonitorList(page);
  await searchServiceMonitorByName(page, name);
  await expect(
    locateServiceMonitorRow(page, name),
    `应已创建非预置实例 ${name}`,
  ).toBeVisible();
}

/** 先搜索再返回测试实例行；不存在则先创建 */
export async function getTestServiceMonitorRow(page: Page): Promise<Locator> {
  const name = uniqueServiceMonitorName();
  await ensureTestServiceMonitor(page);
  await gotoServiceMonitorList(page);
  await searchServiceMonitorByName(page, name);
  const row = locateServiceMonitorRow(page, name);
  await expect(row, `非预置实例 ${name} 应在列表中可见`).toBeVisible();
  return row;
}

/** 先搜索测试实例，再点击名称进入详情 */
export async function openTestServiceMonitorDetail(page: Page): Promise<void> {
  const row = await getTestServiceMonitorRow(page);
  await row.getByRole('link').first().click();
  await page.waitForLoadState('networkidle');
}

export async function deleteNamedServiceMonitorIfExists(page: Page, name: string): Promise<void> {
  await deleteResourceViaApi(page.request, {
    plural: SERVICEMONITOR_PLURAL,
    name,
    namespace: SERVICEMONITOR_NS,
    apiGroup: SERVICEMONITOR_API_GROUP,
  });
}

export async function deleteTestServiceMonitorIfExists(page: Page): Promise<void> {
  await deleteNamedServiceMonitorIfExists(page, uniqueServiceMonitorName());
}

export async function searchServiceMonitorByName(page: Page, name: string): Promise<void> {
  await page.getByPlaceholder('搜索实例名称').fill(name);
  await page.getByPlaceholder('搜索实例名称').press('Enter');
  await page.waitForTimeout(WAIT_LIST_UPDATE);
}

export async function searchMonitorGoal(page: Page, keyword: string): Promise<void> {
  await page.getByPlaceholder('搜索端点或者命名空间').fill(keyword);
  await page.getByPlaceholder('搜索端点或者命名空间').press('Enter');
  await page.waitForTimeout(WAIT_LIST_UPDATE);
}

export type SortCompare = (a: string, b: string) => number;

const parseTime = (s: string): number => {
  const t = Date.parse(s.replace(' ', 'T'));
  return Number.isNaN(t) ? 0 : t;
};

export const compareTime: SortCompare = (a, b) => parseTime(a) - parseTime(b);

/** 表格按列排序：升序 → 降序，断言数据顺序 */
export async function assertTableColumnSort(
  page: Page,
  columnHeaderName: RegExp | string,
  columnIndex: number,
  options: {
    valueTransform?: (s: string) => string;
    compare?: SortCompare;
  } = {},
): Promise<void> {
  const {
    valueTransform = (s: string) => s,
    compare = (a, b) => a.localeCompare(b),
  } = options;

  const rowLocator = page.locator('table tbody tr');
  const rowCount = await rowLocator.count();
  if (rowCount < 2) {
    test.skip(true, '列表不足两条数据，无法校验排序');
    return;
  }

  const header = page.getByRole('columnheader', { name: columnHeaderName });
  await expect(header).toBeVisible();

  const getValues = async (): Promise<string[]> => {
    const values: string[] = [];
    const n = await rowLocator.count();
    for (let i = 0; i < n; i++) {
      const cellText = await rowLocator.nth(i).locator('td').nth(columnIndex).innerText();
      const v = valueTransform(cellText.trim());
      if (v) values.push(v);
    }
    return values;
  };

  await header.click();
  await page.waitForTimeout(WAIT_UI_UPDATE);
  let values = await getValues();
  for (let i = 1; i < values.length; i++) {
    expect(compare(values[i - 1], values[i]) <= 0).toBeTruthy();
  }

  await header.click();
  await page.waitForTimeout(WAIT_UI_UPDATE);
  values = await getValues();
  for (let i = 1; i < values.length; i++) {
    expect(compare(values[i - 1], values[i]) >= 0).toBeTruthy();
  }
}

/** 监控目标页：无结果搜索 → 清空 → 按首行指定列关键词再搜 */
export async function assertMonitorGoalSearch(page: Page, columnIndex: number): Promise<void> {
  await searchMonitorGoal(page, 'xxxxx');
  await expect(page.getByText('暂无数据')).toBeVisible();
  await searchMonitorGoal(page, '');
  await page.waitForTimeout(WAIT_LIST_UPDATE);

  const dataRows = page.locator('table tbody tr').filter({ has: page.locator('td') });
  const rowCount = await dataRows.count();
  if (rowCount === 0) {
    test.skip(true, '当前监控目标列表无数据，跳过搜索校验');
    return;
  }

  const firstRow = dataRows.nth(1);
  const rawText = await firstRow.locator('td').nth(columnIndex).textContent();
  const toSearch = rawText?.trim() ?? '';
  if (!toSearch) {
    test.skip(true, `第 ${columnIndex} 列为空，跳过本次搜索校验`);
    return;
  }

  await searchMonitorGoal(page, toSearch);
  const escaped = toSearch.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  await expect(page.getByRole('row', { name: new RegExp(escaped, 'i') }).first()).toBeVisible();
}

/** ServiceMonitor 实例页：无结果搜索 → 清空 → 按测试实例名称再搜 */
export async function assertServiceMonitorSearch(page: Page, columnIndex: number): Promise<void> {
  const name = uniqueServiceMonitorName();
  await searchServiceMonitorByName(page, 'xxxxxx');
  await expect(page.getByText('暂无数据')).toBeVisible();
  await searchServiceMonitorByName(page, '');
  await page.waitForTimeout(WAIT_LIST_UPDATE);

  await ensureTestServiceMonitor(page);
  await gotoServiceMonitorList(page);
  await searchServiceMonitorByName(page, name);
  const row = locateServiceMonitorRow(page, name);
  await expect(row).toBeVisible();

  const toSearch = (await row.locator('td').nth(columnIndex).textContent())?.trim() ?? '';
  expect(toSearch, `实例 ${name} 第 ${columnIndex} 列应有可搜索内容`).not.toBe('');
  await searchServiceMonitorByName(page, toSearch);
  await expect(page.getByRole('row', { name: new RegExp(toSearch, 'i') }).first()).toBeVisible();
}

export type PaginationOptions = {
  defaultPageSize: number;
  newPageSize: number;
  pageSizeFromLabel: string;
  pageSizeToLabel: string;
  enableJumpWhenTotalGt?: number;
};

/** 列表分页：校验默认页大小、切换每页条数及可选跳页 */
export async function assertPagination(page: Page, opts: PaginationOptions): Promise<void> {
  const {
    defaultPageSize,
    newPageSize,
    pageSizeFromLabel,
    pageSizeToLabel,
    enableJumpWhenTotalGt,
  } = opts;

  const totalText = await page.getByText(/共\d+条/).textContent();
  const totalMatch = typeof totalText === 'string' ? totalText.match(/共(\d+)条/) : null;
  const total = totalMatch?.[1] !== undefined ? parseInt(totalMatch[1], 10) : 0;

  const countRealRows = async (): Promise<number> => {
    return page.locator('table tbody tr').evaluateAll(rows =>
      rows.filter(r => r.querySelectorAll('td').length > 0).length,
    );
  };

  if (total <= defaultPageSize) {
    const rowCount = await countRealRows();
    expect(rowCount).toBeLessThanOrEqual(defaultPageSize);
    return;
  }

  await page.getByText(pageSizeFromLabel, { exact: true }).click();
  await page.waitForTimeout(WAIT_UI_UPDATE);
  await page.getByText(pageSizeToLabel, { exact: true }).click();
  await page.waitForTimeout(WAIT_UI_UPDATE);

  const rowCountNew = await countRealRows() - 1;
  expect(rowCountNew).toBeLessThanOrEqual(newPageSize);

  if (enableJumpWhenTotalGt !== undefined && total > enableJumpWhenTotalGt) {
    const jumperInput = page.getByRole('textbox', { name: '页' });
    await jumperInput.fill('1');
    await jumperInput.press('Enter');
    await page.waitForTimeout(WAIT_UI_UPDATE);

    const rowCountAfterJump = await countRealRows();
    expect(rowCountAfterJump).toBeGreaterThan(0);
  }
}
