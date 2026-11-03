import { Locator, Page } from '@playwright/test';
import {
  COLOCATION_CONFIG_PAGE_URI,
  COLOCATION_MONITOR_PAGE_URI,
  COLOCATION_OVERVIEW_PAGE_URI,
  FUYAO_BASE_URL,
} from '../constants';

const escapeRegExp = (value: string): string => {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
};

export const waitForCheck = async (
  name: string,
  checker: () => Promise<void>,
  timeoutMs = 120000,
  intervalMs = 3000,
): Promise<void> => {
  const deadline = Date.now() + timeoutMs;
  let lastErr: unknown;

  while (Date.now() < deadline) {
    try {
      await checker();
      return;
    } catch (err) {
      lastErr = err;
      await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }
  }

  const detail = lastErr instanceof Error ? lastErr.message : String(lastErr);
  throw new Error(`wait check timeout: ${name}; last error: ${detail}`);
};

export const waitForSustainedCheck = async (
  name: string,
  checker: () => Promise<void>,
  timeoutMs = 180000,
  intervalMs = 5000,
  requiredConsecutive = 2,
): Promise<void> => {
  const deadline = Date.now() + timeoutMs;
  let lastErr: unknown;
  let consecutive = 0;

  while (Date.now() < deadline) {
    try {
      await checker();
      consecutive++;
      if (consecutive >= requiredConsecutive) {
        return;
      }
    } catch (err) {
      lastErr = err;
      consecutive = 0;
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }

  const detail = lastErr instanceof Error ? lastErr.message : String(lastErr);
  throw new Error(`wait check timeout: ${name}; last error: ${detail}`);
};

/** Visit the colocation overview page */
export const visitColocationOverviewPage = async (page: Page): Promise<void> => {
  await page.goto(`${FUYAO_BASE_URL}${COLOCATION_OVERVIEW_PAGE_URI}`);
};

/** Visit the colocation config page */
export const visitColocationConfigPage = async (page: Page): Promise<void> => {
  await waitForCheck('colocation config page is ready', async () => {
    // First try direct route.
    await page.goto(`${FUYAO_BASE_URL}${COLOCATION_CONFIG_PAGE_URI}`, { waitUntil: 'domcontentloaded' });
    await page.waitForLoadState('networkidle');

    const searchbox = getSearchbox(page);
    if ((await searchbox.count()) > 0 && (await searchbox.isVisible())) {
      return;
    }

    // Then try navigation from menu if direct route is redirected.
    await page.goto(FUYAO_BASE_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForLoadState('networkidle');
    const link = page.getByRole('link', { name: '混部策略配置' }).first();
    if ((await link.count()) > 0) {
      await link.click();
      await page.waitForLoadState('networkidle');
    }

    if ((await searchbox.count()) > 0 && (await searchbox.isVisible())) {
      return;
    }

    const modalButton = page.getByRole('button', { name: /混部策略参数配置/ }).first();
    if ((await modalButton.count()) > 0 && (await modalButton.isVisible())) {
      return;
    }

    throw new Error('searchbox and config button are both not visible');
  }, 90000, 3000);
};

/** Visit the colocation monitor page */
export const visitColocationMonitorPage = async (page: Page): Promise<void> => {
  await waitForCheck('colocation monitor page is ready', async () => {
    await page.goto(`${FUYAO_BASE_URL}${COLOCATION_MONITOR_PAGE_URI}`, { waitUntil: 'domcontentloaded' });
    await page.waitForLoadState('networkidle');

    // 等待tab出现
    const clusterTab = page.getByRole('tab', { name: '集群级混部监控' });
    if ((await clusterTab.count()) === 0) {
      throw new Error('cluster monitor tab is not present on page');
    }
    if (!(await clusterTab.isVisible({ timeout: 5000 }))) {
      throw new Error('cluster monitor tab is not visible');
    }
  }, 60000, 3000);
};

// ===== Overview Page Actions =====

/** Get the sidemenu link for '概览' */
export const getSidemenuLinkOverview = (page: Page): Locator => {
  return page.getByRole('link', { name: '概览', exact: true });
};

/** Get the sidemenu link for '混部策略配置' */
export const getSidemenuLinkConfiguration = (page: Page): Locator => {
  return page.getByRole('link', { name: '混部策略配置', exact: true });
};

/** Get the sidemenu link for '混部监控' */
export const getSidemenuLinkMonitor = (page: Page): Locator => {
  return page.getByRole('link', { name: '混部监控', exact: true });
};

/** Click the question mark icon for info section 1 (环境准备) */
export const doClickInfo1 = async (page: Page): Promise<void> => {
  await page.getByText('1环境准备').getByLabel('question-circle').click();
};

/** Get the drawer title for info section 1 */
export const getDrawerTitle1 = (page: Page): Locator => {
  return page.getByRole('heading', { name: '环境准备' });
};

/** Click the question mark icon for info section 3 (工作负载部署) */
export const doClickInfo3 = async (page: Page): Promise<void> => {
  await page.getByText('3工作负载部署').getByLabel('question-circle').click();
};

/** Get the drawer title for info section 3 */
export const getDrawerTitle3 = (page: Page): Locator => {
  return page.getByRole('heading', { name: '工作负载部署' });
};

/** Close the drawer */
export const doCloseDrawer = async (page: Page): Promise<void> => {
  await page.getByRole('img', { name: 'close' }).locator('svg').click();
};

/** Click link to colocation configuration page */
export const doClickLink2 = async (page: Page): Promise<void> => {
  await page.getByRole('link', { name: '配置混部策略' }).click();
};

/** Click link to workload deployment page */
export const doClickLink3 = async (page: Page): Promise<void> => {
  await page.getByRole('link', { name: '部署工作负载' }).click();
};

/** Click link to colocation monitor page */
export const doClickLink4 = async (page: Page): Promise<void> => {
  await page.getByRole('link', { name: '查看在离线混部监控' }).click();
};

// ===== Configuration Page Actions =====

/** Get the searchbox for node search */
export const getSearchbox = (page: Page): Locator => {
  return page.getByRole('searchbox', { name: '搜索节点' });
};

/** Perform node search */
export const doNodeSearch = async (page: Page, nodeName: string): Promise<void> => {
  await getSearchbox(page).fill(nodeName);
  await page.getByRole('button', { name: 'search' }).click();
};

/** Get all node name elements */
export const getNodeName = (page: Page): Promise<Locator[]> => {
  return page.locator('table tbody tr td:first-child').all();
};

/** Get all node status column elements */
export const getNodeStatusColumn = (page: Page): Promise<Locator[]> => {
  return page.locator('table tbody tr td:nth-child(4)').all();
};

/** Get all colocation switches */
export const getColocationSwitches = (page: Page): Promise<Locator[]> => {
  return page.getByRole('table').getByRole('switch').all();
};

/** Get column headers */
export const getColumnheaders = (page: Page): Record<string, Locator> => {
  return {
    name: page.getByRole('columnheader', { name: '节点名称' }),
    status: page.getByRole('columnheader', { name: '状态' }),
    ip_addr: page.getByRole('columnheader', { name: 'IP地址' }),
    os: page.getByRole('columnheader', { name: 'OS版本' }),
    colocation_enabled: page.getByRole('columnheader', { name: '开启混部节点' }),
  };
};

/** Filter by node status */
export const doFilterStatus = async (page: Page, status: string): Promise<void> => {
  await page.getByRole('button', { name: 'filter' }).click();
  await page.getByRole('menuitem', { name: status, exact: true }).click();
  await page.getByRole('button', { name: /确\s*定/ }).click();
};

/** Get table row by node name */
export const getRowByNodeName = (page: Page, nodeName: string): Locator => {
  // Use regex to match exact node name at the start of the row.
  return page.getByRole('row', { name: new RegExp(`^${escapeRegExp(nodeName)}\\s`) });
};

/** Toggle colocation switch for a specific node */
export const doToggleNodeColocation = async (page: Page, nodeName: string): Promise<void> => {
  // 等待开关变为可用（不是disabled）。
  const row = getRowByNodeName(page, nodeName);
  const switchElement = row.getByRole('switch').first();

  await waitForCheck(`node ${nodeName} row should be visible`, async () => {
    if ((await row.count()) === 0) {
      throw new Error(`node ${nodeName} row is not visible in current table view`);
    }
  }, 60000, 2000);

  await waitForCheck(`node ${nodeName} switch should be enabled`, async () => {
    if ((await switchElement.count()) === 0) {
      throw new Error(`node ${nodeName} switch is not found in row`);
    }

    const isDisabled = await switchElement.isDisabled();
    if (isDisabled) {
      const syncButton = getSyncButton(page);
      if ((await syncButton.count()) > 0 && (await syncButton.isVisible())) {
        await syncButton.click();
        await page.waitForLoadState('networkidle');
      }
      throw new Error(`node ${nodeName} switch is disabled (colocation capability may not be detected yet)`);
    }
  }, 180000, 3000);

  await switchElement.click();
};

/** Get whether the node switch is checked */
export const getNodeSwitchIsChecked = async (page: Page, nodeName: string): Promise<boolean> => {
  const row = getRowByNodeName(page, nodeName);
  return await row.getByRole('switch').isChecked();
};

/** Get the sync/refresh button */
export const getSyncButton = (page: Page): Locator => {
  return page.getByRole('button', { name: 'sync' });
};

/** Open the configuration modal */
export const doOpenModal = async (page: Page): Promise<void> => {
  await waitForCheck('open colocation configuration modal', async () => {
    const button = page.getByRole('button', { name: /混部策略参数配置/ }).first();
    if ((await button.count()) === 0 || !(await button.isVisible())) {
      throw new Error('config modal button is not visible');
    }

    await button.click();

    const modal = getModal(page);
    if (!(await modal.isVisible())) {
      throw new Error('config modal is not visible');
    }
  }, 30000, 1000);
};

/** Get the configuration modal */
export const getModal = (page: Page): Locator => {
  return page.locator('[role="dialog"]:visible', { hasText: '混部策略参数配置' }).first();
};

/** Get a switch in modal by its exact label text */
const getModalSwitchByLabel = (page: Page, label: string): Locator => {
  const modal = getModal(page);
  const escaped = label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return modal.locator('div').filter({ hasText: new RegExp(`^\\s*${escaped}\\s*$`) }).getByRole('switch').first();
};

/** Close the modal */
export const doCloseModal = async (page: Page): Promise<void> => {
  await getModal(page).getByRole('button', { name: 'Close' }).click();
};

/** Click modal confirm button */
export const doClickModalConfirm = async (page: Page): Promise<void> => {
  await getModal(page).getByRole('button', { name: /确\s*定/ }).click();
};

/** Click modal cancel button */
export const doClickModalCancel = async (page: Page): Promise<void> => {
  await getModal(page).getByRole('button', { name: /取\s*消/ }).click();
};

/** Get all switches in the modal */
export const getSwitches = (page: Page): Promise<Locator[]> => {
  return getModal(page).getByRole('switch').all();
};

/** Get the usage plugin switch */
export const getUsagePluginSwitch = (page: Page): Locator => {
  return getModalSwitchByLabel(page, '负载感知调度');
};

/** Toggle the usage plugin switch */
export const doToggleUsagePluginSwitch = async (page: Page): Promise<void> => {
  await getUsagePluginSwitch(page).click();
};

/** Get the usage threshold switch */
export const getUsageThresholdSwitch = (page: Page): Locator => {
  return getModal(page).getByRole('switch', { name: '真实负载调度阈值开关' }).first();
};

/** Toggle the usage threshold switch */
export const doToggleUsageThresholdSwitch = async (page: Page): Promise<void> => {
  await getUsageThresholdSwitch(page).click();
};

/** Get the usage threshold textboxes (cpu, memory) */
export const getUsageThresholdTextbox = (page: Page): [Locator, Locator] => {
  const modal = getModal(page);
  return [
    modal.getByRole('textbox', { name: /CPU真实负载调度阈值/ }).first(),
    modal.getByRole('textbox', { name: /内存真实负载调度阈值/ }).first(),
  ];
};

/** Get the eviction switch */
export const getEvictionSwitch = (page: Page): Locator => {
  return getModalSwitchByLabel(page, '离线负载水位线驱逐');
};

/** Toggle the eviction switch */
export const doToggleEvictionSwitch = async (page: Page): Promise<void> => {
  await getEvictionSwitch(page).click();
};

/** Get the eviction threshold textboxes (cpu, memory) */
export const getEvictionThresholdTextbox = (page: Page): [Locator, Locator] => {
  const modal = getModal(page);
  return [
    modal.getByRole('textbox', { name: /CPU驱逐水位线/ }).first(),
    modal.getByRole('textbox', { name: /内存驱逐水位线/ }).first(),
  ];
};

// ===== Monitor Page Actions =====

/** Switch to cluster monitor tab */
export const doSwitchToClusterMonitor = async (page: Page): Promise<void> => {
  const clusterTab = page.getByRole('tab', { name: '集群级混部监控' });

  // 等待tab变为可点击
  await waitForCheck('cluster monitor tab should be enabled', async () => {
    if ((await clusterTab.count()) === 0) {
      throw new Error('cluster monitor tab not found');
    }
    if (!(await clusterTab.isVisible())) {
      throw new Error('cluster monitor tab is not visible');
    }
    if (!(await clusterTab.isEnabled())) {
      throw new Error('cluster monitor tab is disabled');
    }
  }, 30000, 1000);

  await clusterTab.click();

  // 等待图表加载，需要更长的超时因为数据可能需要时间产生
  await page.waitForLoadState('networkidle');
  await new Promise((resolve) => setTimeout(resolve, 2000));
};

/** Switch to node monitor tab */
export const doSwitchToNodeMonitor = async (page: Page): Promise<void> => {
  const nodeTab = page.getByRole('tab', { name: '节点级混部监控' });

  // 等待tab变为可点击
  await waitForCheck('node monitor tab should be enabled', async () => {
    if ((await nodeTab.count()) === 0) {
      throw new Error('node monitor tab not found');
    }
    if (!(await nodeTab.isVisible())) {
      throw new Error('node monitor tab is not visible');
    }
    if (!(await nodeTab.isEnabled())) {
      throw new Error('node monitor tab is disabled');
    }
  }, 30000, 1000);

  await nodeTab.click();

  // 等待图表加载
  await page.waitForLoadState('networkidle');
  await new Promise((resolve) => setTimeout(resolve, 2000));
};

/** Get monitor paragraphs (chart titles) */
export const getMonitorParagraphs = (page: Page): Promise<Locator[]> => {
  return page.locator('.monitor_namespace').getByRole('paragraph').all();
};
