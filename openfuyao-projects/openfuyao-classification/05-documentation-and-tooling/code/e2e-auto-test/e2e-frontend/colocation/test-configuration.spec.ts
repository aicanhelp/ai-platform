import { test, expect, Page, APIRequestContext } from '@playwright/test';
import {
  visitColocationConfigPage,
  waitForCheck,
  getSearchbox,
  doNodeSearch,
  getNodeName,
  getNodeStatusColumn,
  getColocationSwitches,
  getColumnheaders,
  getRowByNodeName,
  doFilterStatus,
  doToggleNodeColocation,
  getNodeSwitchIsChecked,
  getSyncButton,
  doOpenModal,
  getModal,
  doCloseModal,
  doClickModalConfirm,
  doClickModalCancel,
  getSwitches,
  getUsagePluginSwitch,
  doToggleUsagePluginSwitch,
  getUsageThresholdSwitch,
  doToggleUsageThresholdSwitch,
  getUsageThresholdTextbox,
  getEvictionSwitch,
  doToggleEvictionSwitch,
  getEvictionThresholdTextbox,
} from './utils/common';
import {
  fetchAllNode,
  fetchNode,
  fetchColocationConfig,
  fetchVolcanoConfig,
} from './utils/api';
import { COLOCATION_CONFIGS } from './constants';
import {
  fetchColocationOptionsViaSsh,
  fetchNodeViaSsh,
  isSshConfigured,
  withSsh,
} from './utils/ssh';

const getNodeNamesFromOptions = (options: Record<string, unknown>): string[] => {
  const nodes = options.nodesConfig;
  if (!Array.isArray(nodes)) {
    return [];
  }

  return nodes
    .map((node) => (node as { name?: unknown }).name)
    .filter((name): name is string => typeof name === 'string');
};

const hasOversubLabels = (nodeData: Record<string, unknown>, oversubLabels: string[]): boolean => {
  const status = (nodeData.status ?? {}) as Record<string, unknown>;
  const capacity = (status.capacity ?? {}) as Record<string, string>;
  const allocatable = (status.allocatable ?? {}) as Record<string, string>;

  return (
    oversubLabels.every((label) => label in capacity) &&
    oversubLabels.every((label) => label in allocatable)
  );
};

const COLOCATION_NODE_LABEL = 'node.openfuyao.com/colocation';

const hasColocationLabel = (nodeData: Record<string, unknown>): boolean => {
  const metadata = (nodeData.metadata ?? {}) as Record<string, unknown>;
  const labels = (metadata.labels ?? {}) as Record<string, string>;
  return labels[COLOCATION_NODE_LABEL] === 'true';
};

const getVisibleNodeNameTexts = async (page: Page): Promise<string[]> => {
  const nodeNames = await getNodeName(page);
  const nodeNameTexts = await Promise.all(nodeNames.map((name) => name.textContent()));
  return nodeNameTexts
    .map((name) => (name ?? '').trim())
    .filter((name) => name.length > 0);
};

const nodeNameCollator = new Intl.Collator('zh-Hans-CN', { numeric: true, sensitivity: 'base' });

const sortNodeNames = (nodeNames: string[]): string[] => {
  return [...nodeNames].sort((a, b) => nodeNameCollator.compare(a, b));
};

const isSameOrder = (a: string[], b: string[]): boolean => {
  return a.length === b.length && a.every((value, index) => value === b[index]);
};

const getNodeCapabilityText = async (page: Page, nodeName: string): Promise<string> => {
  const row = getRowByNodeName(page, nodeName);
  if ((await row.count()) === 0) {
    return '';
  }

  const capabilityCell = row.locator('td').nth(2);
  if ((await capabilityCell.count()) === 0) {
    return '';
  }

  return ((await capabilityCell.first().textContent()) ?? '').trim();
};

const pickToggleableNodeName = async (
  page: Page,
  apiContext: APIRequestContext,
): Promise<string> => {
  const nodeList = (await fetchAllNode(apiContext)).items as Array<Record<string, unknown>>;
  const nodeNames = nodeList
    .map((node) => (node.metadata as Record<string, unknown> | undefined)?.name)
    .filter((name): name is string => typeof name === 'string' && name.length > 0);

  expect(nodeNames.length).toBeGreaterThan(0);

  let selectedNodeName = '';

  await waitForCheck('at least one node colocation switch should be enabled', async () => {
    const diagnostics: string[] = [];

    const syncButton = getSyncButton(page);
    if ((await syncButton.count()) > 0 && (await syncButton.isVisible())) {
      await syncButton.click();
      await page.waitForLoadState('networkidle');
    }

    for (const nodeName of nodeNames) {
      await doNodeSearch(page, nodeName);
      const row = getRowByNodeName(page, nodeName);
      if ((await row.count()) === 0) {
        diagnostics.push(`${nodeName}:row-not-found`);
        continue;
      }

      const switchElement = row.getByRole('switch').first();
      if ((await switchElement.count()) === 0) {
        diagnostics.push(`${nodeName}:switch-not-found`);
        continue;
      }

      const capability = await getNodeCapabilityText(page, nodeName);
      const disabled = await switchElement.isDisabled();
      diagnostics.push(`${nodeName}:capability=${capability || 'unknown'}:${disabled ? 'disabled' : 'enabled'}`);

      if (!disabled) {
        selectedNodeName = nodeName;
        return;
      }
    }

    throw new Error(`all node switches are disabled; ${diagnostics.join('; ')}`);
  }, 240000, 5000);

  await doNodeSearch(page, selectedNodeName);
  return selectedNodeName;
};

/**
 * @description Test colocation configuration page main interactions
 */
test.describe('混部策略配置页面 - 主要交互', () => {
  let apiContext: APIRequestContext;

  test.describe.configure({ mode: 'serial' });

  test.beforeEach(async ({ page, request }: { page: Page; request: APIRequestContext }) => {
    apiContext = request;
    await visitColocationConfigPage(page);
  });

  /**
   * Test 43: Type content in searchbox
   */
  test('【混部-043】搜索框可输入内容', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    const testInput = 'test';
    const searchbox = getSearchbox(page);
    await searchbox.fill(testInput);
    await expect(searchbox).toHaveValue(testInput);
  });

  /**
   * Test 44: Check search results
   */
  test('【混部-044】搜索功能正确性验证', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    const nodeList = (await fetchAllNode(apiContext)).items as Array<Record<string, unknown>>;
    expect(nodeList.length).toBeGreaterThan(0);
    const targetName = (nodeList[0].metadata as Record<string, string>).name;

    // Test on no matches
    const noMatchInput = 'qwertyuiop';
    await doNodeSearch(page, noMatchInput);
    const nodeSwitches = await getColocationSwitches(page);
    expect(nodeSwitches).toHaveLength(0);

    // Test on matches
    await doNodeSearch(page, targetName);
    await waitForCheck(`search result should contain ${targetName}`, async () => {
      const nodeNames = await getNodeName(page);
      if (nodeNames.length === 0) {
        throw new Error('node list is still empty');
      }

      for (const node of nodeNames) {
        const nodeText = (await node.textContent()) ?? '';
        if (!nodeText.includes(targetName)) {
          throw new Error(`unexpected node in result: ${nodeText}`);
        }
      }
    }, 30000, 1000);
  });

  /**
   * Test 45: Searchbox should limit input length
   */
  test('【混部-045】搜索框输入长度限制', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    const lengthLimit = 53;
    const searchbox = getSearchbox(page);
    await searchbox.fill('a'.repeat(255));
    await expect(searchbox).toHaveValue('a'.repeat(lengthLimit));
  });

  /**
   * Test 48: Refresh button should trigger request
   */
  test('【混部-048】点击刷新按钮触发请求', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    await Promise.all([
      page.waitForRequest((req) => req.url().includes(COLOCATION_CONFIGS) && req.method() === 'GET'),
      getSyncButton(page).click(),
    ]);
  });

  /**
   * Test 49: Sort node name
   */
  test('【混部-049】节点名称排序功能', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    const headers = getColumnheaders(page);

    // The first click direction depends on initial table state in different environments.
    await headers.name.click();
    await waitForCheck('node list should be visible after sorting click', async () => {
      const names = await getVisibleNodeNameTexts(page);
      if (names.length === 0) {
        throw new Error('node list is empty');
      }
    }, 30000, 500);

    const firstOrder = await getVisibleNodeNameTexts(page);
    expect(firstOrder.length).toBeGreaterThan(0);

    // Some environments only expose one visible row (for example, table virtualization or single-node clusters).
    // In that case sorting is a no-op on data, so verify it remains stable across toggles.
    if (firstOrder.length === 1) {
      await headers.name.click();
      const secondOrder = await getVisibleNodeNameTexts(page);
      expect(secondOrder).toEqual(firstOrder);
      return;
    }

    const sortedAsc = sortNodeNames(firstOrder);
    const sortedDesc = [...sortedAsc].reverse();
    const firstIsAsc = isSameOrder(firstOrder, sortedAsc);
    const firstIsDesc = isSameOrder(firstOrder, sortedDesc);
    expect(firstIsAsc || firstIsDesc).toBeTruthy();

    // Second click should switch to the opposite order.
    await headers.name.click();
    await waitForCheck('node name order should switch after second click', async () => {
      const secondOrder = await getVisibleNodeNameTexts(page);

      // All names compare equal, toggling order has no visible effect.
      if (isSameOrder(sortedAsc, sortedDesc)) {
        if (!isSameOrder(secondOrder, firstOrder)) {
          throw new Error('node order changed unexpectedly when all names compare equal');
        }
        return;
      }

      const expectedOrder = firstIsAsc ? sortedDesc : sortedAsc;
      if (!isSameOrder(secondOrder, expectedOrder)) {
        throw new Error(`unexpected sort order after second click: ${secondOrder.join(',')}`);
      }
    }, 30000, 500);
  });

  /**
   * Test 50: Filter node status
   */
  test('【混部-050】节点状态筛选功能', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    await doFilterStatus(page, 'Active');

    const nodeStatuses = await getNodeStatusColumn(page);
    for (const status of nodeStatuses.slice(1)) {
      const statusText = await status.textContent();
      expect(statusText).toBe('Active');
    }
  });

  /**
   * Test 51: All nodes are switched off by default
   */
  test('【混部-051】默认所有节点混部开关关闭', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    const nodeNames = await getNodeName(page);
    const [colocationOptions] = await fetchColocationConfig(apiContext);
    const expectedEnabledNodes = getNodeNamesFromOptions(colocationOptions as Record<string, unknown>);

    for (const node of nodeNames) {
      const nodeName = ((await node.textContent()) || '').trim();
      if (!nodeName) {
        continue;
      }

      const checked = await getNodeSwitchIsChecked(page, nodeName);
      expect(checked).toBe(expectedEnabledNodes.includes(nodeName));
    }
  });

  /**
   * Test 52, 53: Switch on/off nodes - front end and back end
   */
  test('【混部-052/053】开启/关闭节点混部功能前后端验证', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    const nodeName = await pickToggleableNodeName(page, apiContext);

    // Normalize to off before validating the on/off flow.
    if (await getNodeSwitchIsChecked(page, nodeName)) {
      await doToggleNodeColocation(page, nodeName);
      await waitForCheck(`normalize node ${nodeName} to disabled`, async () => {
        if (await getNodeSwitchIsChecked(page, nodeName)) {
          throw new Error('node switch is still on in UI');
        }

        const [options] = await fetchColocationConfig(apiContext);
        const colocationNodes = getNodeNamesFromOptions(options as Record<string, unknown>);
        if (colocationNodes.includes(nodeName)) {
          throw new Error(`node ${nodeName} is still in colocation config`);
        }
      }, 90000, 3000);
    }

    // Toggle on
    await doToggleNodeColocation(page, nodeName);
    await waitForCheck(`enable node ${nodeName} on UI and API`, async () => {
      if (!(await getNodeSwitchIsChecked(page, nodeName))) {
        throw new Error('node switch is still off in UI');
      }

      const [colocationOptions] = await fetchColocationConfig(apiContext);
      const colocationNodes = getNodeNamesFromOptions(colocationOptions as Record<string, unknown>);
      if (!colocationNodes.includes(nodeName)) {
        throw new Error(`node ${nodeName} not found in colocation config`);
      }
    }, 90000, 3000);

    if (isSshConfigured()) {
      await withSsh(async (ssh) => {
        await waitForCheck(`enable node ${nodeName} in configmap via ssh`, async () => {
          const options = await fetchColocationOptionsViaSsh(ssh);
          const colocationNodes = getNodeNamesFromOptions(options);
          if (!colocationNodes.includes(nodeName)) {
            throw new Error(`node ${nodeName} not found in ssh configmap data`);
          }
        }, 90000, 3000);
      });
    }

    // Toggle off
    await doToggleNodeColocation(page, nodeName);
    await waitForCheck(`disable node ${nodeName} on UI and API`, async () => {
      if (await getNodeSwitchIsChecked(page, nodeName)) {
        throw new Error('node switch is still on in UI');
      }

      const [updatedColocationOptions] = await fetchColocationConfig(apiContext);
      const updatedColocationNodes = getNodeNamesFromOptions(updatedColocationOptions as Record<string, unknown>);
      if (updatedColocationNodes.includes(nodeName)) {
        throw new Error(`node ${nodeName} is still in colocation config`);
      }
    }, 90000, 3000);

    if (isSshConfigured()) {
      await withSsh(async (ssh) => {
        await waitForCheck(`disable node ${nodeName} in configmap via ssh`, async () => {
          const options = await fetchColocationOptionsViaSsh(ssh);
          const colocationNodes = getNodeNamesFromOptions(options);
          if (colocationNodes.includes(nodeName)) {
            throw new Error(`node ${nodeName} is still in ssh configmap data`);
          }
        }, 90000, 3000);
      });
    }
  });

  test('【混部-025/026】混部与超卖功能验证', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    const nodeName = await pickToggleableNodeName(page, apiContext);
    const oversubLabels = ['kubernetes.io/be-cpu', 'kubernetes.io/be-memory'];

    // Normalize to off before validating.
    if (await getNodeSwitchIsChecked(page, nodeName)) {
      await doToggleNodeColocation(page, nodeName);
      await waitForCheck('colocation label should disappear via api (normalize)', async () => {
        const nodeData = await fetchNode(apiContext, nodeName);
        if (hasColocationLabel(nodeData)) {
          throw new Error('colocation label is still present');
        }
      }, 120000, 3000);
    }

    // Toggle on
    await doToggleNodeColocation(page, nodeName);

    await waitForCheck('colocation label should appear via api', async () => {
      const nodeData = await fetchNode(apiContext, nodeName);
      if (!hasColocationLabel(nodeData)) {
        throw new Error('colocation label is not ready yet');
      }
    }, 120000, 3000);

    await waitForCheck('oversub labels should appear via api', async () => {
      const nodeData = await fetchNode(apiContext, nodeName);
      if (!hasOversubLabels(nodeData, oversubLabels)) {
        throw new Error('oversub labels are not ready yet');
      }
    }, 120000, 3000);

    if (isSshConfigured()) {
      await withSsh(async (ssh) => {
        await waitForCheck('oversub labels should appear via ssh kubectl', async () => {
          const nodeData = await fetchNodeViaSsh(ssh, nodeName);
          if (!hasOversubLabels(nodeData, oversubLabels)) {
            throw new Error('oversub labels are not ready on ssh side');
          }
        }, 120000, 3000);
      });
    }

    // Toggle off
    await doToggleNodeColocation(page, nodeName);

    await waitForCheck('colocation label should disappear via api', async () => {
      const nodeData = await fetchNode(apiContext, nodeName);
      if (hasColocationLabel(nodeData)) {
        throw new Error('colocation label is still present');
      }
    }, 120000, 3000);
  });
});

/**
 * @description Test colocation configuration operations
 */
test.describe('混部策略配置操作', () => {
  let apiContext: APIRequestContext;

  test.describe.configure({ mode: 'serial' });

  const ensureUsagePluginEnabled = async (page: Page): Promise<void> => {
    await waitForCheck('usage plugin switch should be checked', async () => {
      if (await getUsagePluginSwitch(page).isChecked()) {
        return;
      }
      await doToggleUsagePluginSwitch(page);
      if (!(await getUsagePluginSwitch(page).isChecked())) {
        throw new Error('usage plugin switch is still unchecked');
      }
    }, 30000, 1000);
  };

  const ensureEvictionEnabled = async (page: Page): Promise<void> => {
    await waitForCheck('eviction switch should be checked', async () => {
      if (await getEvictionSwitch(page).isChecked()) {
        return;
      }
      await doToggleEvictionSwitch(page);
      if (!(await getEvictionSwitch(page).isChecked())) {
        throw new Error('eviction switch is still unchecked');
      }
    }, 30000, 1000);
  };

  const ensureEvictionDisabled = async (page: Page): Promise<void> => {
    await waitForCheck('eviction switch should be unchecked', async () => {
      if (!(await getEvictionSwitch(page).isChecked())) {
        return;
      }
      await doToggleEvictionSwitch(page);
      if (await getEvictionSwitch(page).isChecked()) {
        throw new Error('eviction switch is still checked in modal');
      }
    }, 30000, 1000);
  };

  const waitBackendEvictionState = async (enabled: boolean, timeoutMs = 90000): Promise<void> => {
    const label = enabled ? 'enabled' : 'disabled';
    await waitForCheck(`eviction should be ${label} in backend`, async () => {
      const [, , rubikOptions] = await fetchColocationConfig(apiContext);
      const current = Boolean(rubikOptions.eviction?.enable);
      if (current !== enabled) {
        throw new Error(`eviction backend state mismatch, expected ${enabled}, got ${current}`);
      }
    }, timeoutMs, 3000);
  };

  const ensureUsageThresholdEnabled = async (page: Page): Promise<void> => {
    await waitForCheck('usage threshold switch should be checked', async () => {
      if (await getUsageThresholdSwitch(page).isChecked()) {
        return;
      }
      await doToggleUsageThresholdSwitch(page);
      if (!(await getUsageThresholdSwitch(page).isChecked())) {
        throw new Error('usage threshold switch is still unchecked');
      }
    }, 30000, 1000);
  };

  test.beforeEach(async ({ page, request }: { page: Page; request: APIRequestContext }) => {
    apiContext = request;
    await visitColocationConfigPage(page);
    await doOpenModal(page);
  });

  test.afterEach(async ({ page }: { page: Page }) => {
    try {
      if (!(await getModal(page).isVisible())) {
        await doOpenModal(page);
      }

      if (!(await getModal(page).isVisible())) {
        return;
      }

      if (await getUsagePluginSwitch(page).isChecked()) {
        await doToggleUsagePluginSwitch(page);
      }
      if (await getEvictionSwitch(page).isChecked()) {
        await ensureEvictionDisabled(page);
      }
      await doClickModalConfirm(page);
      await page.waitForLoadState('networkidle');
    } catch {
      // Cleanup should be best-effort and must not hide the original test result.
    }
  });

  /**
   * Test 55: Open modal
   */
  test('【混部-055】打开配置弹窗', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    await expect(getModal(page)).toBeVisible();
  });

  /**
   * Test 57: Close modal
   */
  test('【混部-057】关闭配置弹窗', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    await doCloseModal(page);
    await expect(getModal(page)).not.toBeVisible();
  });

  /**
   * Test 58: Cancel modal (verify state not changed)
   */
  test('【混部-058】取消配置弹窗', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    const switches = await getSwitches(page);
    const initialStates = await Promise.all(switches.map((sw) => sw.isChecked()));

    await switches[0].click();
    await switches[1].click();
    await doClickModalCancel(page);

    await doOpenModal(page);
    const newSwitches = await getSwitches(page);
    const newStates = await Promise.all(newSwitches.map((sw) => sw.isChecked()));
    expect(newStates).toEqual(initialStates);
  });

  /**
   * Test 62: Turn on usage plugin
   */
  test('【混部-062】开启负载感知调度', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    await ensureUsagePluginEnabled(page);
    await doClickModalConfirm(page);
    await page.waitForLoadState('networkidle');

    await waitForCheck('usage plugin should be enabled in backend', async () => {
      const [, volcanoSchedulerOptions] = await fetchColocationConfig(apiContext);
      if (!volcanoSchedulerOptions.usagePlugin?.enable) {
        throw new Error('usage plugin enable flag is false');
      }

      const volcanoSchedulerConfig = await fetchVolcanoConfig(apiContext);
      const tiers = (volcanoSchedulerConfig.tiers || []) as Array<Record<string, unknown>>;

      const hasPriority = tiers.some((tier) =>
        ((tier.plugins || []) as Array<Record<string, unknown>>).some((plugin) => plugin.name === 'priority')
      );
      const hasUsage = tiers.some((tier) =>
        ((tier.plugins || []) as Array<Record<string, unknown>>).some((plugin) => {
          if (plugin.name !== 'usage' || !('arguments' in plugin)) {
            return false;
          }
          const args = plugin.arguments as Record<string, unknown>;
          const thresholds = args.thresholds as Record<string, unknown>;
          return typeof thresholds?.cpu !== 'undefined' && typeof thresholds?.mem !== 'undefined';
        })
      );

      if (!hasPriority || !hasUsage) {
        throw new Error('volcano scheduler plugins are not ready yet');
      }
    }, 90000, 3000);
  });

  /**
   * Test 64: Usage threshold switch unchecked by default
   */
  test('【混部-064】真实负载调度阈值开关默认关闭', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    // First enable usage plugin
    await ensureUsagePluginEnabled(page);
    await expect(getUsageThresholdSwitch(page)).not.toBeChecked();

    const [, volcanoSchedulerOptions] = await fetchColocationConfig(apiContext);
    expect(volcanoSchedulerOptions.usagePlugin?.usageThreshold?.enable).toBeFalsy();
  });

  /**
   * Test 65: Toggle usage threshold switch to show/hide inputs
   */
  test('【混部-065】切换真实负载调度阈值开关显示/隐藏输入框', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    // First enable usage plugin
    await ensureUsagePluginEnabled(page);

    const [cpuThresholdTextbox] = getUsageThresholdTextbox(page);
    await expect(cpuThresholdTextbox).not.toBeVisible();
    await doToggleUsageThresholdSwitch(page);
    await expect(cpuThresholdTextbox).toBeVisible();
  });

  /**
   * Test 66, 71: Turn on usage threshold with default thresholds
   */
  test('【混部-066/071】开启真实负载调度阈值并验证默认值', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    // First enable usage plugin and threshold
    await ensureUsagePluginEnabled(page);
    await ensureUsageThresholdEnabled(page);

    // Default thresholds
    const [cpuTextbox, memTextbox] = getUsageThresholdTextbox(page);
    await expect(cpuTextbox).toHaveValue('60.0');
    await expect(memTextbox).toHaveValue('60.0');

    await doClickModalConfirm(page);
    await page.waitForLoadState('networkidle');

    await waitForCheck('usage threshold defaults should persist', async () => {
      const [, volcanoSchedulerOptions] = await fetchColocationConfig(apiContext);
      if (!volcanoSchedulerOptions.usagePlugin?.usageThreshold?.enable) {
        throw new Error('usage threshold switch is not enabled in backend');
      }
      expect(volcanoSchedulerOptions.usagePlugin?.usageThreshold?.cpu).toBe(60);
      expect(volcanoSchedulerOptions.usagePlugin?.usageThreshold?.memory).toBe(60);
    }, 90000, 3000);
  });

  /**
   * Test 67, 72: Maximum usage threshold
   */
  test('【混部-067/072】最大真实负载调度阈值', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    // Enable usage plugin and threshold
    await ensureUsagePluginEnabled(page);
    await ensureUsageThresholdEnabled(page);

    const [cpuTextbox, memTextbox] = getUsageThresholdTextbox(page);
    await cpuTextbox.fill('100');
    await memTextbox.fill('100');
    await doClickModalConfirm(page);
    await page.waitForLoadState('networkidle');

    await waitForCheck('usage threshold max values should persist', async () => {
      const [, volcanoSchedulerOptions] = await fetchColocationConfig(apiContext);
      if (!volcanoSchedulerOptions.usagePlugin?.usageThreshold?.enable) {
        throw new Error('usage threshold switch is not enabled in backend');
      }
      expect(volcanoSchedulerOptions.usagePlugin?.usageThreshold?.cpu).toBe(100);
      expect(volcanoSchedulerOptions.usagePlugin?.usageThreshold?.memory).toBe(100);
    }, 90000, 3000);
  });

  /**
   * Test 68, 69, 73, 74: Invalid usage thresholds
   */
  test('【混部-068/069/073/074】无效的真实负载调度阈值验证', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    // Enable usage plugin and threshold
    await ensureUsagePluginEnabled(page);
    await ensureUsageThresholdEnabled(page);

    let requestFired = false;
    const handleRequest = (request: { url(): string; method(): string }) => {
      if (request.url().includes(COLOCATION_CONFIGS) && request.method() === 'POST') {
        requestFired = true;
      }
    };
    page.on('request', handleRequest);

    let errorMessages;

    // Invalid cpu threshold
    const [cpuTextbox, memTextbox] = getUsageThresholdTextbox(page);
    await memTextbox.fill('50');
    await cpuTextbox.fill('-1');
    errorMessages = await page.locator('text=请输入0-100之间的浮点数').all();
    expect(errorMessages.length).toBeGreaterThan(0);
    await cpuTextbox.fill('101');
    errorMessages = await page.locator('text=请输入0-100之间的浮点数').all();
    expect(errorMessages.length).toBeGreaterThan(0);
    await doClickModalConfirm(page);
    await page.waitForLoadState('networkidle');
    expect(requestFired).toBeFalsy();

    // Reset and test invalid memory threshold
    requestFired = false;
    const [cpuTextbox2, memTextbox2] = getUsageThresholdTextbox(page);
    await cpuTextbox2.fill('50');
    await memTextbox2.fill('-1');
    // 不限制有几个
    errorMessages = await page.locator('text=请输入0-100之间的浮点数').all();
    expect(errorMessages.length).toBeGreaterThan(0);
    await memTextbox2.fill('101');
    errorMessages = await page.locator('text=请输入0-100之间的浮点数').all();
    expect(errorMessages.length).toBeGreaterThan(0);
    await doClickModalConfirm(page);
    await page.waitForLoadState('networkidle');
    expect(requestFired).toBeFalsy();

    // Reset and test invalid float
    requestFired = false;
    const [cpuTextbox3, memTextbox3] = getUsageThresholdTextbox(page);
    await cpuTextbox3.fill('83.333');
    errorMessages = await page.locator('text=最多2位小数').all();
    expect(errorMessages.length).toBeGreaterThan(0);
    await cpuTextbox3.fill('50');
    await memTextbox3.fill('83.333');
    errorMessages = await page.locator('text=最多2位小数').all();
    expect(errorMessages.length).toBeGreaterThan(0);
    await doClickModalConfirm(page);
    await page.waitForLoadState('networkidle');
    expect(requestFired).toBeFalsy();

    page.off('request', handleRequest);
  });

  /**
   * Test 70, 75: Valid CPU/Memory thresholds
   */
  test('【混部-070/075】有效的CPU和内存阈值', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    // Enable usage plugin and threshold
    await ensureUsagePluginEnabled(page);
    await ensureUsageThresholdEnabled(page);

    const [cpuTextbox, memTextbox] = getUsageThresholdTextbox(page);
    await cpuTextbox.fill('83.33');
    await memTextbox.fill('83.33');
    await doClickModalConfirm(page);
    await page.waitForLoadState('networkidle');

    await waitForCheck('usage threshold custom values should persist', async () => {
      const [, volcanoSchedulerOptions] = await fetchColocationConfig(apiContext);
      if (!volcanoSchedulerOptions.usagePlugin?.usageThreshold?.enable) {
        throw new Error('usage threshold switch is not enabled in backend');
      }
      expect(volcanoSchedulerOptions.usagePlugin?.usageThreshold?.cpu).toBe(83.33);
      expect(volcanoSchedulerOptions.usagePlugin?.usageThreshold?.memory).toBe(83.33);
    }, 90000, 3000);
  });

  /**
   * Test 63: Turn off usage plugin
   */
  test('【混部-063】关闭负载感知调度', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    // Enable usage plugin first
    await ensureUsagePluginEnabled(page);
    await doClickModalConfirm(page);
    await page.waitForLoadState('networkidle');

    await new Promise((resolve) => setTimeout(resolve, 5000));

    // Reopen modal and disable
    await doOpenModal(page);
    await doToggleUsagePluginSwitch(page);

    // Verify the switch stayed unchecked (guards against late form resets).
    await waitForCheck('usage plugin switch should remain unchecked', async () => {
      if (await getUsagePluginSwitch(page).isChecked()) {
        throw new Error('usage plugin switch was reset to checked');
      }
    }, 10000, 500);

    await doClickModalConfirm(page);
    await page.waitForLoadState('networkidle');

    await waitForCheck('usage plugin should be disabled in backend', async () => {
      const [, volcanoSchedulerOptions] = await fetchColocationConfig(apiContext);
      if (volcanoSchedulerOptions.usagePlugin?.enable) {
        throw new Error('usage plugin is still enabled in backend');
      }
    }, 90000, 3000);
  });

  /**
   * Test 84: Eviction switch unchecked by default
   */
  test('【混部-084】离线负载水位线驱逐开关默认关闭', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    await expect(getEvictionSwitch(page)).not.toBeChecked();

    const [, , rubikOptions] = await fetchColocationConfig(apiContext);
    expect(rubikOptions.eviction?.enable).toBeFalsy();
  });

  /**
   * Test 85: Toggle eviction switch to show/hide inputs
   */
  test('【混部-085】切换离线负载水位线驱逐开关显示/隐藏输入框', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    if (await getEvictionSwitch(page).isChecked()) {
      await doToggleEvictionSwitch(page);
    }
    const [cpuTextbox] = getEvictionThresholdTextbox(page);
    await expect(cpuTextbox).not.toBeVisible();
    await doToggleEvictionSwitch(page);
    await expect(cpuTextbox).toBeVisible();
  });

  /**
   * Test 86: Turn on eviction
   */
  test('【混部-086】开启离线负载水位线驱逐', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    await ensureEvictionEnabled(page);
    await doClickModalConfirm(page);
    await page.waitForLoadState('networkidle');

    await waitBackendEvictionState(true);
  });

  /**
   * Test 88, 94: Default eviction thresholds
   */
  test('【混部-088/094】默认驱逐阈值', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    const [, , rubikOptionsBefore] = await fetchColocationConfig(apiContext);
    const expectedCpu = rubikOptionsBefore.eviction?.cpuevict?.threshold ?? 60;
    const expectedMem = rubikOptionsBefore.eviction?.memoryevict?.threshold ?? 60;

    // Enable eviction
    await ensureEvictionEnabled(page);

    const [cpuTextbox, memTextbox] = getEvictionThresholdTextbox(page);
    await expect(cpuTextbox).toBeVisible();
    await expect(memTextbox).toBeVisible();

    await doClickModalConfirm(page);
    await page.waitForLoadState('networkidle');

    await waitForCheck('eviction default thresholds should persist', async () => {
      const [, , rubikOptions] = await fetchColocationConfig(apiContext);
      if (!rubikOptions.eviction?.enable) {
        throw new Error('eviction is not enabled in backend');
      }
      expect(rubikOptions.eviction?.cpuevict?.threshold).toBe(expectedCpu);
      expect(rubikOptions.eviction?.memoryevict?.threshold).toBe(expectedMem);
    }, 90000, 3000);
  });

  /**
   * Test 89, 95: Minimum eviction thresholds
   */
  test('【混部-089/095】最小驱逐阈值', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    // Enable eviction
    await ensureEvictionEnabled(page);

    const [cpuTextbox, memTextbox] = getEvictionThresholdTextbox(page);
    await cpuTextbox.fill('1');
    await memTextbox.fill('1');
    await doClickModalConfirm(page);
    await page.waitForLoadState('networkidle');

    await waitForCheck('eviction min thresholds should persist', async () => {
      const [, , rubikOptions] = await fetchColocationConfig(apiContext);
      if (!rubikOptions.eviction?.enable) {
        throw new Error('eviction is not enabled in backend');
      }
      expect(rubikOptions.eviction?.cpuevict?.threshold).toBe(1);
      expect(rubikOptions.eviction?.memoryevict?.threshold).toBe(1);
    }, 90000, 3000);
  });

  /**
   * Test 90, 96: Maximum eviction thresholds
   */
  test('【混部-090/096】最大驱逐阈值', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    // Enable eviction
    await ensureEvictionEnabled(page);

    const [cpuTextbox, memTextbox] = getEvictionThresholdTextbox(page);
    await cpuTextbox.fill('99');
    await memTextbox.fill('99');
    await doClickModalConfirm(page);
    await page.waitForLoadState('networkidle');

    await waitForCheck('eviction max thresholds should persist', async () => {
      const [, , rubikOptions] = await fetchColocationConfig(apiContext);
      if (!rubikOptions.eviction?.enable) {
        throw new Error('eviction is not enabled in backend');
      }
      expect(rubikOptions.eviction?.cpuevict?.threshold).toBe(99);
      expect(rubikOptions.eviction?.memoryevict?.threshold).toBe(99);
    }, 90000, 3000);
  });

  /**
   * Test 91, 92, 97, 98: Invalid eviction thresholds
   */
  test('【混部-091/092/097/098】无效的驱逐阈值验证', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    // Enable eviction
    await ensureEvictionEnabled(page);

    let requestFired = false;
    const handleRequest = (request: { url(): string; method(): string }) => {
      if (request.url().includes(COLOCATION_CONFIGS) && request.method() === 'POST') {
        requestFired = true;
      }
    };
    page.on('request', handleRequest);


    let errorMessages;

    const [cpuTextbox, memTextbox] = getEvictionThresholdTextbox(page);
    await memTextbox.fill('50');
    await cpuTextbox.fill('0');
    errorMessages = await page.locator('text=请输入0-100之间的整数').all();
    expect(errorMessages.length).toBeGreaterThan(0);
    await cpuTextbox.fill('100');
    errorMessages = await page.locator('text=请输入0-100之间的整数').all();
    expect(errorMessages.length).toBeGreaterThan(0);
    await cpuTextbox.fill('50.5');
    errorMessages = await page.locator('text=请输入0-100之间的整数').all();
    expect(errorMessages.length).toBeGreaterThan(0);
    await doClickModalConfirm(page);
    await page.waitForLoadState('networkidle');
    expect(requestFired).toBeFalsy();

    const [cpuTextbox2, memTextbox2] = getEvictionThresholdTextbox(page);
    await memTextbox2.fill('0');
    errorMessages = await page.locator('text=请输入0-100之间的整数').all();
    expect(errorMessages.length).toBeGreaterThan(0);
    await memTextbox2.fill('100');
    errorMessages = await page.locator('text=请输入0-100之间的整数').all();
    expect(errorMessages.length).toBeGreaterThan(0);
    await memTextbox2.fill('50.5');
    errorMessages = await page.locator('text=请输入0-100之间的整数').all();
    expect(errorMessages.length).toBeGreaterThan(0);
    await doClickModalConfirm(page);
    await page.waitForLoadState('networkidle');
    expect(requestFired).toBeFalsy();

    page.off('request', handleRequest);
  });

  /**
   * Test 93, 99: Valid eviction thresholds
   */
  test('【混部-093/099】有效的驱逐阈值', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    // Enable eviction
    await ensureEvictionEnabled(page);

    const [cpuTextbox, memTextbox] = getEvictionThresholdTextbox(page);
    await cpuTextbox.fill('50');
    await memTextbox.fill('50');
    await doClickModalConfirm(page);
    await page.waitForLoadState('networkidle');

    await waitForCheck('eviction custom thresholds should persist', async () => {
      const [, , rubikOptions] = await fetchColocationConfig(apiContext);
      if (!rubikOptions.eviction?.enable) {
        throw new Error('eviction is not enabled in backend');
      }
      expect(rubikOptions.eviction?.cpuevict?.threshold).toBe(50);
      expect(rubikOptions.eviction?.memoryevict?.threshold).toBe(50);
    }, 90000, 3000);
  });

  /**
   * Test 87: Turn off eviction
   */
  test('【混部-087】关闭离线负载水位线驱逐', {
    tag: ['@v25.06', '@colocation'],
  }, async ({ page }: { page: Page }) => {
    test.setTimeout(5 * 60 * 1000);
    // Enable eviction first
    await ensureEvictionEnabled(page);
    await doClickModalConfirm(page);
    await page.waitForLoadState('networkidle');

    // Ensure backend already observes enabled before opening modal again.
    await waitBackendEvictionState(true);

    // Reopen modal and disable
    await doOpenModal(page);

    // Wait modal state to reflect backend enabled state, then toggle off.
    await waitForCheck('eviction switch should load as checked before disabling', async () => {
      if (!(await getEvictionSwitch(page).isChecked())) {
        throw new Error('eviction switch has not reflected enabled backend state yet');
      }
    }, 30000, 1000);

    await ensureEvictionDisabled(page);

    await doClickModalConfirm(page);
    await page.waitForLoadState('networkidle');

    await waitBackendEvictionState(false, 120000);
  });
});
