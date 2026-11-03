import { expect, Locator, Page } from '@playwright/test';
import { TEST_FUYAOURL } from '../consts';

export async function openClusterTerminal(page: Page) {
  const terminalButton = await getVisibleLocator([
    page.getByRole('button', { name: 'kubectl终端工具' }),
    page.getByText('kubectl终端工具', { exact: true }),
    page.getByRole('img', { name: 'code' }),
    page.locator('[aria-label="code"]').first(),
  ]);
  const terminalBox = page.locator('.terminal_box').first();

  for (let attempt = 0; attempt < 2; attempt++) {
    await terminalButton.scrollIntoViewIfNeeded().catch(() => {});
    await terminalButton.click();

    try {
      await expect(terminalBox).toBeVisible({ timeout: 10000 });
      return;
    } catch {
      // Retry once if the first reopen click does not re-mount the terminal.
    }
  }

  await expect(terminalBox).toBeVisible({ timeout: 30000 });
}

export async function waitForTerminalReady(page: Page) {
  await expect.poll(async () => {
    const terminalText = await page.locator('.xterm-rows').first().textContent();
    return terminalText?.replace(/\s+/g, ' ').trim() ?? '';
  }, {
    timeout: 30000,
  }).toMatch(/(~\$|[^\s]+:[^\s]+[#\$]|\/[^\s]*\s*[#\$])\s*$/i);
}

export async function openContainerTerminal(page: Page) {
  const terminalButton = await getVisibleLocator([
    page.locator('tr.container-platform-table-row:visible .common_antd_icon[aria-label="code"]').first(),
    page.locator('tr.container-platform-table-row:visible td').last().locator('.common_antd_icon[aria-label="code"]').first(),
    page.locator('tr.container-platform-table-row').first().locator('.common_antd_icon[aria-label="code"]').first(),
    page.locator('.common_antd_icon[aria-label="code"]').first(),
  ]);
  await terminalButton.scrollIntoViewIfNeeded().catch(() => {});
  await terminalButton.click();
  await page.waitForLoadState('networkidle').catch(() => {});
  await page.waitForTimeout(500);
}

export async function openAvailableContainerTerminal(page: Page, terminalPanel: Locator) {
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

      try {
        await openContainerTerminal(page);
        await confirmTerminalModalIfPresent(page);
        await expect(terminalPanel).toBeVisible({ timeout: 15000 });
        return;
      } catch {
        await page.goto(`${TEST_FUYAOURL}/container_platform/workload/pod`);
        await page.waitForSelector('table', { timeout: 10000 });
        await page.waitForLoadState('networkidle');
      }
    }

    await page.goto(`${TEST_FUYAOURL}/container_platform/workload/pod`);
    await page.waitForSelector('table', { timeout: 10000 });
    await page.waitForLoadState('networkidle');
  }

  throw new Error('未找到可成功打开容器终端的 Running Pod');
}

export async function expandClusterTerminal(page: Page) {
  const fullScreenButton = page.locator('.terminal_header .terminal_tools [aria-label="expand"]').first();
  await expect(fullScreenButton).toBeVisible({ timeout: 30000 });
  await fullScreenButton.click();
}

export async function closeClusterTerminal(page: Page) {
  await page.keyboard.press('Escape').catch(() => {});
  const closeButton = page.locator('.terminal_header .terminal_tools [aria-label="close"]').first();

  if (await closeButton.isVisible().catch(() => false)) {
    await closeButton.click();
    await confirmTerminalModalIfPresent(page);
    await expect(page.locator('.xterm').first()).toBeHidden({ timeout: 10000 }).catch(() => {});
  }
}

export async function waitForPodListReady(page: Page) {
  await expect.poll(async () => {
    const podLinks = await page.locator('tbody tr td a').count();
    const tableText = (((await page.locator('tbody').textContent().catch(() => '')) || '').replace(/\s+/g, ' ').trim());

    if (podLinks > 0) {
      return 'ready';
    }

    if (tableText.includes('暂无数据')) {
      return 'empty';
    }

    return 'loading';
  }, {
    timeout: 30000,
  }).toBe('ready');
}

export async function confirmTerminalModalIfPresent(page: Page) {
  const confirmButton = await getVisibleLocator([
    page.getByRole('button', { name: /确\s*定/ }),
    page.locator('.terminal_console_modal .confirm_btn').first(),
    page.locator('.container-platform-modal .confirm_btn').first(),
  ]).catch(() => null);

  if (confirmButton) {
    await confirmButton.click();
  }
}

export async function waitForTerminalConfirmModal(page: Page) {
  await expect.poll(async () => {
    const button = await getVisibleLocator([
      page.getByRole('button', { name: /确\s*定/ }),
      page.locator('.terminal_console_modal .confirm_btn').first(),
      page.locator('.container-platform-modal .confirm_btn').first(),
    ]).catch(() => null);

    return Boolean(button);
  }, {
    timeout: 30000,
  }).toBe(true);
}

export async function getVisibleLocator(locators: Locator[]): Promise<Locator> {
  for (const locator of locators) {
    if (await locator.first().isVisible().catch(() => false)) {
      return locator.first();
    }
  }

  throw new Error('未找到可见的 kubectl 终端入口');
}
