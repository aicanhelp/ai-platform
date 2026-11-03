import { test, expect, Page, Locator, BrowserContext } from '@playwright/test';
import { TEST_FUYAOURL, TEST_USERNAME, TEST_PASSWORD } from './consts';
import {
  openClusterTerminal,
  waitForTerminalReady,
  openAvailableContainerTerminal,
  expandClusterTerminal,
  closeClusterTerminal,
  waitForTerminalConfirmModal,
  confirmTerminalModalIfPresent,
  openContainerTerminal,
} from './utils/terminal-helpers';
import { newContext } from '@/utils/common';

test.describe('web-terminal-service', () => {
  let context: BrowserContext;
  let page: Page;

  test.beforeEach(async ({ browser }) => {
    // 创建新的浏览器上下文和页面
    context = await newContext(browser);
    page = await context.newPage();

    // 进入登录页并登录
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
  });

  test.afterEach(async () => {
    await page?.close().catch(() => {});
    await context?.close().catch(() => {});
  });

  test('【WebTerminal-001】主集群容器平台点击kubectl终端工具后可打开终端并支持Tab补全kubectl命令', {
    tag: ['@v24.12', '@web-terminal-service'],
  }, async () => {
    await openClusterTerminal(page);
    const terminalPanel = page.locator('.xterm').first();
    const terminalInput = page.locator('.xterm-helper-textarea').first();
    await expect(terminalPanel).toBeVisible({ timeout: 30000 });
    await expect(terminalInput).toBeAttached({ timeout: 30000 });
    await waitForTerminalReady(page);

    await terminalInput.click();
    await page.keyboard.type('kube');
    await page.keyboard.press('Tab');

    await expect.poll(async () => {
      const terminalText = await page.locator('.xterm-rows').first().textContent();
      return terminalText?.replace(/\s+/g, ' ').trim() ?? '';
    }, {
      timeout: 15000,
    }).toContain('kubectl');
  });

  test('【WebTerminal-002】主集群容器平台点击kubectl终端工具后可执行helm list并展示已安装chart信息', {
    tag: ['@v24.12', '@web-terminal-service'],
  }, async () => {
    await openClusterTerminal(page);
    const terminalPanel = page.locator('.xterm').first();
    const terminalInput = page.locator('.xterm-helper-textarea').first();
    await expect(terminalPanel).toBeVisible({ timeout: 30000 });
    await expect(terminalInput).toBeAttached({ timeout: 30000 });
    await waitForTerminalReady(page);

    await terminalInput.click();
    await page.keyboard.type('helm list');
    await page.keyboard.press('Enter');

    await expect.poll(async () => {
      const terminalText = await page.locator('.xterm-rows').first().textContent();
      return terminalText?.replace(/\s+/g, ' ').trim() ?? '';
    }, {
      timeout: 30000,
    }).toContain('helm list');

    await expect.poll(async () => {
      const terminalText = await page.locator('.xterm-rows').first().textContent();
      return terminalText?.replace(/\s+/g, ' ').trim() ?? '';
    }, {
      timeout: 30000,
    }).toMatch(/NAME.*NAMESPACE.*REVISION.*STATUS.*CHART/i);
  });

  test.describe('分辨率显示校验', () => {
    test('【WebTerminal-003】不同分辨率下集群终端中屏显示时显示正常', {
      tag: ['@v24.12', '@web-terminal-service'],
    }, async () => {
      test.setTimeout(5 * 60 * 1000);
      const resolutions = [
        { name: '小于1280', width: 1279, height: 720 },
        { name: '1280-1440', width: 1366, height: 768 },
        { name: '1440', width: 1440, height: 900 },
        { name: '1440-1640', width: 1600, height: 900 },
        { name: '1640', width: 1640, height: 900 },
        { name: '大于1640', width: 1920, height: 1080 },
      ];

      for (const resolution of resolutions) {
        await test.step(`校验${resolution.name}分辨率下集群终端显示正常`, async () => {
          await page.setViewportSize({ width: resolution.width, height: resolution.height });
          await page.goto(`${TEST_FUYAOURL}/container_platform/overview`);
          await page.waitForLoadState('networkidle');

          await openClusterTerminal(page);

          const terminalPanel = page.locator('.xterm').first();
          const terminalInput = page.locator('.xterm-helper-textarea').first();
          const terminalRows = page.locator('.xterm-rows').first();

          await expect(terminalPanel).toBeVisible({ timeout: 30000 });
          await expect(terminalInput).toBeAttached({ timeout: 30000 });
          await expect(terminalRows).toBeVisible({ timeout: 30000 });

          const terminalPanelBox = await terminalPanel.boundingBox();
          const terminalRowsBox = await terminalRows.boundingBox();

          expect(terminalPanelBox).not.toBeNull();
          expect(terminalRowsBox).not.toBeNull();
          expect(terminalPanelBox!.width).toBeGreaterThan(300);
          expect(terminalPanelBox!.height).toBeGreaterThan(120);
          expect(terminalRowsBox!.width).toBeGreaterThan(200);
          expect(terminalRowsBox!.height).toBeGreaterThan(80);

          const terminalText = await terminalRows.textContent();
          expect(terminalText?.trim().length).toBeGreaterThan(0);

          await closeClusterTerminal(page);
        });
      }
    });

    test('【WebTerminal-004】不同分辨率下集群终端全屏显示时显示正常', {
      tag: ['@v24.12', '@web-terminal-service'],
    }, async () => {
      test.setTimeout(5 * 60 * 1000);
      const resolutions = [
        { name: '小于1280', width: 1279, height: 720 },
        { name: '1280-1440', width: 1366, height: 768 },
        { name: '1440', width: 1440, height: 900 },
        { name: '1440-1640', width: 1600, height: 900 },
        { name: '1640', width: 1640, height: 900 },
        { name: '大于1640', width: 1920, height: 1080 },
      ];

      for (const resolution of resolutions) {
        await test.step(`校验${resolution.name}分辨率下集群终端全屏显示正常`, async () => {
          await page.setViewportSize({ width: resolution.width, height: resolution.height });
          await page.goto(`${TEST_FUYAOURL}/container_platform/overview`);
          await page.waitForLoadState('networkidle');

          await openClusterTerminal(page);
          await expandClusterTerminal(page);

          const terminalBox = page.locator('.terminal_box').first();
          const terminalPanel = page.locator('.xterm').first();
          const terminalInput = page.locator('.xterm-helper-textarea').first();
          const terminalRows = page.locator('.xterm-rows').first();

          await expect(terminalBox).toBeVisible({ timeout: 30000 });
          await expect(terminalPanel).toBeVisible({ timeout: 30000 });
          await expect(terminalInput).toBeAttached({ timeout: 30000 });
          await expect(terminalRows).toBeVisible({ timeout: 30000 });

          await expect.poll(async () => (await terminalBox.boundingBox())?.width ?? 0, {
            timeout: 15000,
          }).toBeGreaterThan(resolution.width * 0.8);
          await expect.poll(async () => (await terminalBox.boundingBox())?.height ?? 0, {
            timeout: 15000,
          }).toBeGreaterThanOrEqual((resolution.height * 0.5) - 2);
          await expect.poll(async () => (await terminalRows.boundingBox())?.width ?? 0, {
            timeout: 15000,
          }).toBeGreaterThan(500);
          await expect.poll(async () => (await terminalRows.boundingBox())?.height ?? 0, {
            timeout: 15000,
          }).toBeGreaterThan(180);

          const terminalText = await terminalRows.textContent();
          expect(terminalText?.trim().length).toBeGreaterThan(0);

          await closeClusterTerminal(page);
        });
      }
    });

    test('【WebTerminal-005】不同分辨率下容器终端中屏显示时显示正常', {
      tag: ['@v24.12', '@web-terminal-service'],
    }, async () => {
      test.setTimeout(5 * 60 * 1000);
      const resolutions = [
        { name: '小于1280', width: 1279, height: 720 },
        { name: '1280-1440', width: 1366, height: 768 },
        { name: '1440', width: 1440, height: 900 },
        { name: '1440-1640', width: 1600, height: 900 },
        { name: '1640', width: 1640, height: 900 },
        { name: '大于1640', width: 1920, height: 1080 },
      ];

      for (const resolution of resolutions) {
        await test.step(`校验${resolution.name}分辨率下容器终端中屏显示正常`, async () => {
          await page.setViewportSize({ width: resolution.width, height: resolution.height });
          await page.goto(`${TEST_FUYAOURL}/container_platform/workload/pod`);
          await page.waitForSelector('table', { timeout: 10000 });
          await page.waitForLoadState('networkidle');

          const terminalPanel = page.locator('.xterm').first();
          const terminalInput = page.locator('.xterm-helper-textarea').first();
          const terminalRows = page.locator('.xterm-rows').first();
          await openAvailableContainerTerminal(page, terminalPanel);

          await expect(terminalPanel).toBeVisible({ timeout: 30000 });
          await expect(terminalInput).toBeAttached({ timeout: 30000 });
          await expect(terminalRows).toBeVisible({ timeout: 30000 });

          const terminalPanelBox = await terminalPanel.boundingBox();
          const terminalRowsBox = await terminalRows.boundingBox();

          expect(terminalPanelBox).not.toBeNull();
          expect(terminalRowsBox).not.toBeNull();
          expect(terminalPanelBox!.width).toBeGreaterThan(300);
          expect(terminalPanelBox!.height).toBeGreaterThan(120);
          expect(terminalRowsBox!.width).toBeGreaterThan(200);
          expect(terminalRowsBox!.height).toBeGreaterThan(80);

          const terminalText = await terminalRows.textContent();
          expect(terminalText?.trim().length).toBeGreaterThan(0);

          await closeClusterTerminal(page);
        });
      }
    });

    test('【WebTerminal-006】不同分辨率下容器终端全屏显示时显示正常', {
      tag: ['@v24.12', '@web-terminal-service'],
    }, async () => {
      test.setTimeout(5 * 60 * 1000);
      const resolutions = [
        { name: '小于1280', width: 1279, height: 720 },
        { name: '1280-1440', width: 1366, height: 768 },
        { name: '1440', width: 1440, height: 900 },
        { name: '1440-1640', width: 1600, height: 900 },
        { name: '1640', width: 1640, height: 900 },
        { name: '大于1640', width: 1920, height: 1080 },
      ];

      for (const resolution of resolutions) {
        await test.step(`校验${resolution.name}分辨率下容器终端全屏显示正常`, async () => {
          await page.setViewportSize({ width: resolution.width, height: resolution.height });
          await page.goto(`${TEST_FUYAOURL}/container_platform/workload/pod`);
          await page.waitForSelector('table', { timeout: 10000 });
          await page.waitForLoadState('networkidle');

          const terminalPanel = page.locator('.xterm').first();
          const terminalInput = page.locator('.xterm-helper-textarea').first();
          const terminalRows = page.locator('.xterm-rows').first();
          const terminalBox = page.locator('.terminal_box').first();

          await openAvailableContainerTerminal(page, terminalPanel);
          await expandClusterTerminal(page);

          await expect(terminalBox).toBeVisible({ timeout: 30000 });
          await expect(terminalPanel).toBeVisible({ timeout: 30000 });
          await expect(terminalInput).toBeAttached({ timeout: 30000 });
          await expect(terminalRows).toBeVisible({ timeout: 30000 });

          await expect.poll(async () => (await terminalBox.boundingBox())?.width ?? 0, {
            timeout: 15000,
          }).toBeGreaterThan(resolution.width * 0.8);
          await expect.poll(async () => (await terminalBox.boundingBox())?.height ?? 0, {
            timeout: 15000,
          }).toBeGreaterThanOrEqual((resolution.height * 0.5) - 2);
          await expect.poll(async () => (await terminalRows.boundingBox())?.width ?? 0, {
            timeout: 15000,
          }).toBeGreaterThan(500);
          await expect.poll(async () => (await terminalRows.boundingBox())?.height ?? 0, {
            timeout: 15000,
          }).toBeGreaterThan(180);

          const terminalText = await terminalRows.textContent();
          expect(terminalText?.trim().length).toBeGreaterThan(0);

          await closeClusterTerminal(page);
        });
      }
    });
  });

  test('【WebTerminal-M01-010,012,014】主集群上集群终端基础交互正常', {
    tag: ['@v24.12', '@web-terminal-service'],
  }, async () => {
    const terminalTextValue = 'web-terminal-m01-input';

    await openClusterTerminal(page);
    const terminalPanel = page.locator('.xterm').first();
    const terminalInput = page.locator('.xterm-helper-textarea').first();

    await expect(terminalPanel).toBeVisible({ timeout: 30000 });
    await expect(terminalInput).toBeAttached({ timeout: 30000 });
    await waitForTerminalReady(page);

    await terminalInput.click();
    await page.keyboard.type(terminalTextValue);

    await expect.poll(async () => {
      const terminalText = await page.locator('.xterm-rows').first().textContent();
      return terminalText?.replace(/\s+/g, ' ').trim() ?? '';
    }, {
      timeout: 15000,
    }).toContain(terminalTextValue);

    for (let i = 0; i < terminalTextValue.length; i++) {
      await page.keyboard.press('Backspace');
    }

    await page.keyboard.type('kubectl get ns openfuyao-system');
    await page.keyboard.press('Enter');

    await expect.poll(async () => {
      const terminalText = await page.locator('.xterm-rows').first().textContent();
      return terminalText?.replace(/\s+/g, ' ').trim() ?? '';
    }, {
      timeout: 30000,
    }).toMatch(/openfuyao-system\s*Active/i);
  });

  test('【WebTerminal-M02-007,011,013】主集群上容器终端基础交互正常', {
    tag: ['@v24.12', '@web-terminal-service'],
  }, async () => {
    const terminalTextValue = 'web-terminal-m02-input';
    const terminalPanel = page.locator('.xterm').first();
    const terminalInput = page.locator('.xterm-helper-textarea').first();

    await page.goto(`${TEST_FUYAOURL}/container_platform/workload/pod`);
    await page.waitForSelector('table', { timeout: 10000 });
    await page.waitForLoadState('networkidle');

    await openAvailableContainerTerminal(page, terminalPanel);

    await expect(terminalPanel).toBeVisible({ timeout: 30000 });
    await expect(terminalInput).toBeAttached({ timeout: 30000 });
    await waitForTerminalReady(page);

    await terminalInput.click();
    await page.keyboard.type(terminalTextValue);

    await expect.poll(async () => {
      const terminalText = await page.locator('.xterm-rows').first().textContent();
      return terminalText?.replace(/\s+/g, ' ').trim() ?? '';
    }, {
      timeout: 15000,
    }).toContain(terminalTextValue);

    for (let i = 0; i < terminalTextValue.length; i++) {
      await page.keyboard.press('Backspace');
    }

    await page.keyboard.type('pwd');
    await page.keyboard.press('Enter');

    await expect.poll(async () => {
      const terminalText = await page.locator('.xterm-rows').first().textContent();
      return terminalText?.replace(/\s+/g, ' ').trim() ?? '';
    }, {
      timeout: 30000,
    }).toContain('pwd');

    await expect.poll(async () => {
      const terminalText = await page.locator('.xterm-rows').first().textContent();
      return terminalText?.replace(/\s+/g, ' ').trim() ?? '';
    }, {
      timeout: 30000,
    }).toMatch(/\/artifacts|\//i);
  });

  test('【WebTerminal-M03-033,034,039,042】主集群上集群终端中屏、最小化窗口控制正常', {
    tag: ['@v24.12', '@web-terminal-service'],
  }, async () => {
    const terminalBox = page.locator('.terminal_box').first();
    const terminalPanel = page.locator('.xterm').first();
    const lineButton = page.locator('.terminal_header .terminal_tools [aria-label="line"]').first();
    const expandButton = page.locator('.terminal_header .terminal_tools [aria-label="expand"]').first();
    const compressButton = page.locator('.terminal_header .terminal_tools [aria-label="compress"]').first();
    const closeButton = page.locator('.terminal_header .terminal_tools [aria-label="close"]').first();
    const terminalTitle = page.locator('.terminal_header').first();
    const clusterSelectionItem = page.locator('.container-platform-select-selection-item[title]').first();
    const modalWrap = page.locator('.container-platform-modal-wrap, .ant-modal-wrap').last();

    await openClusterTerminal(page);

    await expect(terminalBox).toBeVisible({ timeout: 30000 });
    await expect(terminalPanel).toBeVisible({ timeout: 30000 });
    const currentClusterName = ((await clusterSelectionItem.getAttribute('title')) || (await clusterSelectionItem.textContent()) || '').replace(/\s+/g, ' ').trim();
    await expect(terminalTitle).toContainText('集群：');
    await expect(terminalTitle).toContainText(currentClusterName);
    await expect(lineButton).toBeVisible({ timeout: 30000 });
    await expect(expandButton).toBeVisible({ timeout: 30000 });
    await expect(closeButton).toBeVisible({ timeout: 30000 });
    await expect(compressButton).toHaveCount(0);

    const middleBox = await terminalBox.boundingBox();
    expect(middleBox).not.toBeNull();
    expect(middleBox!.height).toBeGreaterThan(120);
    const middleHeight = middleBox!.height;

    await lineButton.click();
    await expect.poll(async () => (await terminalBox.boundingBox())?.height ?? 0, {
      timeout: 15000,
    }).toBeLessThanOrEqual(60);

    const restoreMiddleButton = page.locator('.terminal_header .terminal_tools [aria-label="expand-alt"]').first();
    await expect(restoreMiddleButton).toBeVisible({ timeout: 30000 });
    await restoreMiddleButton.click();

    await expect.poll(async () => {
      const currentHeight = (await terminalBox.boundingBox())?.height ?? 0;
      return Math.abs(currentHeight - middleHeight);
    }, {
      timeout: 15000,
    }).toBeLessThanOrEqual(80);

    await closeButton.click();
    await expect(modalWrap).toBeVisible({ timeout: 30000 });
    await confirmTerminalModalIfPresent(page);
    await expect(terminalPanel).toBeHidden({ timeout: 30000 });

    await openClusterTerminal(page);
    await expect(terminalPanel).toBeVisible({ timeout: 30000 });

    await lineButton.click();
    await expect.poll(async () => (await terminalBox.boundingBox())?.height ?? 0, {
      timeout: 15000,
    }).toBeLessThanOrEqual(60);

    await closeButton.click();
    await expect(modalWrap).toBeVisible({ timeout: 30000 });
    await confirmTerminalModalIfPresent(page);
    await expect(terminalPanel).toBeHidden({ timeout: 30000 });
  });

  test('【WebTerminal-M04-040,041,059,060】主集群上集群终端全屏、最小化窗口控制正常', {
    tag: ['@v24.12', '@web-terminal-service'],
  }, async () => {
    const terminalBox = page.locator('.terminal_box').first();
    const terminalPanel = page.locator('.xterm').first();
    const expandButton = page.locator('.terminal_header .terminal_tools [aria-label="expand"]').first();
    const closeButton = page.locator('.terminal_header .terminal_tools [aria-label="close"]').first();
    const modalWrap = page.locator('.container-platform-modal-wrap, .ant-modal-wrap').last();

    await openClusterTerminal(page);
    await expect(terminalPanel).toBeVisible({ timeout: 30000 });

    const middleBox = await terminalBox.boundingBox();
    expect(middleBox).not.toBeNull();
    const middleHeight = middleBox!.height;

    await expandButton.click();
    await expect.poll(async () => (await terminalBox.boundingBox())?.height ?? 0, {
      timeout: 15000,
    }).toBeGreaterThan(middleHeight + 200);

    const fullHeight = (await terminalBox.boundingBox())?.height ?? 0;
    const lineButton = page.locator('.terminal_header .terminal_tools [aria-label="line"]').first();
    const compressButton = page.locator('.terminal_header .terminal_tools [aria-label="compress"]').first();
    await expect(lineButton).toBeVisible({ timeout: 30000 });
    await expect(compressButton).toBeVisible({ timeout: 30000 });
    await expect(closeButton).toBeVisible({ timeout: 30000 });
    expect(fullHeight).toBeGreaterThan(middleHeight);

    await compressButton.click();
    await expect.poll(async () => {
      const currentHeight = (await terminalBox.boundingBox())?.height ?? 0;
      return Math.abs(currentHeight - middleHeight);
    }, {
      timeout: 15000,
    }).toBeLessThanOrEqual(80);

    await expandButton.click();
    await expect.poll(async () => (await terminalBox.boundingBox())?.height ?? 0, {
      timeout: 15000,
    }).toBeGreaterThan(middleHeight + 200);

    await lineButton.click();
    await expect.poll(async () => (await terminalBox.boundingBox())?.height ?? 0, {
      timeout: 15000,
    }).toBeLessThanOrEqual(60);

    await expandButton.click();
    await expect.poll(async () => (await terminalBox.boundingBox())?.height ?? 0, {
      timeout: 15000,
    }).toBeGreaterThan(middleHeight + 200);

    await closeButton.click();
    await expect(modalWrap).toBeVisible({ timeout: 30000 });
    await confirmTerminalModalIfPresent(page);
    await expect(terminalPanel).toBeHidden({ timeout: 30000 });
  });

  test('【WebTerminal-M05-021,022,025,026,032】主集群上容器终端中屏、最小化窗口控制正常', {
    tag: ['@v24.12', '@web-terminal-service'],
  }, async () => {
    test.setTimeout(5 * 60 * 1000);

    const terminalBox = page.locator('.terminal_box').first();
    const terminalPanel = page.locator('.xterm').first();
    const terminalInput = page.locator('.xterm-helper-textarea').first();
    const terminalRows = page.locator('.xterm-rows').first();
    const lineButton = page.locator('.terminal_header .terminal_tools [aria-label="line"]').first();
    const expandButton = page.locator('.terminal_header .terminal_tools [aria-label="expand"]').first();
    const compressButton = page.locator('.terminal_header .terminal_tools [aria-label="compress"]').first();
    const closeButton = page.locator('.terminal_header .terminal_tools [aria-label="close"]').first();
    const terminalTitle = page.locator('.terminal_header').first();

    await page.goto(`${TEST_FUYAOURL}/container_platform/workload/pod`);
    await page.waitForSelector('table', { timeout: 10000 });
    await page.waitForLoadState('networkidle');

    await openAvailableContainerTerminal(page, terminalPanel);

    const containerRow = page.locator('tr.container-platform-table-row').first();
    await expect(containerRow).toBeVisible({ timeout: 30000 });
    const containerName = ((await containerRow.locator('td').first().textContent()) || '').replace(/\s+/g, ' ').trim();

    await expect(terminalBox).toBeVisible({ timeout: 30000 });
    await expect(terminalPanel).toBeVisible({ timeout: 30000 });
    await expect(terminalInput).toBeAttached({ timeout: 30000 });
    await expect(terminalRows).toBeVisible({ timeout: 30000 });
    await expect(terminalTitle).toContainText('容器：');
    await expect(terminalTitle).toContainText(containerName);
    await expect(lineButton).toBeVisible({ timeout: 30000 });
    await expect(expandButton).toBeVisible({ timeout: 30000 });
    await expect(closeButton).toBeVisible({ timeout: 30000 });
    await expect(compressButton).toHaveCount(0);

    const middleBox = await terminalBox.boundingBox();
    const terminalRowsBox = await terminalRows.boundingBox();
    expect(middleBox).not.toBeNull();
    expect(terminalRowsBox).not.toBeNull();
    expect(middleBox!.height).toBeGreaterThan(120);
    expect(terminalRowsBox!.height).toBeGreaterThan(80);
    const middleHeight = middleBox!.height;

    await lineButton.click();
    await expect.poll(async () => (await terminalBox.boundingBox())?.height ?? 0, {
      timeout: 15000,
    }).toBeLessThanOrEqual(60);

    const restoreMiddleButton = page.locator('.terminal_header .terminal_tools [aria-label="expand-alt"]').first();
    await expect(restoreMiddleButton).toBeVisible({ timeout: 30000 });
    await restoreMiddleButton.click();

    await expect.poll(async () => {
      const currentHeight = (await terminalBox.boundingBox())?.height ?? 0;
      return Math.abs(currentHeight - middleHeight);
    }, {
      timeout: 15000,
    }).toBeLessThanOrEqual(80);

    await closeButton.click();
    await waitForTerminalConfirmModal(page);
    await confirmTerminalModalIfPresent(page);
    await expect(terminalPanel).toBeHidden({ timeout: 30000 });

    await openContainerTerminal(page);
    await expect(terminalPanel).toBeVisible({ timeout: 30000 });

    await lineButton.click();
    await expect.poll(async () => (await terminalBox.boundingBox())?.height ?? 0, {
      timeout: 15000,
    }).toBeLessThanOrEqual(60);

    await closeButton.click();
    await waitForTerminalConfirmModal(page);
    await confirmTerminalModalIfPresent(page);
    await expect(terminalPanel).toBeHidden({ timeout: 30000 });
  });

  test('【WebTerminal-M06-023,024,053,054】主集群上容器终端全屏、最小化窗口控制正常', {
    tag: ['@v24.12', '@web-terminal-service'],
  }, async () => {
    test.setTimeout(5 * 60 * 1000);

    const terminalBox = page.locator('.terminal_box').first();
    const terminalPanel = page.locator('.xterm').first();
    const expandButton = page.locator('.terminal_header .terminal_tools [aria-label="expand"]').first();
    const closeButton = page.locator('.terminal_header .terminal_tools [aria-label="close"]').first();

    await page.goto(`${TEST_FUYAOURL}/container_platform/workload/pod`);
    await page.waitForSelector('table', { timeout: 10000 });
    await page.waitForLoadState('networkidle');

    await openAvailableContainerTerminal(page, terminalPanel);
    await expect(terminalPanel).toBeVisible({ timeout: 30000 });

    const middleBox = await terminalBox.boundingBox();
    expect(middleBox).not.toBeNull();
    const middleHeight = middleBox!.height;

    await expandButton.click();
    await expect.poll(async () => (await terminalBox.boundingBox())?.height ?? 0, {
      timeout: 15000,
    }).toBeGreaterThan(middleHeight + 200);

    const lineButton = page.locator('.terminal_header .terminal_tools [aria-label="line"]').first();
    const compressButton = page.locator('.terminal_header .terminal_tools [aria-label="compress"]').first();
    await expect(lineButton).toBeVisible({ timeout: 30000 });
    await expect(compressButton).toBeVisible({ timeout: 30000 });
    await expect(closeButton).toBeVisible({ timeout: 30000 });

    await compressButton.click();
    await expect.poll(async () => {
      const currentHeight = (await terminalBox.boundingBox())?.height ?? 0;
      return Math.abs(currentHeight - middleHeight);
    }, {
      timeout: 15000,
    }).toBeLessThanOrEqual(80);

    await expandButton.click();
    await expect.poll(async () => (await terminalBox.boundingBox())?.height ?? 0, {
      timeout: 15000,
    }).toBeGreaterThan(middleHeight + 200);

    await lineButton.click();
    await expect.poll(async () => (await terminalBox.boundingBox())?.height ?? 0, {
      timeout: 15000,
    }).toBeLessThanOrEqual(60);

    await expandButton.click();
    await expect.poll(async () => (await terminalBox.boundingBox())?.height ?? 0, {
      timeout: 15000,
    }).toBeGreaterThan(middleHeight + 200);

    await closeButton.click();
    await waitForTerminalConfirmModal(page);
    await confirmTerminalModalIfPresent(page);
    await expect(terminalPanel).toBeHidden({ timeout: 30000 });
  });

  test('【WebTerminal-M07-035,036,037】主集群上集群终端中屏帮助说明功能正常', {
    tag: ['@v24.12', '@web-terminal-service'],
  }, async () => {
    const terminalPanel = page.locator('.xterm').first();
    const helpPanel = page.locator('.terminal_example_menu').first();
    const expandHelpButton = page.locator('[aria-label="caret-left"]').first();
    const collapseButton = page.locator('[aria-label="caret-right"]').first();
    const learnMoreLink = page.getByRole('link', { name: '了解更多' }).first();

    await openClusterTerminal(page);
    await expect(terminalPanel).toBeVisible({ timeout: 30000 });
    await expect(helpPanel).toBeVisible({ timeout: 30000 });
    await expect.poll(async () => (await helpPanel.boundingBox())?.width ?? 0, {
      timeout: 15000,
    }).toBeGreaterThan(100);

    await expect(collapseButton).toBeVisible({ timeout: 30000 });
    await collapseButton.click();
    await expect.poll(async () => await helpPanel.getAttribute('class'), {
      timeout: 30000,
    }).toContain('collapse_terminal_example_menu');
    await expect.poll(async () => (await helpPanel.boundingBox())?.width ?? 0, {
      timeout: 15000,
    }).toBeLessThanOrEqual(20);

    await expect(expandHelpButton).toBeVisible({ timeout: 30000 });
    await expandHelpButton.click();
    await expect(helpPanel).toBeVisible({ timeout: 30000 });
    await expect(helpPanel).toContainText('参考以下kubectl命令');
    await expect(learnMoreLink).toBeVisible({ timeout: 30000 });
    await expect.poll(async () => (await helpPanel.boundingBox())?.width ?? 0, {
      timeout: 15000,
    }).toBeGreaterThan(100);

    const [docPage] = await Promise.all([
      page.context().waitForEvent('page'),
      learnMoreLink.click(),
    ]);
    await docPage.waitForLoadState('domcontentloaded');
    await expect.poll(async () => docPage.url(), {
      timeout: 30000,
    }).toContain('https://kubernetes.io/zh-cn/docs/reference/kubectl/');
    await docPage.close();
    await page.bringToFront();

    await expect(collapseButton).toBeVisible({ timeout: 30000 });
    await collapseButton.click();
    await expect.poll(async () => await helpPanel.getAttribute('class'), {
      timeout: 30000,
    }).toContain('collapse_terminal_example_menu');
    await expect.poll(async () => (await helpPanel.boundingBox())?.width ?? 0, {
      timeout: 15000,
    }).toBeLessThanOrEqual(20);
  });

  test('【WebTerminal-M08-055,056,057】主集群上集群终端全屏帮助说明功能正常', {
    tag: ['@v24.12', '@web-terminal-service'],
  }, async () => {
    const terminalBox = page.locator('.terminal_box').first();
    const terminalPanel = page.locator('.xterm').first();
    const expandButton = page.locator('.terminal_header .terminal_tools [aria-label="expand"]').first();
    const lineButton = page.locator('.terminal_header .terminal_tools [aria-label="line"]').first();
    const compressButton = page.locator('.terminal_header .terminal_tools [aria-label="compress"]').first();
    const closeButton = page.locator('.terminal_header .terminal_tools [aria-label="close"]').first();
    const helpPanel = page.locator('.terminal_example_menu').first();
    const expandHelpButton = page.locator('[aria-label="caret-left"]').first();
    const collapseButton = page.locator('[aria-label="caret-right"]').first();
    const learnMoreLink = page.getByRole('link', { name: '了解更多' }).first();

    await openClusterTerminal(page);
    await expect(terminalPanel).toBeVisible({ timeout: 30000 });

    const middleBox = await terminalBox.boundingBox();
    expect(middleBox).not.toBeNull();
    const middleHeight = middleBox!.height;

    await expandButton.click();
    await expect.poll(async () => (await terminalBox.boundingBox())?.height ?? 0, {
      timeout: 15000,
    }).toBeGreaterThan(middleHeight + 200);
    await expect(lineButton).toBeVisible({ timeout: 30000 });
    await expect(compressButton).toBeVisible({ timeout: 30000 });
    await expect(closeButton).toBeVisible({ timeout: 30000 });

    await expect(helpPanel).toBeVisible({ timeout: 30000 });
    await expect.poll(async () => (await helpPanel.boundingBox())?.width ?? 0, {
      timeout: 15000,
    }).toBeGreaterThan(100);

    await expect(collapseButton).toBeVisible({ timeout: 30000 });
    await collapseButton.click();
    await expect.poll(async () => await helpPanel.getAttribute('class'), {
      timeout: 30000,
    }).toContain('collapse_terminal_example_menu');
    await expect.poll(async () => (await helpPanel.boundingBox())?.width ?? 0, {
      timeout: 15000,
    }).toBeLessThanOrEqual(20);

    await expect(expandHelpButton).toBeVisible({ timeout: 30000 });
    await expandHelpButton.click();
    await expect(helpPanel).toBeVisible({ timeout: 30000 });
    await expect(helpPanel).toContainText('参考以下kubectl命令');
    await expect(learnMoreLink).toBeVisible({ timeout: 30000 });
    await expect.poll(async () => (await helpPanel.boundingBox())?.width ?? 0, {
      timeout: 15000,
    }).toBeGreaterThan(100);

    const [docPage] = await Promise.all([
      page.context().waitForEvent('page'),
      learnMoreLink.click(),
    ]);
    await docPage.waitForLoadState('domcontentloaded');
    await expect.poll(async () => docPage.url(), {
      timeout: 30000,
    }).toContain('https://kubernetes.io/zh-cn/docs/reference/kubectl/');
    await docPage.close();
    await page.bringToFront();

    await expect(collapseButton).toBeVisible({ timeout: 30000 });
    await collapseButton.click();
    await expect.poll(async () => await helpPanel.getAttribute('class'), {
      timeout: 30000,
    }).toContain('collapse_terminal_example_menu');
    await expect.poll(async () => (await helpPanel.boundingBox())?.width ?? 0, {
      timeout: 15000,
    }).toBeLessThanOrEqual(20);
  });

  test('【WebTerminal-M09-008,009,015,045,050】主集群上重复开启集群终端冲突处理正常', {
    tag: ['@v24.12', '@web-terminal-service'],
  }, async () => {
    const terminalPanel = page.locator('.xterm').first();
    const terminalTitle = page.locator('.terminal_header').first();
    const terminalInput = page.locator('.xterm-helper-textarea').first();
    const terminalRows = page.locator('.xterm-rows').first();
    const openConflictBody = page.locator('.container-platform-modal-body').filter({ hasText: '开启新终端将关闭已开启的终端' }).first();
    const cancelOpenButton = page.locator('.cancel_btn').filter({ hasText: /取\s*消/ }).first();
    const startNewButton = page.locator('.primary_btn').filter({ hasText: /开\s*启/ }).first();
    const clusterSelectionItem = page.locator('.container-platform-select-selection-item[title]').first();
    const keepOpenMarker = 'm09-keep-open-check';

    await openClusterTerminal(page);
    await expect(terminalPanel).toBeVisible({ timeout: 30000 });
    await waitForTerminalReady(page);

    const originalTitle = ((await terminalTitle.textContent()) || '').replace(/\s+/g, ' ').trim();
    const currentClusterName = ((await clusterSelectionItem.getAttribute('title')) || (await clusterSelectionItem.textContent()) || '').replace(/\s+/g, ' ').trim();
    await terminalInput.click();
    await page.keyboard.type(keepOpenMarker);
    await expect.poll(async () => {
      const text = await terminalRows.textContent();
      return text?.replace(/\s+/g, ' ').trim() ?? '';
    }, {
      timeout: 30000,
    }).toContain(keepOpenMarker);
    const originalContent = ((await terminalRows.textContent()) || '').replace(/\s+/g, ' ').trim();

    await openClusterTerminal(page);
    await expect(openConflictBody).toBeVisible({ timeout: 30000 });
    await expect(openConflictBody).toContainText('开启新终端将关闭已开启的终端');

    await cancelOpenButton.click();
    await expect(openConflictBody).toBeHidden({ timeout: 30000 });

    const titleAfterCancel = ((await terminalTitle.textContent()) || '').replace(/\s+/g, ' ').trim();
    const contentAfterCancel = ((await terminalRows.textContent()) || '').replace(/\s+/g, ' ').trim();
    expect(titleAfterCancel).toBe(originalTitle);
    expect(contentAfterCancel).toBe(originalContent);

    await openClusterTerminal(page);
    await expect(openConflictBody).toBeVisible({ timeout: 30000 });
    await expect(openConflictBody).toContainText('开启新终端将关闭已开启的终端');

    await startNewButton.click();
    await expect(openConflictBody).toBeHidden({ timeout: 30000 });
    await expect(terminalPanel).toBeVisible({ timeout: 30000 });
    await waitForTerminalReady(page);

    const newTitle = ((await terminalTitle.textContent()) || '').replace(/\s+/g, ' ').trim();
    const newContent = ((await terminalRows.textContent()) || '').replace(/\s+/g, ' ').trim();
    expect(newTitle).toContain(`集群：${currentClusterName}`);
    expect(newTitle).toBe(originalTitle);
    expect(newContent).not.toContain(keepOpenMarker);
    expect(newContent).toMatch(/connection\(updated at .*?\)\.\.\.|(~\$|[^\s]+:[^\s]+[#\$]|\/[^\s]*\s*[#\$])\s*$/i);
  });

  test('【WebTerminal-016】主集群上，platform-admin角色用户能正常访问容器终端', {
    tag: ['@v24.12', '@web-terminal-service'],
  }, async () => {
    const terminalPanel = page.locator('.xterm').first();
    const terminalTitle = page.locator('.terminal_header').first();

    await page.goto(`${TEST_FUYAOURL}/container_platform/workload/pod`);
    await page.waitForSelector('table', { timeout: 10000 });
    await page.waitForLoadState('networkidle');
    await openAvailableContainerTerminal(page, terminalPanel);

    const containerRow = page.locator('tr.container-platform-table-row').first();
    await expect(containerRow).toBeVisible({ timeout: 30000 });
    const containerName = ((await containerRow.locator('td').first().textContent()) || '').replace(/\s+/g, ' ').trim();

    await expect(terminalPanel).toBeVisible({ timeout: 30000 });
    await expect(terminalTitle).toContainText('容器：');
    await expect(terminalTitle).toContainText(containerName);
  });

  test('【WebTerminal-027】主集群上，platform-admin角色用户能正常访问集群终端', {
    tag: ['@v24.12', '@web-terminal-service'],
  }, async () => {
    const terminalPanel = page.locator('.xterm').first();
    const terminalTitle = page.locator('.terminal_header').first();
    const clusterSelectionItem = page.locator('.container-platform-select-selection-item[title]').first();
    const currentClusterName = ((await clusterSelectionItem.getAttribute('title')) || (await clusterSelectionItem.textContent()) || '').replace(/\s+/g, ' ').trim();

    await openClusterTerminal(page);
    await expect(terminalPanel).toBeVisible({ timeout: 30000 });
    await waitForTerminalReady(page);
    await expect(terminalTitle).toContainText(`集群：${currentClusterName}`);
  });

});
