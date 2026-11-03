import { expect, test as base, type Locator, type Page } from '@playwright/test';
import { NodeSSH } from 'node-ssh';
import { FUYAO_BASE_URL } from '@/utils/constants';
import {
  NUMA_MONITOR_PATH,
  NUMA_POLICY_PATH,
  NUMA_RELEASE_NAME,
  NUMA_RELEASE_NAMESPACE,
  NUMA_RELEASE_REPO,
  NUMA_RELEASE_VERSION,
  TEST_KUBELET_CONFIG_PATH,
  TEST_SSH_ADDRESS,
  TEST_SSH_PASSWORD,
  TEST_SSH_USERNAME,
} from './constants';

type NumaWorkerFixtures = {
  ssh: NodeSSH;
};

export const test = base.extend<{}, NumaWorkerFixtures>({
  ssh: [
    async ({}, use) => {
      expect(FUYAO_BASE_URL).not.toEqual('');
      expect(TEST_SSH_ADDRESS).not.toEqual('');
      expect(TEST_SSH_PASSWORD).not.toEqual('');

      const ssh = new NodeSSH();
      await ssh.connect({
        host: TEST_SSH_ADDRESS,
        username: TEST_SSH_USERNAME,
        password: TEST_SSH_PASSWORD,
        readyTimeout: 30000,
      });

      try {
        await use(ssh);
      } finally {
        ssh.dispose();
      }
    },
    { scope: 'worker' },
  ],
});

export { expect };

export async function openPolicyConfigPage(page: Page): Promise<void> {
  await page.route('**/rest/scheduling/v1/numaFast', (route) => {
    route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });

  await waitForCheck(
    `open NUMA policy page ${NUMA_POLICY_PATH}`,
    async () => {
      await page.goto(FUYAO_BASE_URL, { waitUntil: 'domcontentloaded' });
      await page.waitForLoadState('networkidle', { timeout: 20000 });

      const numaMenu = page.getByText('NUMA亲和调度').first();
      if ((await numaMenu.count()) > 0) {
        await numaMenu.click();
      }

      const policyLink = page.getByRole('link', { name: '亲和策略配置' }).first();
      if ((await policyLink.count()) > 0) {
        await policyLink.click();
        await page.waitForLoadState('networkidle', { timeout: 20000 });
      } else {
        await page.goto(`${FUYAO_BASE_URL}${NUMA_POLICY_PATH}`, { waitUntil: 'domcontentloaded' });
        await page.waitForLoadState('networkidle', { timeout: 20000 });
      }

      const panel = page.getByText('numa-aware').first();
      if ((await panel.count()) > 0) {
        return;
      }

      const switches = page.getByRole('switch');
      if ((await switches.count()) > 0) {
        return;
      }

      await page.reload({ waitUntil: 'domcontentloaded' });
      throw new Error('policy panel is not rendered yet');
    },
    120000,
    4000,
  );
}

export async function openMonitorPageAndWait(page: Page): Promise<void> {
  await openNumaPageWithRetry(page, NUMA_MONITOR_PATH, async () => {
    const table = page.locator('table').first();
    if ((await table.count()) > 0 && (await table.isVisible())) {
      return;
    }

    const emptyMsg = page.getByText(/暂无数据/).first();
    if ((await emptyMsg.count()) > 0 && (await emptyMsg.isVisible())) {
      return;
    }

    throw new Error('monitor table is not visible yet');
  });
}

export async function setPolicySwitchByRecordedFlow(
  page: Page,
  policyText: string,
  switchIndex: number,
  expectedEnabled: boolean,
): Promise<void> {
  // 增强的策略文本查找和点击逻辑
  await waitForCheck(`find and click policy text: ${policyText}`, async () => {
    const element = page.getByText(policyText).first();
    if ((await element.count()) === 0) {
      throw new Error(`Policy text "${policyText}" not found on page`);
    }
    if (!(await element.isVisible({ timeout: 5000 }))) {
      throw new Error(`Policy text "${policyText}" is not visible`);
    }
    await element.click();
  }, 30000, 2000);

  const toggle = page.getByRole('switch').nth(switchIndex);
  await expect(toggle).toBeVisible({ timeout: 15000 });

  const current = await getToggleState(toggle);
  if (current === expectedEnabled) {
    return;
  }

  await toggle.click({ force: true });
  await clickConfirmIfVisible(page, expectedEnabled);

  await waitForCheck(`switch state should become ${expectedEnabled ? 'enabled' : 'disabled'}`, async () => {
    // 确保页面 DOM 加载完毕
    await page.waitForLoadState('domcontentloaded');

    // 尝试点击策略文本以展开配置面板
    try {
      const policyElement = page.getByText(policyText).first();
      if ((await policyElement.count()) > 0 && (await policyElement.isVisible({ timeout: 3000 }))) {
        await policyElement.click();
      }
    } catch {
      // 如果点击失败则重新加载页面
      await page.reload({ waitUntil: 'domcontentloaded' });
    }

    const currentToggle = page.getByRole('switch').nth(switchIndex);
    const enabled = await getToggleState(currentToggle);
    if (enabled !== expectedEnabled) {
      throw new Error(`switch ${policyText} is still ${enabled ? 'enabled' : 'disabled'}`);
    }
  }, 120000, 3000);

  await page.waitForLoadState('domcontentloaded');
}

async function clickConfirmIfVisible(page: Page, expectedEnabled: boolean): Promise<void> {
  // 某些策略关闭时不弹确认框；开启时通常会弹框，给更长等待时间。
  const timeout = expectedEnabled ? 8000 : 1500;
  const confirmButton = page.getByRole('button', { name: /确\s*认/ }).first();

  let visible = false;
  try {
    visible = await confirmButton.isVisible({ timeout });
  } catch {
    visible = false;
  }

  if (!visible) {
    return;
  }

  await confirmButton.click();
}

export async function readKubeletConfig(ssh: NodeSSH): Promise<string> {
  return execMustSucceed(ssh, `cat ${shellArg(TEST_KUBELET_CONFIG_PATH)}`);
}

export async function execMustSucceed(ssh: NodeSSH, command: string): Promise<string> {
  const result = await ssh.execCommand(command);
  if ((result.code ?? 1) !== 0) {
    const stderr = result.stderr?.trim() ?? '';
    const stdout = result.stdout?.trim() ?? '';
    throw new Error(`command failed: ${command}\nstdout: ${stdout}\nstderr: ${stderr}`);
  }
  return result.stdout?.trim() ?? '';
}

export async function waitForCheck(
  name: string,
  checker: () => Promise<void>,
  timeoutMs = 120000,
  intervalMs = 5000,
): Promise<void> {
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
}

export async function createAnyWorkloadAndVerifyCreated(ssh: NodeSSH, caseKey: string): Promise<void> {
  const namespace = `${caseKey}-${Date.now().toString().slice(-6)}`;
  const deploymentName = 'any-pod-demo';

  try {
    await execMustSucceed(
      ssh,
      `kubectl create namespace ${shellArg(namespace)} --dry-run=client -o yaml | kubectl apply -f -`,
    );

    const yaml = `apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${deploymentName}
  namespace: ${namespace}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ${deploymentName}
  template:
    metadata:
      labels:
        app: ${deploymentName}
    spec:
      containers:
      - name: pause
        image: alpine:latest
        imagePullPolicy: IfNotPresent
        command: ["sleep", "360000"]`;

    await execMustSucceed(ssh, `cat <<'EOF' | kubectl apply -f -\n${yaml}\nEOF`);

    const deployment = await execMustSucceed(
      ssh,
      `kubectl get deployment ${shellArg(deploymentName)} -n ${shellArg(namespace)} -o jsonpath='{.metadata.name}'`,
    );

    if (deployment.trim() !== deploymentName) {
      throw new Error(`deployment not created as expected: ${deployment}`);
    }
  } finally {
    await ssh.execCommand(`kubectl delete namespace ${shellArg(namespace)} --ignore-not-found=true`);
  }
}

function shellArg(raw: string): string {
  return `'${raw.replace(/'/g, `"'"'`)}'`;
}

async function getToggleState(toggle: Locator): Promise<boolean> {
  const ariaChecked = await toggle.getAttribute('aria-checked');
  if (ariaChecked === 'true') {
    return true;
  }
  if (ariaChecked === 'false') {
    return false;
  }

  const checkbox = toggle.locator('input[type="checkbox"]').first();
  if ((await checkbox.count()) > 0) {
    return checkbox.isChecked();
  }

  const tagName = await toggle.evaluate((node) => node.tagName.toLowerCase());
  if (tagName === 'input') {
    const typeAttr = await toggle.getAttribute('type');
    if (typeAttr === 'checkbox') {
      return toggle.isChecked();
    }
  }

  throw new Error('cannot determine toggle checked state by aria-checked/checkbox');
}

export async function installNumaComponent(ssh: NodeSSH): Promise<void> {
  await ssh.execCommand(`helm uninstall ${shellArg(NUMA_RELEASE_NAME)} -n ${shellArg(NUMA_RELEASE_NAMESPACE)} --ignore-not-found`);

  const installCmd = [
    `helm install ${shellArg(NUMA_RELEASE_NAME)} ${shellArg(NUMA_RELEASE_REPO)}`,
    `--version ${shellArg(NUMA_RELEASE_VERSION)}`,
    `-n ${shellArg(NUMA_RELEASE_NAMESPACE)}`,
    '--create-namespace',
  ].join(' ');

  await execMustSucceed(ssh, installCmd);
}

export async function waitForNumaComponentReady(
  ssh: NodeSSH,
  options?: { timeoutMs?: number; failOnTimeout?: boolean },
): Promise<void> {
  const timeoutMs = options?.timeoutMs ?? 90000;
  const failOnTimeout = options?.failOnTimeout ?? false;

  try {
    await waitForCheck('numa helm release deployed', async () => {
      const output = await execMustSucceed(
        ssh,
        `helm status ${shellArg(NUMA_RELEASE_NAME)} -n ${shellArg(NUMA_RELEASE_NAMESPACE)} --output json`,
      );

      let parsed: { info?: { status?: string } };
      try {
        parsed = JSON.parse(output) as { info?: { status?: string } };
      } catch {
        throw new Error(`helm status output is not valid json: ${output}`);
      }

      const status = parsed.info?.status?.toLowerCase() ?? '';
      if (status !== 'deployed') {
        throw new Error(`helm release status is ${status || 'unknown'}`);
      }
    }, timeoutMs, 5000);
  } catch (err) {
    if (failOnTimeout) {
      throw err;
    }
    const detail = err instanceof Error ? err.message : String(err);
    console.warn(`[numa-affinity] setup ready check skipped: ${detail}`);
  }

  await new Promise((resolve) => setTimeout(resolve, 10000));
}

export async function uninstallNumaComponent(ssh: NodeSSH): Promise<void> {
  await ssh.execCommand(`helm uninstall ${shellArg(NUMA_RELEASE_NAME)} -n ${shellArg(NUMA_RELEASE_NAMESPACE)} --ignore-not-found`);
}

async function openNumaPageWithRetry(
  page: Page,
  relativePath: string,
  readyCheck: () => Promise<void>,
): Promise<void> {
  await waitForCheck(
    `open NUMA page ${relativePath}`,
    async () => {
      await page.goto(`${FUYAO_BASE_URL}${relativePath}`, { waitUntil: 'domcontentloaded' });
      await page.waitForLoadState('networkidle', { timeout: 20000 });
      await readyCheck();
    },
    120000,
    4000,
  );
}
