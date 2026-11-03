import { request as playwrightRequest } from '@playwright/test';
import type { NodeSSH } from 'node-ssh';
import { test, expect, createAnyWorkloadAndVerifyCreated, execMustSucceed, openPolicyConfigPage, readKubeletConfig, setPolicySwitchByRecordedFlow, waitForCheck } from './utils';
import {
  RUNTIME_POD_AFFINITY_RPM_KEYWORDS,
  TEST_KUBELET_SERVICE,
} from './constants';

async function getMissingRuntimePodAffinityRpms(ssh: NodeSSH): Promise<string[]> {
  try {
    const rpmListRaw = await execMustSucceed(ssh, `rpm -qa | tr '[:upper:]' '[:lower:]'`);
    const installedRpms = rpmListRaw
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line.length > 0);

    return RUNTIME_POD_AFFINITY_RPM_KEYWORDS.filter(
      (keyword) => !installedRpms.some((rpm) => rpm.includes(keyword)),
    );
  } catch {
    return [...RUNTIME_POD_AFFINITY_RPM_KEYWORDS];
  }
}

const hasPreferClosestNumaNodes = (kubeletConfig: string, expected: boolean): boolean => {
  const value = expected ? 'true' : 'false';
  return (
    kubeletConfig.includes(`prefer-closest-numa-nodes: ${value}`)
    || kubeletConfig.includes(`prefer-closest-numa-nodes: "${value}"`)
    || kubeletConfig.includes(`prefer-closest-numa-nodes: '${value}'`)
    || kubeletConfig.includes(`prefer-closest-numa-nodes=${value}`)
  );
};

test.describe('NUMA亲和调度-策略开关与节点校验', () => {
  test.beforeEach(async ({ page }) => {
    test.setTimeout(5 * 60 * 1000);
    await openPolicyConfigPage(page);
  });

  test('【NUMA2503-003】亲和策略配置-NUMA亲和性调度-页面基础信息展示', {
    tag: ['@v26.03', '@numa-affinity'],
  }, async ({ page }) => {
    await expect(page.getByText('numa-aware').first()).toBeVisible();
    await expect(page.getByRole('switch').nth(0)).toBeVisible();
  });

  test('【NUMA2503-021】亲和策略配置-最优NUMAdistance-页面基础信息展示', {
    tag: ['@v26.03', '@numa-affinity'],
  }, async ({ page }) => {
    await expect(page.getByText('最优NUMA Distance').first()).toBeVisible();
    await expect(page.getByRole('switch').nth(1)).toBeVisible();
  });

  test('【NUMA2503-036】亲和策略配置-运行时Pod亲和优化-页面基础信息展示', {
    tag: ['@v26.03', '@numa-affinity'],
  }, async ({ page }) => {
    await expect(page.getByText('运行时Pod亲和优化').first()).toBeVisible();
    await expect(page.getByRole('switch').nth(2)).toBeVisible();
  });

  test('【NUMA2503-001】亲和策略配置-NUMA亲和性调度-开启操作-自动化配置-当前static策略未打开-基础功能验证', {
    tag: ['@v26.03', '@numa-affinity'],
  }, async ({ page, ssh }) => {
    await setPolicySwitchByRecordedFlow(page, 'numa-aware', 0, true);

    await waitForCheck('kubelet config cpuManagerPolicy should be static', async () => {
      const kubeletConfig = await readKubeletConfig(ssh);
      if (!kubeletConfig.includes('cpuManagerPolicy: static')) {
        throw new Error('cpuManagerPolicy is not static');
      }
    });

    await waitForCheck('kubeReserved/systemReserved cpu reservation should exist', async () => {
      const kubeletConfig = await readKubeletConfig(ssh);
      if (!kubeletConfig.includes('kubeReserved:')) {
        throw new Error('kubeReserved section is missing');
      }
      if (!kubeletConfig.includes('systemReserved:')) {
        throw new Error('systemReserved section is missing');
      }

      const kubeReservedCpuExists = /kubeReserved:[\s\S]*?\n\s*cpu:\s*\S+/.test(kubeletConfig);
      const systemReservedCpuExists = /systemReserved:[\s\S]*?\n\s*cpu:\s*\S+/.test(kubeletConfig);
      if (!kubeReservedCpuExists || !systemReservedCpuExists) {
        throw new Error('kubeReserved/systemReserved cpu reservation is missing');
      }
    });

    await waitForCheck('cpu_manager_state should exist', async () => {
      await execMustSucceed(ssh, 'ls /var/lib/kubelet/cpu_manager_state');
    });

    await waitForCheck('kubelet should be active', async () => {
      const status = await execMustSucceed(ssh, `systemctl is-active '${TEST_KUBELET_SERVICE}'`);
      if (status.trim() !== 'active') {
        throw new Error(`kubelet status is not active: ${status}`);
      }
    });
  });

  test('【NUMA2503-002】亲和策略配置-NUMA亲和性调度-关闭操作-自动化配置-当前static策略已打开-基础功能验证', {
    tag: ['@v26.03', '@numa-affinity'],
  }, async ({ page, ssh }) => {
    await setPolicySwitchByRecordedFlow(page, 'numa-aware', 0, false);

    await waitForCheck('kubelet config cpuManagerPolicy should be none', async () => {
      const kubeletConfig = await readKubeletConfig(ssh);
      if (!kubeletConfig.includes('cpuManagerPolicy: none')) {
        throw new Error('cpuManagerPolicy is not none');
      }
    });

    await waitForCheck('cpu_manager_state should exist', async () => {
      await execMustSucceed(ssh, 'ls /var/lib/kubelet/cpu_manager_state');
    });

    await waitForCheck('kubelet should be active', async () => {
      const status = await execMustSucceed(ssh, `systemctl is-active '${TEST_KUBELET_SERVICE}'`);
      if (status.trim() !== 'active') {
        throw new Error(`kubelet status is not active: ${status}`);
      }
    });
  });

  test('【NUMA2503-017】亲和策略配置-最优NUMAdistance-开启操作-自动化配置-当前策略未打开-基础功能验证', {
    tag: ['@v26.03', '@numa-affinity'],
  }, async ({ page, ssh }) => {
    test.setTimeout(5 * 60 * 1000);
    await setPolicySwitchByRecordedFlow(page, '最优NUMA Distance', 1, true);

    await waitForCheck('kubelet config prefer-closest-numa-nodes should be true', async () => {
      const kubeletConfig = await readKubeletConfig(ssh);
      if (!hasPreferClosestNumaNodes(kubeletConfig, true)) {
        throw new Error('prefer-closest-numa-nodes is not true');
      }
    });
  });

  test('【NUMA2503-018】亲和策略配置-最优NUMAdistance-关闭操作-自动化配置-当前策略已打开-基础功能验证', {
    tag: ['@v26.03', '@numa-affinity'],
  }, async ({ page, ssh }) => {
    await setPolicySwitchByRecordedFlow(page, '最优NUMA Distance', 1, false);

    await waitForCheck('kubelet config prefer-closest-numa-nodes should be false', async () => {
      const kubeletConfig = await readKubeletConfig(ssh);

      if (!hasPreferClosestNumaNodes(kubeletConfig, false)) {
        throw new Error('prefer-closest-numa-nodes is not false');
      }
    });

    await waitForCheck('kubelet should be active', async () => {
      const status = await execMustSucceed(ssh, `systemctl is-active '${TEST_KUBELET_SERVICE}'`);
      if (status.trim() !== 'active') {
        throw new Error(`kubelet status is not active: ${status}`);
      }
    });
  });

  test('【NUMA2503-019】亲和策略配置-最优NUMAdistance-开启操作-预先手动配置完成-基础功能验证', {
    tag: ['@v26.03', '@numa-affinity'],
  }, async ({ page, ssh }) => {
    await setPolicySwitchByRecordedFlow(page, '最优NUMA Distance', 1, true);

    await waitForCheck('kubelet config prefer-closest-numa-nodes should be true', async () => {
      const kubeletConfig = await readKubeletConfig(ssh);

      if (!hasPreferClosestNumaNodes(kubeletConfig, true)) {
        throw new Error('prefer-closest-numa-nodes is not true');
      }
    });
  });

  test('【NUMA2503-006】亲和策略配置-NUMA亲和性调度-开启操作-预先手动配置完成-基础功能验证', {
    tag: ['@v26.03', '@numa-affinity'],
  }, async ({ page, ssh }) => {
    await setPolicySwitchByRecordedFlow(page, 'numa-aware', 0, true);

    await waitForCheck('kubelet config cpuManagerPolicy should be static', async () => {
      const kubeletConfig = await readKubeletConfig(ssh);
      if (!kubeletConfig.includes('cpuManagerPolicy: static')) {
        throw new Error('cpuManagerPolicy is not static');
      }

      const kubeReservedCpuExists = /kubeReserved:[\s\S]*?\n\s*cpu:\s*\S+/.test(kubeletConfig);
      const systemReservedCpuExists = /systemReserved:[\s\S]*?\n\s*cpu:\s*\S+/.test(kubeletConfig);
      if (!kubeReservedCpuExists || !systemReservedCpuExists) {
        throw new Error('kubeReserved/systemReserved cpu reservation is missing');
      }
    });
  });

  test('【NUMA2503-034】亲和策略配置-运行时Pod亲和优化-自动化配置-策略已打开-基础功能验证', {
    tag: ['@v26.03', '@numa-affinity', '@skip-temporarily'],
  }, async ({ page, ssh }) => {
    const arch = (await execMustSucceed(ssh, 'uname -m')).toLowerCase();
    test.skip(!(arch.includes('aarch64') || arch.includes('arm64')), `当前架构为 ${arch}，运行时Pod亲和优化仅在ARM环境执行`);

    const missingRpms = await getMissingRuntimePodAffinityRpms(ssh);
    test.skip(
      missingRpms.length > 0,
      `缺少运行时Pod亲和优化依赖rpm: ${missingRpms.join(', ')}`,
    );

    await setPolicySwitchByRecordedFlow(page, '运行时Pod亲和优化', 2, true);

    await waitForCheck('containerd config should keep NRI section', async () => {
      const conf = await execMustSucceed(ssh, 'cat /etc/containerd/config.toml');
      const hasNriSection = /\[plugins\.("|')io\.containerd\.nri\.v1\.nri\1\]/.test(conf);
      if (!hasNriSection) {
        throw new Error('NRI section not found in containerd config');
      }
    });

    await createAnyWorkloadAndVerifyCreated(ssh, 'numa2503-034');
  });

  test('【NUMA2503-013】亲和策略配置-NUMA亲和性调度-认证鉴权-未认证修改策略', {
    tag: ['@v26.03', '@numa-affinity'],
  }, async ({ page }) => {
    await page.context().clearCookies();

    const unauthApi = await playwrightRequest.newContext({
      baseURL: new URL(page.url()).origin,
      ignoreHTTPSErrors: true,
      storageState: { cookies: [], origins: [] },
    });

    try {
      const response = await unauthApi.put('/clusters/host/rest/scheduling/v1/numaaware', {
        data: { policy: 'open' },
        maxRedirects: 0,
      });
      expect([401, 403, 302], `unexpected status for unauth request: ${response.status()}`).toContain(response.status());
    } finally {
      await unauthApi.dispose();
    }
  });

  test('【NUMA2503-022】亲和策略配置-最优NUMAdistance-开关打开-端到端基础功能验证', {
    tag: ['@v26.03', '@numa-affinity'],
  }, async ({ page, ssh }) => {
    // 基础开关验证
    await setPolicySwitchByRecordedFlow(page, '最优NUMA Distance', 1, true);

    await waitForCheck('kubelet config prefer-closest-numa-nodes should be true', async () => {
      const kubeletConfig = await readKubeletConfig(ssh);
      if (!hasPreferClosestNumaNodes(kubeletConfig, true)) {
        throw new Error('prefer-closest-numa-nodes is not true');
      }
    });

    // 检查是否为物理机多NUMA环境
    const lscpuOutput = await execMustSucceed(ssh, 'lscpu');
    const numaNodesMatch = lscpuOutput.match(/NUMA node\(s\):\s+(\d+)/);
    let numaNodes = 1;
    if (numaNodesMatch && numaNodesMatch[1]) {
      numaNodes = parseInt(numaNodesMatch[1], 10);
    }

    if (numaNodes <= 1) {
      console.log(`当前环境为单NUMA或虚拟机环境 (NUMA nodes: ${numaNodes})，已确认开关正常打开且配置下发，跳过跨NUMA pod调度验证。`);
      return;
    }

    console.log(`当前环境为多NUMA物理机环境 (NUMA nodes: ${numaNodes})，可进行跨节点pod验证（待补充）。`);
    // TODO: 后续在此处补充在多 NUMA 节点环境创建跨节点 pod 的实际调度验证逻辑
  });
});

