import type { NodeSSH } from 'node-ssh';

const sleep = async (ms: number): Promise<void> => {
  await new Promise((resolve) => setTimeout(resolve, ms));
};

const readStatus = (value: unknown): string => {
  return typeof value === 'string' ? value.toLowerCase() : '';
};

const shellArg = (raw: string): string => {
  return `'${raw.replace(/'/g, `"'"'`)}'`;
};

const hasResourceKey = (obj: Record<string, unknown> | undefined, key: string): boolean => {
  return typeof obj?.[key] === 'string' && String(obj[key]).trim().length > 0;
};

const hasFeatureLabel = (labels: Record<string, unknown> | undefined, key: string): boolean => {
  return String(labels?.[key] ?? '').toLowerCase() === 'true';
};

const getEnabledColocationFeatureLabels = (labels: Record<string, unknown> | undefined): string[] => {
  if (!labels) {
    return [];
  }

  return Object.entries(labels)
    .filter(([key, value]) => key.includes('/colocation.feature.') && String(value).toLowerCase() === 'true')
    .map(([key]) => key);
};

export const waitForHelmReleaseDeployed = async (
  ssh: NodeSSH,
  releaseName: string,
  namespace: string,
  timeoutMs = 180000,
  intervalMs = 5000,
): Promise<void> => {
  const deadline = Date.now() + timeoutMs;
  let lastStatus = 'unknown';

  while (Date.now() < deadline) {
    const result = await ssh.execCommand(
      `helm status ${shellArg(releaseName)} -n ${shellArg(namespace)} --output json`,
    );

    if ((result.code ?? 1) === 0) {
      try {
        const parsed = JSON.parse(result.stdout || '{}') as { info?: { status?: string } };
        const status = (parsed.info?.status || '').toLowerCase();
        lastStatus = status || 'unknown';
        if (status === 'deployed') {
          return;
        }
      } catch {
        lastStatus = 'invalid-json';
      }
    }

    await sleep(intervalMs);
  }

  throw new Error(
    `helm release ${releaseName} in namespace ${namespace} is not deployed within timeout; last status: ${lastStatus}`,
  );
};

export const waitForNamespacePodsOrServicesReady = async (
  ssh: NodeSSH,
  namespace: string,
  timeoutMs = 240000,
  intervalMs = 5000,
): Promise<void> => {
  const deadline = Date.now() + timeoutMs;
  let lastReason = 'resources not observed yet';

  while (Date.now() < deadline) {
    const result = await ssh.execCommand(
      `kubectl get pods,svc -n ${shellArg(namespace)} -o json`,
    );

    if ((result.code ?? 1) === 0) {
      try {
        const parsed = JSON.parse(result.stdout || '{}') as {
          items?: Array<{
            kind?: string;
            metadata?: { name?: string };
            status?: {
              phase?: string;
              conditions?: Array<{ type?: string; status?: string }>;
            };
          }>;
        };

        const items = parsed.items ?? [];
        const pods = items.filter((item) => readStatus(item.kind) === 'pod');
        const services = items.filter((item) => readStatus(item.kind) === 'service');

        if (pods.length === 0 && services.length === 0) {
          lastReason = `namespace ${namespace} has no pods or services yet`;
          await sleep(intervalMs);
          continue;
        }

        if (pods.length === 0) {
          return;
        }

        const notReadyPods = pods.filter((pod) => {
          const phase = readStatus(pod.status?.phase);
          if (phase === 'succeeded') {
            return false;
          }

          if (phase !== 'running') {
            return true;
          }

          const conditions = pod.status?.conditions ?? [];
          const readyCondition = conditions.find((condition) => condition.type === 'Ready');
          return (readyCondition?.status || '').toLowerCase() !== 'true';
        });

        if (notReadyPods.length === 0) {
          return;
        }

        lastReason = `not ready pods: ${notReadyPods
          .map((pod) => `${pod.metadata?.name ?? '<unknown>'}(${pod.status?.phase ?? 'unknown'})`)
          .join(', ')}`;
      } catch {
        lastReason = 'failed to parse kubectl json output';
      }
    } else {
      const stderr = result.stderr?.trim();
      lastReason = stderr ? `kubectl failed: ${stderr}` : 'kubectl failed without stderr';
    }

    await sleep(intervalMs);
  }

  throw new Error(
    `namespace ${namespace} pods/services are not ready within timeout; last state: ${lastReason}`,
  );
};

export const waitForAnyNodeColocationCapability = async (
  ssh: NodeSSH,
  timeoutMs = 300000,
  intervalMs = 5000,
): Promise<void> => {
  const deadline = Date.now() + timeoutMs;
  let lastReason = 'nodes are not observed yet';

  while (Date.now() < deadline) {
    const result = await ssh.execCommand('kubectl get nodes -o json');

    if ((result.code ?? 1) === 0) {
      try {
        const parsed = JSON.parse(result.stdout || '{}') as {
          items?: Array<{
            metadata?: {
              name?: string;
              labels?: Record<string, unknown>;
            };
            status?: {
              capacity?: Record<string, unknown>;
              allocatable?: Record<string, unknown>;
              conditions?: Array<{ type?: string; status?: string }>;
            };
          }>;
        };

        const items = parsed.items ?? [];
        if (items.length === 0) {
          lastReason = 'cluster has no nodes yet';
          await sleep(intervalMs);
          continue;
        }

        const diagnostics: string[] = [];

        for (const node of items) {
          const nodeName = node.metadata?.name ?? '<unknown>';
          const labels = node.metadata?.labels;
          const capacity = node.status?.capacity;
          const allocatable = node.status?.allocatable;
          const conditions = node.status?.conditions ?? [];
          const readyCondition = conditions.find((condition) => condition.type === 'Ready');
          const isReady = (readyCondition?.status ?? '').toLowerCase() === 'true';

          const hasBeCpu = hasResourceKey(capacity, 'kubernetes.io/be-cpu')
            && hasResourceKey(allocatable, 'kubernetes.io/be-cpu');
          const hasBeMem = hasResourceKey(capacity, 'kubernetes.io/be-memory')
            && hasResourceKey(allocatable, 'kubernetes.io/be-memory');

          const hasCpuPreemption = hasFeatureLabel(labels, 'openfuyao.com/colocation.feature.cpupreemption');
          const hasMemoryPreemption = hasFeatureLabel(labels, 'openfuyao.com/colocation.feature.memorypreemption');
          const hasQuotaTurbo = hasFeatureLabel(labels, 'openfuyao.com/colocation.feature.quotaturbo');
          const enabledFeatureLabels = getEnabledColocationFeatureLabels(labels);
          const hasFeatureSignals = enabledFeatureLabels.length > 0;

          const capabilityReady = (hasBeCpu && hasBeMem) || hasFeatureSignals;

          diagnostics.push(
            `${nodeName}(ready=${isReady},be-cpu=${hasBeCpu},be-memory=${hasBeMem},feature-cpu=${hasCpuPreemption},feature-memory=${hasMemoryPreemption},feature-quotaturbo=${hasQuotaTurbo},feature-any=${hasFeatureSignals},feature-count=${enabledFeatureLabels.length})`,
          );

          if (isReady && capabilityReady) {
            return;
          }
        }

        lastReason = diagnostics.join(', ');
      } catch {
        lastReason = 'failed to parse node list json';
      }
    } else {
      const stderr = result.stderr?.trim();
      lastReason = stderr ? `kubectl failed: ${stderr}` : 'kubectl failed without stderr';
    }

    await sleep(intervalMs);
  }

  throw new Error(`no ready node has colocation capability signals within timeout; last state: ${lastReason}`);
};
