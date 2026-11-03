import type { APIRequestContext, TestInfo } from '@playwright/test';
import { FUYAO_BASE_URL } from '@/utils/constants';

export type WorkloadKind =
  | 'pod'
  | 'deployment'
  | 'statefulset'
  | 'daemonset'
  | 'job'
  | 'cronjob';

const K8S_PROXY_PREFIX = '/clusters/host/api/kubernetes';

const workloadApiPath = (
  kind: WorkloadKind,
  namespace: string,
  name: string,
): string => {
  switch (kind) {
    case 'pod':
      return `${K8S_PROXY_PREFIX}/api/v1/namespaces/${namespace}/pods/${name}`;
    case 'deployment':
      return `${K8S_PROXY_PREFIX}/apis/apps/v1/namespaces/${namespace}/deployments/${name}`;
    case 'statefulset':
      return `${K8S_PROXY_PREFIX}/apis/apps/v1/namespaces/${namespace}/statefulsets/${name}`;
    case 'daemonset':
      return `${K8S_PROXY_PREFIX}/apis/apps/v1/namespaces/${namespace}/daemonsets/${name}`;
    case 'job':
      return `${K8S_PROXY_PREFIX}/apis/batch/v1/namespaces/${namespace}/jobs/${name}`;
    case 'cronjob':
      return `${K8S_PROXY_PREFIX}/apis/batch/v1/namespaces/${namespace}/cronjobs/${name}`;
    default: {
      const _exhaustive: never = kind;
      throw new Error(`unsupported workload kind: ${_exhaustive}`);
    }
  }
};

const publishDump = async (
  testInfo: TestInfo,
  label: string,
  body: string,
  contentType: 'application/json' | 'text/plain',
  fileName: string,
): Promise<void> => {
  console.log(`${label}\n${body}`);
  // testInfo.attach 会被 allure-playwright / Playwright HTML reporter 一并收录
  await testInfo.attach(fileName, {
    body: Buffer.from(body, 'utf-8'),
    contentType,
  });
};

/**
 * On failed/interrupted tests: GET workload via K8s API proxy, attach to the case
 * (Allure / HTML report) and also console.log for local CLI.
 * 404 and other request errors are attached/logged only — never thrown.
 */
export const dumpWorkloadOnFailure = async (
  apiContext: APIRequestContext,
  kind: WorkloadKind,
  namespace: string,
  name: string,
  testInfo: TestInfo,
): Promise<void> => {
  if (testInfo.status === 'passed' || testInfo.status === 'skipped') {
    return;
  }

  const url = `${FUYAO_BASE_URL}${workloadApiPath(kind, namespace, name)}`;
  const label = `[workload-debug] ${kind}/${namespace}/${name}`;
  const attachBase = `workload-${kind}-${namespace}-${name}`;

  try {
    const response = await apiContext.get(url);
    const body = await response.text();

    if (response.status() === 404) {
      await publishDump(
        testInfo,
        `${label}: not found (404), resource may not have been created`,
        `${label}: not found (404), resource may not have been created`,
        'text/plain',
        `${attachBase}-404.txt`,
      );
      return;
    }

    if (!response.ok()) {
      const text = `${label}: GET failed status=${response.status()} ${response.statusText()}\n${body}`;
      await publishDump(testInfo, text, text, 'text/plain', `${attachBase}-error.txt`);
      return;
    }

    try {
      const pretty = JSON.stringify(JSON.parse(body) as unknown, null, 2);
      await publishDump(
        testInfo,
        `${label}: detail`,
        pretty,
        'application/json',
        `${attachBase}.json`,
      );
    } catch {
      await publishDump(
        testInfo,
        `${label}: detail (non-JSON)`,
        body,
        'text/plain',
        `${attachBase}.txt`,
      );
    }
  } catch (err) {
    const text = `${label}: request error: ${err instanceof Error ? err.message : String(err)}`;
    await publishDump(testInfo, text, text, 'text/plain', `${attachBase}-error.txt`);
  }
};
