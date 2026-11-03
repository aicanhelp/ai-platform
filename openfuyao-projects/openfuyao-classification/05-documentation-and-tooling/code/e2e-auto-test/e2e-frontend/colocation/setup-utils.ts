import { NodeSSH } from 'node-ssh';
import { execMustSucceed } from './utils/ssh';
import {
  waitForAnyNodeColocationCapability,
  waitForHelmReleaseDeployed,
  waitForNamespacePodsOrServicesReady,
} from './utils/helm';

const VOLCANO_RELEASE_NAME = 'volcano';
const VOLCANO_NAMESPACE = 'volcano-system';
const VOLCANO_CHART = 'volcano-sh/volcano';
const VOLCANO_VERSION = '1.9.0';

const COLOCATION_RELEASE_NAME = 'colocation-package';
const COLOCATION_NAMESPACE = 'default';
const COLOCATION_NAMESPACE_FALLBACK = 'cluster-system';
const COLOCATION_CHART = 'oci://cr.openfuyao.cn/charts/colocation-package';
const COLOCATION_VERSION = '0.0.0-latest';

const shellArg = (raw: string): string => {
  return `'${raw.replace(/'/g, `"'"'`)}'`;
};

export const installColocationDependencies = async (ssh: NodeSSH): Promise<void> => {
  await execMustSucceed(
    ssh,
    'helm repo list | grep "^volcano-sh" >/dev/null 2>&1 || helm repo add volcano-sh https://volcano-sh.github.io/helm-charts',
  );
  await execMustSucceed(ssh, 'helm repo update');

  await ssh.execCommand(
    `helm uninstall ${shellArg(COLOCATION_RELEASE_NAME)} -n ${shellArg(COLOCATION_NAMESPACE)} --ignore-not-found`,
  );
  await ssh.execCommand(
    `helm uninstall ${shellArg(COLOCATION_RELEASE_NAME)} -n ${shellArg(COLOCATION_NAMESPACE_FALLBACK)} --ignore-not-found`,
  );
  await ssh.execCommand(
    `helm uninstall ${shellArg(VOLCANO_RELEASE_NAME)} -n ${shellArg(VOLCANO_NAMESPACE)} --ignore-not-found`,
  );

  await execMustSucceed(
    ssh,
    `helm install ${shellArg(VOLCANO_RELEASE_NAME)} ${shellArg(VOLCANO_CHART)} --version ${shellArg(VOLCANO_VERSION)} -n ${shellArg(VOLCANO_NAMESPACE)} --create-namespace`,
  );
  await waitForHelmReleaseDeployed(ssh, VOLCANO_RELEASE_NAME, VOLCANO_NAMESPACE, 240000, 5000);

  let installedNamespace = COLOCATION_NAMESPACE;
  const installResult = await ssh.execCommand(
    `helm install ${shellArg(COLOCATION_RELEASE_NAME)} ${shellArg(COLOCATION_CHART)} --version ${shellArg(COLOCATION_VERSION)} -n ${shellArg(COLOCATION_NAMESPACE)} --create-namespace`,
  );

  if ((installResult.code ?? 1) !== 0) {
    const stderr = installResult.stderr || '';
    const shouldFallback = stderr.includes('meta.helm.sh/release-namespace')
      && stderr.includes(COLOCATION_NAMESPACE_FALLBACK);

    if (!shouldFallback) {
      const stdout = installResult.stdout?.trim() ?? '';
      throw new Error(`command failed: helm install colocation-package\nstdout: ${stdout}\nstderr: ${stderr.trim()}`);
    }

    await execMustSucceed(
      ssh,
      `helm install ${shellArg(COLOCATION_RELEASE_NAME)} ${shellArg(COLOCATION_CHART)} --version ${shellArg(COLOCATION_VERSION)} -n ${shellArg(COLOCATION_NAMESPACE_FALLBACK)} --create-namespace`,
    );
    installedNamespace = COLOCATION_NAMESPACE_FALLBACK;
  }

  await waitForHelmReleaseDeployed(ssh, COLOCATION_RELEASE_NAME, installedNamespace, 240000, 5000);
  await waitForNamespacePodsOrServicesReady(ssh, VOLCANO_NAMESPACE, 240000, 5000);
  await waitForNamespacePodsOrServicesReady(ssh, 'openfuyao-colocation', 240000, 5000);

  // 节点能力检测是异步收敛过程，setup阶段先尽量等待，避免进入用例时全量节点开关仍为 disabled。
  try {
    await waitForAnyNodeColocationCapability(ssh, 300000, 5000);
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    console.warn(`[colocation-setup] node capability not ready yet: ${detail}`);
  }
};

export const uninstallColocationDependencies = async (ssh: NodeSSH): Promise<void> => {
  await ssh.execCommand(
    `helm uninstall ${shellArg(COLOCATION_RELEASE_NAME)} -n ${shellArg(COLOCATION_NAMESPACE)} --ignore-not-found`,
  );
  await ssh.execCommand(
    `helm uninstall ${shellArg(COLOCATION_RELEASE_NAME)} -n ${shellArg(COLOCATION_NAMESPACE_FALLBACK)} --ignore-not-found`,
  );
  await ssh.execCommand(
    `helm uninstall ${shellArg(VOLCANO_RELEASE_NAME)} -n ${shellArg(VOLCANO_NAMESPACE)} --ignore-not-found`,
  );
};
