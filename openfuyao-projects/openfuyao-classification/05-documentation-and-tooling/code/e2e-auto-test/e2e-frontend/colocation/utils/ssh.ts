import { NodeSSH } from 'node-ssh';

export const TEST_SSH_ADDRESS = process.env.TEST_NODE2_IP ?? '';
export const TEST_SSH_USERNAME = process.env.TEST_NODE2_USER ?? 'root';
export const TEST_SSH_PASSWORD = process.env.TEST_NODE2_PASSWORD ?? '';

export const isSshConfigured = (): boolean => {
  return TEST_SSH_ADDRESS.length > 0 && TEST_SSH_PASSWORD.length > 0;
};

export const withSsh = async <T>(fn: (ssh: NodeSSH) => Promise<T>): Promise<T> => {
  const ssh = new NodeSSH();
  await ssh.connect({
    host: TEST_SSH_ADDRESS,
    username: TEST_SSH_USERNAME,
    password: TEST_SSH_PASSWORD,
    readyTimeout: 30000,
  });

  try {
    return await fn(ssh);
  } finally {
    ssh.dispose();
  }
};

export const execMustSucceed = async (ssh: NodeSSH, command: string): Promise<string> => {
  const result = await ssh.execCommand(command);
  if ((result.code ?? 1) !== 0) {
    const stdout = result.stdout?.trim() ?? '';
    const stderr = result.stderr?.trim() ?? '';
    throw new Error(`command failed: ${command}\nstdout: ${stdout}\nstderr: ${stderr}`);
  }
  return result.stdout?.trim() ?? '';
};

export const fetchNodeViaSsh = async (ssh: NodeSSH, nodeName: string): Promise<Record<string, unknown>> => {
  const output = await execMustSucceed(
    ssh,
    `kubectl get node '${nodeName.replace(/'/g, `"'"'`)}' -o json`,
  );
  return JSON.parse(output) as Record<string, unknown>;
};

export const fetchColocationOptionsViaSsh = async (ssh: NodeSSH): Promise<Record<string, unknown>> => {
  const output = await execMustSucceed(
    ssh,
    'kubectl get configmap colocation-config -n openfuyao-colocation -o json',
  );
  const configMap = JSON.parse(output) as { data?: Record<string, string> };
  const options = configMap.data?.['colocation-options'] ?? '{}';
  return JSON.parse(options) as Record<string, unknown>;
};
