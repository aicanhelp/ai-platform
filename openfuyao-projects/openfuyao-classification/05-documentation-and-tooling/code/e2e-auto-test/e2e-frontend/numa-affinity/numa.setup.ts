import { test } from '@playwright/test';
import { NodeSSH } from 'node-ssh';
import { TEST_SSH_ADDRESS, TEST_SSH_PASSWORD, TEST_SSH_USERNAME } from './constants';
import { installNumaComponent, waitForNumaComponentReady } from './utils';

test('numa-affinity setup: install component', async () => {
  test.setTimeout(3 * 60 * 1000);

  if (!TEST_SSH_ADDRESS || !TEST_SSH_PASSWORD) {
    throw new Error('TEST_NODE2_IP or TEST_NODE2_PASSWORD is empty');
  }

  const ssh = new NodeSSH();
  await ssh.connect({
    host: TEST_SSH_ADDRESS,
    username: TEST_SSH_USERNAME,
    password: TEST_SSH_PASSWORD,
    readyTimeout: 30000,
  });

  try {
    await installNumaComponent(ssh);
    await waitForNumaComponentReady(ssh, { timeoutMs: 90000, failOnTimeout: false });
  } finally {
    ssh.dispose();
  }
});
