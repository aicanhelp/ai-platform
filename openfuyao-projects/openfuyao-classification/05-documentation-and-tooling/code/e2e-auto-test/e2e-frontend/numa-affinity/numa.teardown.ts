import { test } from '@playwright/test';
import { NodeSSH } from 'node-ssh';
import { TEST_SSH_ADDRESS, TEST_SSH_PASSWORD, TEST_SSH_USERNAME } from './constants';
import { uninstallNumaComponent } from './utils';

test('numa-affinity teardown: uninstall component', async () => {
  if (!TEST_SSH_ADDRESS || !TEST_SSH_PASSWORD) {
    return;
  }

  const ssh = new NodeSSH();
  await ssh.connect({
    host: TEST_SSH_ADDRESS,
    username: TEST_SSH_USERNAME,
    password: TEST_SSH_PASSWORD,
    readyTimeout: 30000,
  });

  try {
    await uninstallNumaComponent(ssh);
  } finally {
    ssh.dispose();
  }
});
