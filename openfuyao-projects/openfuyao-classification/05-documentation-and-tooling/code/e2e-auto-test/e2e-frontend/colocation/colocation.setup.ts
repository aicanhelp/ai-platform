import { test } from '@playwright/test';
import { isSshConfigured, withSsh } from './utils/ssh';
import { installColocationDependencies } from './setup-utils';

test('colocation setup: install volcano and colocation package', async () => {
  test.setTimeout(10 * 60 * 1000);

  if (!isSshConfigured()) {
    throw new Error('TEST_NODE2_IP or TEST_NODE2_PASSWORD is empty');
  }

  await withSsh(async (ssh) => {
    await installColocationDependencies(ssh);
  });
});