import { test } from '@playwright/test';
import { isSshConfigured, withSsh } from './utils/ssh';
import { uninstallColocationDependencies } from './setup-utils';

test('colocation teardown: uninstall volcano and colocation package', async () => {
  test.setTimeout(6 * 60 * 1000);

  if (!isSshConfigured()) {
    return;
  }

  await withSsh(async (ssh) => {
    await uninstallColocationDependencies(ssh);
  });
});