import path from 'path';
import { fileURLToPath } from 'url';
import { test } from '@playwright/test';
import { logout, newAuthorizedContext } from './utils/common';
import { writeFile } from 'fs/promises';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

test('teardown: logout and clear auth state', async ({ browser }) => {
    const context = await newAuthorizedContext(browser);
    const page = await context.newPage();
    await page.goto('/home');
    await page.waitForLoadState('networkidle');
    if (await page.getByText(process.env.TEST_USERNAME!).isVisible()) {
        await logout(page);
    }
    await context.close();
    await writeFile(path.resolve(__dirname, './auth.json'), '{}');
});