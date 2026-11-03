import path from 'path';
import { fileURLToPath } from 'url';
import { test, expect } from '@playwright/test';
import { newAuthorizedContext, login } from './utils/common';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

test('setup: login and save auth state', async ({ browser }) => {
    const context = await newAuthorizedContext(browser);
    const page = await context.newPage();

    await page.goto('/home');
    await page.waitForLoadState('networkidle');
    if (page.url().includes('/oauth2/auth/login/fuyaoPasswordProvider')) {
        // not logged in, login and save auth.json...
        await login(page, process.env.TEST_USERNAME!, process.env.TEST_PASSWORD!);
        await page.waitForLoadState('networkidle');
        await expect(page.getByText(process.env.TEST_USERNAME!)).toBeVisible();
        await context.storageState({ path: path.resolve(__dirname, './auth.json') });
    }
    await context.close();
});