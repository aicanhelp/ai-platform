import type { Browser, BrowserContext, BrowserContextOptions, Page } from "@playwright/test";
import path from "path";
import { fileURLToPath } from 'url';

const DEFAULT_LOCALE = 'zh-CN';
const DEFAULT_TIMEZONE = 'Asia/Shanghai';
const __dirname = path.dirname(fileURLToPath(import.meta.url));

export async function newAuthorizedContext(
    browser: Browser,
    options: BrowserContextOptions = {},
): Promise<BrowserContext> {
    return browser.newContext({
        locale: DEFAULT_LOCALE,
        timezoneId: DEFAULT_TIMEZONE,
        ignoreHTTPSErrors: true,
        baseURL: process.env.TEST_FUYAOURL!,
        storageState: path.resolve(__dirname, '../auth.json'),
        ...options,
    });
}

export async function newContext(
    browser: Browser,
    options: BrowserContextOptions = {},
): Promise<BrowserContext> {
    return browser.newContext({
        locale: DEFAULT_LOCALE,
        timezoneId: DEFAULT_TIMEZONE,
        ignoreHTTPSErrors: true,
        ...options,
    });
}

// 登录操作
export async function login(page: Page, username: string, password: string) {
    // 输入用户名
    await page.getByRole('textbox', { name: '*用户名' }).fill(username);
    // 输入密码
    await page.getByRole('textbox', { name: '*密码' }).fill(password);
    // 登录
    await page.getByRole('button', { name: '登录' }).click();
}

// 对应首次登录修改密码界面操作
export async function firstLoginChangePassword(page: Page, newPs: string) {
    // 输入新密码
    await page.getByRole('textbox', { name: '*新密码' }).fill(newPs);
    // 再次输入新密码
    await page.locator('#confirm-password').fill(newPs);
    // 确认密码并登录
    await page.getByRole('button', { name: /确\s*认/ }).click();
}

// 注销登录
export async function logout(page: Page) {
    // 点击用户菜单
    await page.getByRole('img', { name: 'user' }).locator('svg').click();
    // 点击退出登录
    await page.getByRole('button', { name: '退出登录' }).click();
    await page.waitForLoadState('networkidle');
}