import { test, expect } from "@playwright/test";
import { urls } from "./constants/common";
import { logout, firstLoginChangePassword } from "./utils/rbac";


test.describe('首次登录修改密码', {
    tag: ['@prerequisite', '@online'],
}, () => {

    test.describe('验证首次登录与修改密码', () => {
        test('验证首次登录引导节点修改密码', async ({ page }) => {
            await page.goto(process.env.TEST_GUIDEURL + urls.loginUrl);

            await test.step("验证使用正确的用户名和密码登陆成功", async () => {
                await page.getByRole('textbox', { name: '*用户名' }).fill(process.env.TEST_USERNAME);
                await page.getByRole('textbox', { name: '*密码' }).fill(process.env.TEST_INITIAL_PASSWORD);
                await page.getByRole('button', { name: '登录' }).click();
            });

            await test.step("验证首次登录必须修改密码", async () => {
                await expect(page.getByRole('heading', { name: '初次修改密码' })).toBeVisible();
            });

            await test.step("初次登录修改密码成功", async () => {
                await firstLoginChangePassword(page);
                await expect(page.getByText(process.env.TEST_USERNAME)).toBeVisible();
            });

            await test.step("验证注销登录成功", async () => {
                await logout(page);
                await expect(page.getByRole('heading', { name: '欢迎登录openFuyao' })).toBeVisible();
            });
        });
    });
});
