import { test, expect } from '@playwright/test';
import { TEST_FUYAOURL, TEST_USERNAME, TEST_PASSWORD, TEST_INITIAL_PASSWORD } from './consts';

test.describe('Login', {
    tag: ['@prerequisite'],
}, () => {
    test.beforeEach(async ({ page }) => {
        await page.goto(TEST_FUYAOURL!);
        await page.waitForLoadState('networkidle');
    });
    
    test.afterEach(async ({ page }) => {
        await page.locator('p').filter({ hasText: TEST_USERNAME! }).hover();
        await page.getByRole('button', { name: '退出登录' }).click();
        await expect(page.getByText('退出成功')).toBeVisible({ timeout: 5000 }).catch(() => {});
    });

    test('【用户管理-009】验证部署完环境后收集登录必须修改密码', {
        tag: ['@v24.06', '@用户管理'],
    }, async ({ page }) => {
        await page.getByRole('textbox', { name: '*用户名' }).fill(TEST_USERNAME!);
        await page.getByRole('textbox', { name: '*密码' }).fill(TEST_INITIAL_PASSWORD!);
        await page.getByRole('button', { name: '登录' }).click();
        await expect(page.getByRole('heading', { name: '初次修改密码' })).toBeVisible();

        await page.getByRole('textbox', { name: '*新密码' }).fill(TEST_PASSWORD!);
        await page.locator('#confirm-password').fill(TEST_PASSWORD!);
        await page.getByRole('button', { name: /确\s*定|确\s*认/ }).click();
        await expect(page.getByText(TEST_USERNAME!)).toBeVisible({ timeout: 10000 });
    });

    test('【用户管理-010】验证使用正确的用户名和密码登陆成功', {
        tag: ['@v24.06', '@用户管理'],
    }, async ({ page }) => {
        await page.getByRole('textbox', { name: '*用户名' }).fill(TEST_USERNAME!);
        await page.getByRole('textbox', { name: '*密码' }).fill(TEST_PASSWORD!);
        await page.getByRole('button', { name: '登录' }).click();
        await expect(page.getByText(TEST_USERNAME!)).toBeVisible({ timeout: 10000 });
    });

    test('318【用户管理-026】验证登录和退出登录的响应时间不超过1s'+
        '【用户管理-051】验证注销登陆的响应时间不超过1s'
        , {
        tag: ['@v24.06', '@用户管理']
    }, async ({ page }) => {
        // 记录登录响应时间（从点击登录按钮到OAuth2回调完成）
        let loginResponseTime = 0;
        let loginClickTime = 0;
        let loginCallbackReceived = false;

        // 记录退出登录响应时间（从点击退出按钮到退出回调完成）
        let logoutResponseTime = 0;
        let logoutClickTime = 0;
        let logoutCallbackReceived = false;

        // 监听关键API响应
        page.on('response', async response => {
            const url = response.url();

            // 检测OAuth2回调完成
            if (url.includes('/rest/auth/callback') && !loginCallbackReceived) {
                const responseEndTime = Date.now();
                loginResponseTime = responseEndTime - loginClickTime;
                loginCallbackReceived = true;
                console.log(`[OAuth2回调] 登录回调完成，响应时间: ${loginResponseTime}ms`);
            }

            // 检测退出登录回调完成
            if (url.includes('/rest/auth/logout') && !logoutCallbackReceived) {
                const responseEndTime = Date.now();
                logoutResponseTime = responseEndTime - logoutClickTime;
                logoutCallbackReceived = true;
                console.log(`[退出登录] 退出回调完成，响应时间: ${logoutResponseTime}ms`);
            }
        });

        console.log(`\n=== 开始登录和退出登录性能测试 ===`);
        console.log(`测试地址: ${TEST_FUYAOURL}`);
        console.log(`测试用户: ${TEST_USERNAME}`);
        console.log(`开始时间: ${new Date().toISOString()}`);

        // ==================== 第一步：登录性能测试 ====================
        console.log(`\n========== 第一步：登录性能测试 ==========`);

        // 访问登录页面
        await page.goto(TEST_FUYAOURL!);
        await page.waitForLoadState('networkidle');

        try {
            // 等待用户名输入框出现
            await page.waitForSelector('input[name="username"], [role="textbox"]', { timeout: 5000 });

            // 填写用户名和密码
            await page.getByRole('textbox', { name: '*用户名' }).fill(TEST_USERNAME!);
            await page.getByRole('textbox', { name: '*密码' }).fill(TEST_PASSWORD!);

            // 点击登录按钮 - 开始计时
            console.log(`[点击登录按钮] 开始计时...`);
            loginClickTime = Date.now();
            await page.getByRole('button', { name: '登录' }).click();

            // 等待登录成功 - 用户名出现表示登录成功
            await expect(page.getByText(TEST_USERNAME!)).toBeVisible({ timeout: 10000 });
            console.log(`[登录成功] 用户已登录`);

            // 等待一小段时间确保回调被捕获
            await page.waitForTimeout(500);

            // 如果没有捕获到回调，使用从点击到登录成功的总时间
            if (!loginCallbackReceived) {
                console.warn(`[警告] 未捕获到OAuth2回调，使用登录成功时间`);
                loginResponseTime = Date.now() - loginClickTime;
            }

        } catch (error) {
            console.error(`[登录失败] ${error instanceof Error ? error.message : String(error)}`);
            throw error;
        }

        // ==================== 第二步：退出登录性能测试 ====================
        console.log(`\n========== 第二步：退出登录性能测试 ==========`);

        // 等待一下确保页面完全加载
        await page.waitForTimeout(1000);

        try {
            // 点击用户名显示菜单
            await page.locator('p').filter({ hasText: TEST_USERNAME! }).hover();

            // 点击退出登录按钮 - 开始计时
            console.log(`[点击退出登录按钮] 开始计时...`);
            logoutClickTime = Date.now();
            await page.getByRole('button', { name: '退出登录' }).click();

            // 等待退出成功提示出现
            await expect(page.getByText('退出成功')).toBeVisible({ timeout: 10000 });
            console.log(`[退出成功] 收到退出成功提示`);

            // 等待一小段时间确保回调被捕获
            await page.waitForTimeout(500);

            // 如果没有捕获到回调，使用从点击到看到提示的总时间
            if (!logoutCallbackReceived) {
                console.warn(`[警告] 未捕获到退出登录回调，使用退出成功提示时间`);
                logoutResponseTime = Date.now() - logoutClickTime;
            }

        } catch (error) {
            console.error(`[退出失败] ${error instanceof Error ? error.message : String(error)}`);
            throw error;
        }

        // ==================== 性能测试报告 ====================
        console.log(`\n========== 性能测试报告 ==========`);
        console.log(`登录响应时间: ${loginResponseTime}ms`);
        console.log(`退出登录响应时间: ${logoutResponseTime}ms`);
        console.log(`===================================\n`);

        // 性能断言 - 验证登录响应时间不超过1秒
        expect(loginResponseTime).toBeLessThan(1000);
        console.log(`✓ 登录时间符合预期 (< 1秒)`);

        // 性能断言 - 验证退出登录响应时间不超过1秒
        expect(logoutResponseTime).toBeLessThan(1000);
        console.log(`✓ 退出登录时间符合预期 (< 1秒)`);
    });
});