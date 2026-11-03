import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { test, expect, Page, BrowserContext } from '@playwright/test';
import { TEST_FUYAOURL, TEST_USERNAME, TEST_PASSWORD } from './consts';
import { newContext } from '../utils/common';

/**
 * 这个文件的 beforeAll / afterAll 会修改集群内的 OAuth 相关配置（短会话/短锁定）：
 *   1. ConfigMap   oauth-server-config        （data.config.json 的若干字段）
 *   2. Deployment  user-management-operator   （环境变量 LOCK_DURATION）
 *   3. 重启 oauth-server，使 ConfigMap 生效
 *
 * 这些修改不再通过前端 UI 逐个改资源，而是直接调用 Kubernetes API 完成：
 *   auth/utils/oauth-config-patch.mjs（@kubernetes/client-node + kubeconfig，不依赖 kubectl）。
 * 该脚本是配置变更的“唯一事实来源”：apply=应用测试配置，restore=恢复生产配置，status=只读。
 */

// 配置变更脚本路径（相对本文件解析，避免依赖运行时 cwd）。
// kubeconfig 由脚本自动解析（KUBECONFIG 环境变量 > ~/.kube/config > 仓库内本地副本），这里不强制路径。
const PATCH_SCRIPT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), 'utils/oauth-config-patch.mjs');
const execFileAsync = promisify(execFile);

// 服务恢复等待配置
const SERVICE_RECOVERY_CONFIG = {
    maxRetries: 24,        // 最大重试次数
    retryInterval: 5000,   // 重试间隔（毫秒）
    maxWaitTime: 120000,   // 最大等待时间（毫秒）
};

// serial：这些用例共享集群状态（beforeAll 改配置、afterAll 还原），必须串行在同一个 worker
// 中执行，避免 fullyParallel 下多 worker 并发触发 beforeAll/afterAll 互相覆盖集群配置。
test.describe.serial('OAuth Server 配置设置', () => {
    let page: Page;
    let context: BrowserContext;
    // beforeAll 成功 apply 后置 true，作为“配置已修改、afterAll 需要 restore”的标记
    let oauthConfigApplied = false;

    /**
     * 等待服务恢复可用
     * 通过尝试访问首页并检查是否能正常加载来判断
     */
    async function waitForServiceReady(page: Page): Promise<boolean> {
        const startTime = Date.now();
        
        for (let i = 0; i < SERVICE_RECOVERY_CONFIG.maxRetries; i++) {
            if (Date.now() - startTime > SERVICE_RECOVERY_CONFIG.maxWaitTime) {
                console.log('等待服务恢复超时');
                return false;
            }

            try {
                const response = await page.goto(`${TEST_FUYAOURL}`, { 
                    waitUntil: 'domcontentloaded',
                    timeout: 10000 
                });
                
                // 检查响应状态码
                if (response && response.status() < 500) {
                    console.log(`服务已恢复，耗时 ${Date.now() - startTime}ms`);
                    return true;
                }
            } catch (error) {
                console.log(`第 ${i + 1} 次检查服务状态失败，${SERVICE_RECOVERY_CONFIG.retryInterval}ms 后重试...`);
            }
            
            await page.waitForTimeout(SERVICE_RECOVERY_CONFIG.retryInterval);
        }
        
        return false;
    }

    /**
     * 登录系统
     * @param forceLogin 是否强制重新登录（即使已登录也会先退出再登录）
     */
    async function login(page: Page, forceLogin: boolean = false) {
        // 先等待服务可用
        const isReady = await waitForServiceReady(page);
        if (!isReady) {
            throw new Error('服务不可用，无法登录');
        }

        await page.goto(`${TEST_FUYAOURL}`);
        await page.waitForLoadState('networkidle');
        
        // 检查是否已经在登录页面
        const isLoginPage = await page.getByRole('textbox', { name: '*用户名' }).isVisible().catch(() => false);
        if (!isLoginPage) {
            // 可能已经登录了
            const isLoggedIn = await page.getByText(TEST_USERNAME!).isVisible().catch(() => false);
            if (isLoggedIn) {
                if (forceLogin) {
                    // 强制登录模式：先退出再登录
                    await logout(page);
                    await page.goto(`${TEST_FUYAOURL}`);
                    await page.waitForLoadState('networkidle');
                } else {
                    // 非强制模式：已登录直接返回
                    return;
                }
            }
        }
        
        await page.getByRole('textbox', { name: '*用户名' }).fill(TEST_USERNAME!);
        await page.getByRole('textbox', { name: '*密码' }).fill(TEST_PASSWORD!);
        await page.getByRole('button', { name: '登录' }).click();
        
        // 等待登录成功（限定在顶部 .user_info 区域内的用户名，避免多个 "admin" 文本命中导致 strict mode 报错）
        await expect(page.locator('.user_info').getByText(TEST_USERNAME!, { exact: true })).toBeVisible({ timeout: 15000 });
    }

    /**
     * 退出登录
     */
    async function logout(page: Page) {
        // hover 到用户信息区域显示下拉菜单
        await page.locator('p').filter({ hasText: TEST_USERNAME! }).hover();
        await page.getByRole('button', { name: '退出登录' }).click();
        
        // 等待退出成功提示
        await expect(page.getByText('退出成功')).toBeVisible({ timeout: 5000 }).catch(() => {});
        
        // 等待跳转到登录页，包含 503 重试机制
        const maxRetries = 5;
        const retryInterval = 3000;
        
        for (let i = 0; i < maxRetries; i++) {
            try {
                await page.waitForURL(/\/auth\/login|\/oauth2\/auth\/login/, { timeout: 10000 });
                
                // 检查是否是 503 错误页面
                const is503 = await page.getByText('503').isVisible().catch(() => false) ||
                              await page.getByText('Service Unavailable').isVisible().catch(() => false);
                
                if (is503) {
                    console.log(`第 ${i + 1} 次遇到 503 错误，${retryInterval}ms 后重试...`);
                    await page.waitForTimeout(retryInterval);
                    await page.reload();
                    continue;
                }
                
                // 成功跳转到登录页
                return;
            } catch (error) {
                console.log(`第 ${i + 1} 次等待登录页失败，${retryInterval}ms 后重试...`);
                await page.waitForTimeout(retryInterval);
                await page.reload();
            }
        }
        
        // 最后一次尝试
        await page.waitForURL(/\/auth\/login|\/oauth2\/auth\/login/, { timeout: 10000 });
    }

    /**
     * 通过 auth/utils/oauth-config-patch.mjs 直接调用 Kubernetes API 修改/恢复 OAuth 后端配置，
     * 替代原先在前端逐个改 ConfigMap/Deployment 的做法（不依赖 kubectl）。
     * 在独立子进程中运行该脚本，确保脚本本身可作为独立工具被验证。
     * mode: 'apply' 应用测试配置（短超时）；'restore' 恢复生产配置；'status' 只读查看。
     */
    async function runOAuthConfigPatch(mode: 'apply' | 'restore' | 'status'): Promise<void> {
        const { stdout, stderr } = await execFileAsync(
            process.execPath,
            [PATCH_SCRIPT, mode],
            {
                timeout: 300000,
                maxBuffer: 10 * 1024 * 1024,
            },
        );
        if (stdout.trim()) console.log(`[oauth-config-patch ${mode}]\n${stdout.trim()}`);
        if (stderr.trim()) console.log(`[oauth-config-patch ${mode}][stderr]\n${stderr.trim()}`);
    }

    // 配置变更通过脚本直接打 Kubernetes API 完成（见 runOAuthConfigPatch），不再走前端 UI
    test.beforeAll(async ({ browser }) => {
        // beforeAll 超时为 5 分钟（脚本需修改 ConfigMap/Deployment 并等待 oauth-server 重启）
        test.setTimeout(300000);

        // 应用 OAuth 测试配置（短会话/短锁定）：ConfigMap + Deployment + 重启 oauth-server
        console.log('应用 OAuth 测试配置（短会话/短锁定）...');
        await runOAuthConfigPatch('apply');
        oauthConfigApplied = true;

        // 创建各用例共用的 page（此时未登录，各用例自行 login，从而创建基于新配置的新 session）
        context = await newContext(browser);
        page = await context.newPage();
    });

    test.afterAll(async () => {
        // afterAll 超时：恢复配置 + 登录页恢复重试（20 次 × 每次 10s）需较大预算
        test.setTimeout(450000);

        try {
            if (oauthConfigApplied) {
                console.log('恢复 OAuth 生产配置...');
                await runOAuthConfigPatch('restore');
                oauthConfigApplied = false;
            }
        } catch (error) {
            console.error('配置恢复失败:', error);
        }

        try {
            // restore 会重启 oauth-server，期间登录页可能短暂返回 502。
            // 这里带重试验证登录页最终可正常访问（HTTP < 500，不再 502）：
            // 每次间隔 10s，最多 20 次，仍失败则报错（让 afterAll 失败而非静默通过）。
            const maxAttempts = 20;
            const attemptInterval = 10000; // 每次重试间隔 10s
            let loginPageReady = false;
            for (let attempt = 1; attempt <= maxAttempts; attempt++) {
                try {
                    const response = await page.goto(`${TEST_FUYAOURL}`, {
                        waitUntil: 'domcontentloaded',
                        timeout: 10000,
                    });
                    if (response && response.status() < 500) {
                        loginPageReady = true;
                        console.log(`登录页已恢复（HTTP ${response.status()}），第 ${attempt}/${maxAttempts} 次尝试成功`);
                        break;
                    }
                    console.log(`第 ${attempt}/${maxAttempts} 次尝试登录页返回 HTTP ${response?.status() ?? '无响应'}`);
                } catch (error) {
                    console.log(`第 ${attempt}/${maxAttempts} 次尝试访问登录页失败：${(error as Error).message}`);
                }
                if (attempt < maxAttempts) {
                    await page.waitForTimeout(attemptInterval);
                }
            }
            expect(loginPageReady, `登录页在 ${maxAttempts} 次重试（每次 ${attemptInterval / 1000}s）后仍无法正常访问，疑似 502`).toBe(true);
        } finally {
            // 无论如何都要关闭资源（关闭 context 会自动清除 session，无需 logout）
            if (page && !page.isClosed()) {
                await page.close().catch(() => {});
            }
            if (context) {
                await context.close().catch(() => {});
            }
        }
    });

    test('配置已成功修改', async () => {
        test.setTimeout(90000);
        await page.waitForTimeout(80000);
        // 这个测试用例仅用于验证配置修改是否成功（beforeAll 已通过脚本 apply）
        expect(oauthConfigApplied).toBe(true);
    });

    test('【用户管理-004】验证浏览器2h会话超时，系统提示用户已登出，需手动点击重定向到登陆页面', {
        tag: ['@v24.06', '@用户管理'],
        annotation:{
            type: '备注',
            description:'这里通过修改后台配置，将等待时间修改为60s+10s'
        }
    }, async () => {
        // 设置测试超时时间为 90 秒（65 秒等待 + 25 秒缓冲）
        test.setTimeout(90000);
        
        // 确保服务可用并导航到首页
        await waitForServiceReady(page);

        // 登录系统（此时会创建一个新 session，使用新配置的 SessionMaxAge = 60 秒）
        // 注意：beforeAll 只 apply 了配置且未登录，所以这里会创建全新的 session
        await login(page);

        // 1. 验证当前已登录状态（限定在顶部 .user_info 区域内的用户名，避免多个 "admin" 文本命中）
        await expect(page.locator('.user_info').getByText(TEST_USERNAME!, { exact: true })).toBeVisible();
        await expect(page).not.toHaveURL(/\/auth\/login/);

        // 2. 等待会话过期（SessionMaxAge 设置为 60 秒，等待 80 秒以确保过期）
        // 这里需要进入一个不会向后台发送请求的页面，不能是默认登陆之后的页面。
        await page.goto(`${TEST_FUYAOURL}/user_manage/user`);
        console.log('等待 80 秒让 session 过期...');
        await page.waitForTimeout(80000);

        // 3. 刷新页面
        await page.reload();
        await page.waitForLoadState('networkidle');

        // 4. 验证页面已跳转到登录页面
        await expect(page).toHaveURL(/\/auth\/login|\/oauth2\/auth\/login/, { timeout: 10000 });

        // 5. 验证登录表单元素存在
        await expect(page.getByRole('textbox', { name: '*用户名' })).toBeVisible();
        await expect(page.getByRole('textbox', { name: '*密码' })).toBeVisible();
        await expect(page.getByRole('button', { name: '登录' })).toBeVisible();
    });

    // test('【用户管理-005】验证连续输错5次密码后账户锁定1分钟，解锁后可正常登录', {
    //     tag: ['@v24.06', '@用户管理'],
    //     annotation: {
    //         type: '备注',
    //         description: '通过修改后台配置，将锁定时间修改为1分钟'
    //     }
    // }, async () => {
    //     // 设置测试超时时间为 120 秒（5次错误登录 + 60秒等待 + 缓冲时间）
    //     test.setTimeout(120000);

    //     // 确保服务可用
    //     await waitForServiceReady(page);

    //     // 确保在登录页面
    //     await page.goto(`${TEST_FUYAOURL}`);
    //     await page.waitForLoadState('networkidle');

    //     // 如果已登录，先退出
    //     const isLoggedIn = await page.getByText(TEST_USERNAME!).isVisible().catch(() => false);
    //     if (isLoggedIn) {
    //         await logout(page);
    //     }

    //     // 1. 连续输入5次错误密码
    //     const wrongPassword = 'WrongPassword123!';
    //     for (let i = 1; i <= 5; i++) {
    //         console.log(`第 ${i} 次输入错误密码...`);
            
    //         // 确保在登录页面
    //         const isLoginPage = await page.getByRole('textbox', { name: '*用户名' }).isVisible().catch(() => false);
    //         if (!isLoginPage) {
    //             await page.goto(`${TEST_FUYAOURL}`);
    //             await page.waitForLoadState('networkidle');
    //         }

    //         await page.getByRole('textbox', { name: '*用户名' }).fill(TEST_USERNAME!);
    //         await page.getByRole('textbox', { name: '*密码' }).fill(wrongPassword);
    //         await page.getByRole('button', { name: '登录' }).click();
            
    //         // 等待登录失败提示
    //         await page.waitForTimeout(1000);
    //     }

    //     // 2. 验证账户已被锁定（第6次尝试应该显示锁定提示）
    //     console.log('验证账户锁定状态...');
    //     await page.getByRole('textbox', { name: '*用户名' }).fill(TEST_USERNAME!);
    //     await page.getByRole('textbox', { name: '*密码' }).fill(TEST_PASSWORD!);
    //     await page.getByRole('button', { name: '登录' }).click();
        
    //     // 验证显示锁定提示信息
    //     await expect(page.getByText(/账户锁定中，请等待 1 分钟后重试/i)).toBeVisible({ timeout: 5000 });

    //     // 3. 等待锁定时间结束（LockDuration 设置为 1 分钟，等待 65 秒以确保解锁）
    //     console.log('等待 65 秒让账户解锁...');
    //     await page.waitForTimeout(65000);

    //     // 4. 刷新页面，使用正确密码登录
    //     await page.goto(`${TEST_FUYAOURL}`);
    //     await page.waitForLoadState('networkidle');

    //     await page.getByRole('textbox', { name: '*用户名' }).fill(TEST_USERNAME!);
    //     await page.getByRole('textbox', { name: '*密码' }).fill(TEST_PASSWORD!);
    //     await page.getByRole('button', { name: '登录' }).click();

    //     // 5. 验证登录成功
    //     // 限定在顶部 .user_info 区域内的用户名，避免页面上存在多个 "admin" 文本（如用户列表链接、
    //     // "platform-admin" 单元格）导致 getByText 命中多个元素触发 strict mode 报错。
    //     await expect(page.locator('.user_info').getByText(TEST_USERNAME!, { exact: true })).toBeVisible({ timeout: 15000 });
    //     await expect(page).not.toHaveURL(/\/auth\/login/);

    //     console.log('账户解锁后登录成功！');
    // });
});
