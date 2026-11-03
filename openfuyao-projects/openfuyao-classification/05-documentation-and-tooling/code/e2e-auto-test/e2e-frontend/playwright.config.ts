import { defineConfig, devices } from '@playwright/test';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

// 加载环境变量
const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({
    path: path.resolve(__dirname, './.env.test'),
    quiet: true,
    override: true,
});
const authJsonPath = path.resolve(__dirname, './auth.json');
if (!fs.existsSync(authJsonPath)) {
    fs.writeFileSync(authJsonPath, '{}');
}

const chromeBrowserZH: any = {
    ...devices['Desktop Chrome'],
    locale: 'zh-CN',
    timezoneId: 'Asia/Shanghai',
};

export default defineConfig({
    testDir: './',
    testMatch: '**/*.spec.ts',
    testIgnore: ['playwright-report/**', 'test-results/**', 'kubevirt/**'],
    timeout: 2 * 60 * 1000,
    expect: {
        timeout: 1 * 60 * 1000,
    },
    fullyParallel: true,
    forbidOnly: !!process.env.CI,
    retries: process.env.CI ? 2 : 0,
    workers: process.env.CI ? 1 : undefined,
    reporter: [process.env.CI ? ['allure-playwright'] : ['html']],
    use: {
        ignoreHTTPSErrors: true,
        trace: 'retain-on-failure',
        actionTimeout: 30 * 1000,
        navigationTimeout: 60 * 1000,
        storageState: authJsonPath,
        locale: 'zh-CN',
        timezoneId: 'Asia/Shanghai',
    },
    projects: [
        {
            name: 'global-setup',
            testMatch: 'global.setup.ts',
            teardown: 'global-teardown',
            use: { ...chromeBrowserZH },
        },
        {
            name: 'global-teardown',
            testMatch: 'global.teardown.ts',
            use: { ...chromeBrowserZH },
        },
        // 认证鉴权测试
        {
            name: 'auth',
            testMatch: 'auth/**/*.spec.ts',
            use: { ...chromeBrowserZH },
        },
        // 安装部署测试
        {
            name: 'installation',
            testMatch: 'installation/**/*.spec.ts',
            use: { ...chromeBrowserZH },
            timeout: 120 * 60 * 1000, // 安装部署耗时长
        },
        // 应用市场测试
        {
            name: 'marketplace',
            testMatch: 'marketplace/**/*.spec.ts',
            use: { ...chromeBrowserZH },
            dependencies: ['global-setup'],
        },
        // NUMA 亲和调度测试
        {
            name: 'numa-affinity-setup',
            testMatch: 'numa-affinity/numa.setup.ts',
            teardown: 'numa-affinity-teardown',
            use: { ...chromeBrowserZH },
        },
        {
            name: 'numa-affinity-teardown',
            testMatch: 'numa-affinity/numa.teardown.ts',
            use: { ...chromeBrowserZH },
        },
        {
            name: 'numa-affinity',
            testMatch: 'numa-affinity/**/*.spec.ts',
            use: { ...chromeBrowserZH },
            dependencies: ['global-setup', 'numa-affinity-setup'],
        },
        // 管理面k8s资源测试
        {
            name: 'console',
            testMatch: 'console/**/*.spec.ts',
            use: { ...chromeBrowserZH },
            dependencies: ['global-setup'],
        },
        // 混部（Colocation）页面测试
        {
            name: 'colocation-setup',
            testMatch: 'colocation/colocation.setup.ts',
            teardown: 'colocation-teardown',
            use: { ...chromeBrowserZH },
        },
        {
            name: 'colocation-teardown',
            testMatch: 'colocation/colocation.teardown.ts',
            use: { ...chromeBrowserZH },
        },
        {
            name: 'colocation',
            testMatch: 'colocation/**/*.spec.ts',
            use: { ...chromeBrowserZH },
            dependencies: ['global-setup', 'colocation-setup'],
        },
        // web-terminal-service 测试
        {
            name: 'web-terminal-service',
            testMatch: 'web-terminal-service/**/*.spec.ts',
            use: { ...chromeBrowserZH },
            dependencies: ['global-setup'],
        },
        // 应用生命周期管理测试
        {
            name: 'application-setup',
            testMatch: 'application/application.setup.ts',
            teardown: 'application-teardown',
            use: { ...chromeBrowserZH },
            dependencies: ['global-setup'],
        },
        {
            name: 'application-teardown',
            testMatch: 'application/application.teardown.ts',
            use: { ...chromeBrowserZH },
        },
        {
            name: 'application',
            testMatch: 'application/**/*.spec.ts',
            use: { ...chromeBrowserZH },
            dependencies: ['global-setup', 'application-setup'],
        },
    ],
});
