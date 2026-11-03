import { test, expect, Page } from '@playwright/test';
import { TEST_FUYAOURL, TEST_USERNAME, TEST_PASSWORD } from './consts';
import { newContext } from '../utils/common';

test.describe('未登录测试', () => {
    let page: Page;

    test.beforeAll(async ({ browser }) => {
        // 创建新的浏览器上下文和页面，确保无登录状态
        const context = await newContext(browser);
        page = await context.newPage();
    });

    test.afterAll(async () => {
        // 关闭页面
        await page.close();
    });

    test.afterEach(async()=>{
        await page.goto(`${TEST_FUYAOURL}`);
      })

    test('【用户管理-003】未登录时，输入任何路由都会跳转到登陆页面', {
        tag: ['@v24.06', '@用户管理']
    }, async () => {
        // 定义需要测试的页面路径列表
        const testPaths = [
            '/container_platform/overview',           // 总览页面
            '/container_platform/nodeManage',         // 节点管理页面
            '/container_platform/clusterUser',        // 集群角色页面
            '/container_platform/clusterMember',      // 集群成员页面
            '/user_manage/user/usermanage',           // 用户管理页面
            '/setting/userinfo',                      // 用户设置页面
            '/home',                                  // 首页
        ];

        for (const path of testPaths) {
            // 访问测试路径
            await page.goto(`${TEST_FUYAOURL}${path}`);
            
            // 等待页面加载完成
            await page.waitForLoadState('networkidle');
            
            // 验证页面被重定向到登录页面
            // 检查URL是否包含登录相关路径
            await expect(page).toHaveURL(/\/auth\/login/);
            
            // 验证登录表单元素存在
            await expect(page.getByRole('textbox', { name: '*用户名' })).toBeVisible();
            await expect(page.getByRole('textbox', { name: '*密码' })).toBeVisible();
            await expect(page.getByRole('button', { name: '登录' })).toBeVisible();
        }
    });

    test('【用户管理-018】对登录时5分钟内连续输入1、2、3、4次错误的密码时验证能提示用户剩余4、3、2、1次'+
        '【用户管理-020】在同一台机器上验证分别5分钟内连续输入错误密码4次，第5次输入正确', {
        tag: ['@v24.06', '@用户管理']
    }, async () => {
        // 1. 打开登录页面
        await page.goto(`${TEST_FUYAOURL}`);
        await page.waitForLoadState('networkidle');

        const wrongPassword = 'wrongpassword';

        // 2. 连续四次输入错误密码，验证剩余次数提示
        for (let i = 1; i <= 4; i++) {
            const remainingAttempts = 5 - i; // 剩余次数：4, 3, 2, 1
            
            await page.getByRole('textbox', { name: '*用户名' }).fill(TEST_USERNAME!);
            await page.getByRole('textbox', { name: '*密码' }).fill(wrongPassword);
            await page.getByRole('button', { name: '登录' }).click();
            
            // 验证错误提示中的剩余次数
            await expect(page.getByText(`用户名或密码错误，再输错 ${remainingAttempts} 次用户将锁定`)).toBeVisible({ timeout: 5000 });
        }

        // 3. 第五次输入正确密码
        await page.getByRole('textbox', { name: '*用户名' }).fill(TEST_USERNAME!);
        await page.getByRole('textbox', { name: '*密码' }).fill(TEST_PASSWORD!);
        await page.getByRole('button', { name: '登录' }).click();

        // 4. 验证登录成功 - 等待页面上出现用户名
        await expect(page.getByText(TEST_USERNAME!)).toBeVisible({ timeout: 10000 });

        // 5. 验证页面跳转到非登录页面
        await expect(page).not.toHaveURL(/\/auth\/login/);

        // 6. 退出登录，恢复测试环境
        await page.locator('p').filter({ hasText: TEST_USERNAME! }).hover();
        await page.getByText('退出登录').click();
        await expect(page).toHaveURL(/\/auth\/login|\/oauth2\/auth\/login/, { timeout: 10000 });
    });

    test('【用户管理-046】验证注销登录成功'+
        '【用户管理-047】同一浏览器多个页签都注销成功', {
        tag: ['@v24.06', '@用户管理']
    }, async ({ browser }) => {
        // 创建新的浏览器上下文（共享 Cookie/Session）
        const context = await newContext(browser);
        
        // 创建第一个标签页并登录
        const page1 = await context.newPage();
        await page1.goto(`${TEST_FUYAOURL}`);
        await page1.waitForLoadState('networkidle');
        
        // 登录操作
        await page1.getByRole('textbox', { name: '*用户名' }).fill(TEST_USERNAME!);
        await page1.getByRole('textbox', { name: '*密码' }).fill(TEST_PASSWORD!);
        await page1.getByRole('button', { name: '登录' }).click();

        // 验证登录成功 - 使用locator定位顶部的用户名段落元素
        await expect(page1.locator('p').filter({ hasText: TEST_USERNAME! })).toBeVisible({ timeout: 10000 });

        // 创建第二个标签页，访问容器平台总览页面
        const page2 = await context.newPage();
        await page2.goto(`${TEST_FUYAOURL}/container_platform/overview`);
        await page2.waitForLoadState('networkidle');

        // 验证第二个标签页自动登录成功（显示用户名）
        await expect(page2.locator('p').filter({ hasText: TEST_USERNAME! })).toBeVisible({ timeout: 10000 });

        // 创建第三个标签页，访问用户管理页面
        const page3 = await context.newPage();
        await page3.goto(`${TEST_FUYAOURL}/user_manage/user`);
        await page3.waitForLoadState('networkidle');

        // 验证第三个标签页自动登录成功（显示用户名）
        await expect(page3.locator('p').filter({ hasText: TEST_USERNAME! })).toBeVisible({ timeout: 10000 });
        
        // 在第一个标签页退出登录
        await page1.locator('p').filter({ hasText: TEST_USERNAME! }).hover();
        await page1.getByText('退出登录').click();
        
        // 验证第一个标签页跳转到登录页
        await expect(page1).toHaveURL(/\/auth\/login|\/oauth2\/auth\/login/, { timeout: 10000 });
        
        // 在第二个标签页点击链接访问新页面
        await page2.getByRole('menuitem', { name: '节点' }).click();
        
        // 验证弹出登录超时提示
        await expect(page2.getByText('您尚未登录或您的登录会话已超时，请重新登录')).toBeVisible({ timeout: 10000 });
        
        // 点击确定按钮
        await page2.getByRole('button', { name: '确定' }).click();
        
        // 验证第二个标签页跳转到登录页
        await expect(page2).toHaveURL(/\/auth\/login|\/oauth2\/auth\/login/, { timeout: 10000 });
        
        // 在第三个标签页点击链接访问新页面
        await page3.getByRole('menuitem', { name: '角色管理' }).click();
        
        // 验证弹出登录超时提示
        await expect(page3.getByText('您尚未登录或您的登录会话已超时，请重新登录')).toBeVisible({ timeout: 10000 });
        
        // 点击确定按钮
        await page3.getByRole('button', { name: '确定' }).click();
        
        // 验证第三个标签页跳转到登录页
        await expect(page3).toHaveURL(/\/auth\/login|\/oauth2\/auth\/login/, { timeout: 10000 });
        
        // 验证登录表单元素存在
        await expect(page3.getByRole('textbox', { name: '*用户名' })).toBeVisible();
        await expect(page3.getByRole('textbox', { name: '*密码' })).toBeVisible();
        await expect(page3.getByRole('button', { name: '登录' })).toBeVisible();
        
        // 关闭所有页面和上下文
        await page1.close();
        await page2.close();
        await page3.close();
        await context.close();
    });

    test('317【用户管理-027】验证显示密码功能正常', {
        tag: ['@v24.09', '@多集群多用户分权分域认证鉴权']
    }, async () => {
        await test.step('进入登录页面', async () => {
            await page.goto(`${TEST_FUYAOURL}`);
            await page.waitForLoadState('networkidle');
            // 验证登录表单元素存在
            await expect(page.getByRole('textbox', { name: '*用户名' })).toBeVisible();
            await expect(page.getByRole('textbox', { name: '*密码' })).toBeVisible();
            await expect(page.getByRole('button', { name: '登录' })).toBeVisible();
        });

        await test.step('用户名输入成功', async () => {
            await page.getByRole('textbox', { name: '*用户名' }).fill(TEST_USERNAME!);
            // 验证用户名已输入
            await expect(page.getByRole('textbox', { name: '*用户名' })).toHaveValue(TEST_USERNAME!);
        });

        await test.step('密码输入成功', async () => {
            await page.getByRole('textbox', { name: '*密码' }).fill(TEST_PASSWORD!);
            // 验证密码已输入（密码框默认应该隐藏，type="password"）
            const passwordInput = page.locator('input[type="password"]');
            await expect(passwordInput).toHaveValue(TEST_PASSWORD!);
        });

        await test.step('能正确显示密码', async () => {
            // 获取密码输入框
            const passwordInput = page.getByRole('textbox', { name: '*密码' });

            // 验证初始状态密码框是隐藏的（type="password"）
            await expect(passwordInput).toHaveAttribute('type', 'password');

            // 点击密码框旁边的眼睛图标显示密码
            // 使用 getByRole 定位第一个 img 元素（眼睛图标）
            const eyeIcon = page.getByRole('img').first();
            await eyeIcon.click();

            // 验证密码输入框的 type 属性变为 text（密码可见）
            await expect(passwordInput).toHaveAttribute('type', 'text');
        });
    });
});
