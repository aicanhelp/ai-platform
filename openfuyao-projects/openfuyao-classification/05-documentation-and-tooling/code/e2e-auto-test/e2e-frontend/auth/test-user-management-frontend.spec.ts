import { existsSync, unlinkSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { expect, test } from "@playwright/test";
import { 
    newContext,
    login,
    logout,
    firstLoginChangePassword
} from "../utils/common";

import {
    createUser,
    deleteUser,
    inviteUserWithRole,
    removeClusterUser,
} from "./utils/user-management";

const loginUrl = "/oauth2/auth/login/fuyaoPasswordProvider"
const homeUrl = "/container_platform/overview"

test.describe("用户管理纯前端", () => {
    const AUTH_FILE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "./user-management-auth.json");
    let sharepage: any;
    // 构造 业务集群URL
    const serviceClusterUrl = process.env.TEST_FUYAOURL;
    test.beforeAll(async ({ browser }) => {
        const context = await newContext(browser);
        const authpage = await context.newPage();
        // 登录 admin 账号
        await test.step("登录业务集群", async () => {
            await authpage.goto(serviceClusterUrl + loginUrl);
            await login(authpage, process.env.TEST_USERNAME!, process.env.TEST_PASSWORD!);
        });
        await expect(authpage.getByText( process.env.TEST_USERNAME!)).toBeVisible();
        // 保存认证状态
        await context.storageState({ path: AUTH_FILE });
        await authpage.close();
        await context.close();
    });
    test.afterAll(async ({ browser }) => {
            // 登出逻辑
        if (existsSync(AUTH_FILE)) {
        const context = await newContext(browser, { storageState: AUTH_FILE });
        const authpage = await context.newPage();
        await authpage.goto(serviceClusterUrl + homeUrl);
        await logout(authpage);

        await authpage.close();
        await context.close();

        // 清理认证文件
        try {
            unlinkSync(AUTH_FILE);
        } catch (error: any) {
            // 忽略文件不存在的错误
            if (error.code !== 'ENOENT') {
                throw error;
            }
        }
        }
    });
    test.beforeEach(async ( { browser } ) => {
        const context = await newContext(browser, {
            storageState: AUTH_FILE
        });
        sharepage = await context.newPage();
        // 确保在首页
        await test.step("返回首页", async () => {
            await sharepage.goto(serviceClusterUrl + homeUrl);
        });
    });

    test("157【用户管理-015】用户管理页面，点击用户名称可跳转到用户详情【用户管理-001】用户管理页面展示正常", {
            tag: ['@v24.09', '@user-management']
        }, async ( ) => {
        await test.step("进入用户管理", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await expect(sharepage.getByRole('navigation').getByText('用户管理')).toBeVisible();
        });
        await test.step("进入用户详情", async () => {
            await sharepage.getByRole('link', { name: 'admin', exact: true }).click();
            await expect(sharepage.getByText('用户管理/用户详情')).toBeVisible();
        });      
    });

    test("161【用户管理-001】用户管理页面展示正常", {
            tag: ['@v24.09', '@user-management']
        }, async ( ) => {
        await test.step("进入用户管理", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await expect(sharepage.getByRole('navigation').getByText('用户管理')).toBeVisible();
            await expect(sharepage.getByRole('menuitem', { name: '用户管理' }).getByRole('link')).toBeVisible();
            await expect(sharepage.getByRole('link', { name: '角色管理' })).toBeVisible();
        });    
    });

    test("314【用户管理-025】用户管理页面，表头排序", {
            tag: ['@v24.09', '@user-management']
        }, async ( ) => {
        let testUser1 = "test-user";
        let testUser2 = "test-user";
        const platformRole = "platform-regular"
        await test.step("进入用户管理", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await expect(sharepage.getByRole('navigation').getByText('用户管理')).toBeVisible();
        });
        await test.step("创建两个用户", async () => {
            testUser1 = await createUser(sharepage, testUser1, process.env.TEST_INITIAL_PASSWORD!, platformRole);
            testUser2 = await createUser(sharepage, testUser2, process.env.TEST_INITIAL_PASSWORD!, platformRole);
        });
        await test.step("用户名称排序", async () => {
            // 初始状态：可能未排序或默认排序
            let userNames = await sharepage.locator('table tbody tr td:first-child a').allTextContents();
            // 第一次点击：升序排序
            await sharepage.getByRole('table').getByText('用户名称').click();
            await sharepage.waitForTimeout(300);
            
            userNames = await sharepage.locator('table tbody tr td:first-child a').allTextContents();
            const ascendingNames = [...userNames].sort((a, b) => a.localeCompare(b));
            expect(userNames).toEqual(ascendingNames);

            // 第二次点击：降序排序
            await sharepage.getByRole('table').getByText('用户名称').click();
            await sharepage.waitForTimeout(300);
            
            userNames = await sharepage.locator('table tbody tr td:first-child a').allTextContents();
            const descendingNames = [...userNames].sort((a, b) => b.localeCompare(a));
            expect(userNames).toEqual(descendingNames);
        });
        await test.step("删除创建的用户", async () => {
            await deleteUser(sharepage, testUser1, platformRole);
            await deleteUser(sharepage, testUser2, platformRole);
        });
    });

    test("319【用户管理-019】修改用户信息时，用户名置灰不可修改", {
            tag: ['@v24.09', '@user-management']
        }, async ( ) => {
        let invitedUser = "test-user";
        await test.step("创建新用户", async () => {
            invitedUser = await createUser(sharepage, invitedUser, process.env.TEST_INITIAL_PASSWORD!, "platform-regular");
        });
        await test.step("修改用户信息，用户名置灰不可修改", async () => {
        await sharepage.getByRole('row', { name: invitedUser + " platform-regular" }).getByLabel('more').click();
        await sharepage.getByRole('button', { name: '修改' }).click();
        await expect(sharepage.getByRole('textbox', { name: '* 用户名称' })).toBeDisabled();
        await sharepage.getByRole('button', { name: '取 消' }).click();
        });
        await test.step("删除创建的用户", async () => {
            await deleteUser(sharepage, invitedUser, "platform-regular");
        });      
    });

    test("309【用户管理-043】容器平台-权限管理-集群成员，操作列-移除，取消移除", {
            tag: ['@v24.09', '@user-management']
        }, async ( ) => {
        let invitedUser = "test-user";
        await test.step("创建新用户", async () => {
            invitedUser = await createUser(sharepage, invitedUser, process.env.TEST_INITIAL_PASSWORD!, "platform-regular");
        });
        await test.step("进入集群成员界面", async () => {
            await sharepage.getByRole('img', { name: 'home' }).locator('path').click();
            await sharepage.getByRole('link', { name: '集群成员' }).click();
            await expect(sharepage.getByRole('navigation').getByText('集群成员')).toBeVisible();
        });
        await test.step("邀请用户并绑定集群角色", async () => {
            await inviteUserWithRole(sharepage, invitedUser, "cluster-admin");
            await expect(sharepage.getByText(`邀请${invitedUser}成员成功！`)).toBeVisible();
            await expect(sharepage.getByRole('cell', { name: invitedUser, exact: true })).toBeVisible();
            await sharepage.waitForTimeout(300);
        });
        await test.step("移除集群成员时取消", async () => {
            await sharepage.getByRole('row', { name: invitedUser + " " + "cluster-admin" + " more" }).locator('svg').click();
            await sharepage.getByRole('button', { name: '移除' }).click();
            await sharepage.getByRole('button', { name: '取 消' }).click();
            await expect(sharepage.getByRole('cell', { name: invitedUser, exact: true })).toBeVisible();
        });
        await test.step("删除创建的用户", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await deleteUser(sharepage, invitedUser, "platform-regular");
        });      
    });

    test("302【用户管理-047】容器平台-权限管理-集群成员，表头排序", {
            tag: ['@v24.09', '@user-management']
        }, async ( ) => {
        let testUser1 = "test-user";
        let testUser2 = "test-user";
        const platformRole = "platform-regular"
        await test.step("进入用户管理", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await expect(sharepage.getByRole('navigation').getByText('用户管理')).toBeVisible();
        });
        await test.step("创建两个用户", async () => {
            testUser1 = await createUser(sharepage, testUser1, process.env.TEST_INITIAL_PASSWORD!, platformRole);
            testUser2 = await createUser(sharepage, testUser2, process.env.TEST_INITIAL_PASSWORD!, platformRole);
        });

        await test.step("进入集群成员界面", async () => {
            await sharepage.getByRole('img', { name: 'home' }).locator('path').click();
            await sharepage.getByRole('link', { name: '集群成员' }).click();
            await expect(sharepage.getByRole('navigation').getByText('集群成员')).toBeVisible();
        });
        await test.step("邀请用户并绑定集群角色", async () => {
            // 等待表格数据行加载完成
            await sharepage.waitForSelector('table tr', { timeout: 5000 });
            await inviteUserWithRole(sharepage, testUser1, "cluster-admin");
            await expect(sharepage.getByText(`邀请${testUser1}成员成功！`)).toBeVisible();
            await inviteUserWithRole(sharepage, testUser2, "cluster-admin");
            await expect(sharepage.getByText(`邀请${testUser2}成员成功！`)).toBeVisible();
            await sharepage.waitForTimeout(300);
        });

        await test.step("用户名称排序", async () => {
            // 初始状态：可能未排序或默认排序
            await sharepage.reload();
            await expect(sharepage.getByRole('cell', { name: testUser1, exact: true })).toBeVisible();
            let userNames = await sharepage.getByRole('table').first().locator('tbody tr.container-platform-table-row td.container-platform-table-cell:first-child').allTextContents();
            // 第一次点击：升序排序
            await sharepage.getByRole('table').getByText('成员名称').click();
            await sharepage.waitForTimeout(300);
            
            userNames = await sharepage.getByRole('table').first().locator('tbody tr.container-platform-table-row td.container-platform-table-cell:first-child').allTextContents();
            const ascendingNames = [...userNames].sort((a, b) => a.localeCompare(b));
            expect(userNames).toEqual(ascendingNames);

            // 第二次点击：降序排序
            await sharepage.getByRole('table').getByText('成员名称').click();
            await sharepage.waitForTimeout(300);
            
            userNames = await sharepage.getByRole('table').first().locator('tbody tr.container-platform-table-row td.container-platform-table-cell:first-child').allTextContents();
            const descendingNames = [...userNames].sort((a, b) => b.localeCompare(a));
            expect(userNames).toEqual(descendingNames);
        });
        await test.step("删除创建的用户", async () => {
            // await removeClusterUser(sharepage, "admin", "cluster-admin");
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await deleteUser(sharepage, testUser1, platformRole);
            await deleteUser(sharepage, testUser2, platformRole);
        });
    });

    test("304【用户管理-037】容器平台-权限管理-集群成员，取消邀请成员", {
            tag: ['@v24.09', '@user-management']
        }, async ( ) => {
        let invitedUser = "test-user";
        await test.step("创建新用户", async () => {
            invitedUser = await createUser(sharepage, invitedUser, process.env.TEST_INITIAL_PASSWORD!, "platform-regular");
        });
        await test.step("进入集群成员界面", async () => {
            await sharepage.getByRole('img', { name: 'home' }).locator('path').click();
            await sharepage.getByRole('link', { name: '集群成员' }).click();
            await expect(sharepage.getByRole('navigation').getByText('集群成员')).toBeVisible();
        });
        await test.step("邀请用户时取消", async () => {
            await sharepage.getByRole('button', { name: '邀 请' }).click();
            await sharepage.getByRole('row', { name: invitedUser + ' -- plus-circle' }).locator('svg').nth(1).click();
            await sharepage.getByRole('tooltip').locator('span').nth(1).click();
            await sharepage.getByTitle("cluster-admin").locator('div').click();
            await sharepage.getByRole('button', { name: '取 消' }).click();
            await expect(sharepage.getByRole('cell', { name: invitedUser, exact: true })).not.toBeVisible();
        });
        await test.step("删除创建的用户", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await deleteUser(sharepage, invitedUser, "platform-regular");
        });      
    });

    test("305【用户管理-039】容器平台-权限管理-集群成员，邀请成员页面按不存在的成员名称搜索结果正常显示", {
            tag: ['@v24.09', '@user-management']
        }, async ( ) => {
        await test.step("进入集群成员界面", async () => {
            await sharepage.getByRole('link', { name: '集群成员' }).click();
            await expect(sharepage.getByRole('navigation').getByText('集群成员')).toBeVisible();
        });
        await test.step("邀请用户时取消", async () => {
            await sharepage.getByRole('button', { name: '邀 请' }).click();
            await sharepage.getByRole('dialog', { name: '邀请成员' }).getByPlaceholder('搜索成员名称').fill('nonexistentuser');
            await sharepage.getByLabel('邀请成员').getByRole('button', { name: 'search' }).click();
            await expect(sharepage.getByLabel('邀请成员').getByText('暂无数据')).toBeVisible();
            await sharepage.getByRole('button', { name: '取 消' }).click();
        });    
    });

    test("306【用户管理-040】容器平台-权限管理-集群成员，邀请成员页面按成员名称搜索支持模糊查询", {
            tag: ['@v24.09', '@user-management']
        }, async ( ) => {
        let invitedUser = "test-user";
        await test.step("创建新用户", async () => {
            invitedUser = await createUser(sharepage, invitedUser, process.env.TEST_INITIAL_PASSWORD!, "platform-regular");
        });
        await test.step("进入集群成员界面", async () => {
            await sharepage.getByRole('img', { name: 'home' }).locator('path').click();
            await sharepage.getByRole('link', { name: '集群成员' }).click();
            await expect(sharepage.getByRole('navigation').getByText('集群成员')).toBeVisible();
        });
        await test.step("邀请用户时支持模糊搜索", async () => {
            await sharepage.getByRole('button', { name: '邀 请' }).click();
            await sharepage.getByRole('dialog', { name: '邀请成员' }).getByPlaceholder('搜索成员名称').fill('test-user');
            await sharepage.getByLabel('邀请成员').getByRole('button', { name: 'search' }).click();
            await expect(sharepage.getByRole('cell', { name: invitedUser })).toBeVisible();
            await sharepage.getByRole('button', { name: '取 消' }).click();
        });
        await test.step("删除创建的用户", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await deleteUser(sharepage, invitedUser, "platform-regular");
        });      
    });

    test("150【用户管理-038】容器平台-权限管理-集群成员，邀请成员页面按正确的成员名称搜索结果正常显示", {
            tag: ['@v24.09', '@user-management']
        }, async ({ browser }) => {
        let clusterAdminUser = "test-cluster-user";
        let invitedUser = "test-user";
        await test.step("创建集群管理员用户和新用户", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            clusterAdminUser = await createUser(sharepage, clusterAdminUser, process.env.TEST_INITIAL_PASSWORD!, "platform-regular");
            invitedUser = await createUser(sharepage, invitedUser, process.env.TEST_INITIAL_PASSWORD!, "platform-regular");
        });
        await test.step("进入集群成员界面", async () => {
            await sharepage.getByRole('img', { name: 'home' }).locator('path').click();
            await sharepage.getByRole('link', { name: '集群成员' }).click();
            await expect(sharepage.getByRole('navigation').getByText('集群成员')).toBeVisible();
        });
        await test.step("邀请用户并绑定集群管理员角色", async () => {
            await inviteUserWithRole(sharepage, clusterAdminUser, "cluster-admin");
            await expect(sharepage.getByText(`邀请${clusterAdminUser}成员成功！`)).toBeVisible();
            await expect(sharepage.getByRole('cell', { name: clusterAdminUser, exact: true })).toBeVisible();
        });
        await test.step("以集群管理员在独立 context 登录并搜索用户（不登出 admin）", async () => {
            const nonAdminContext = await newContext(browser);
            const nonAdminPage = await nonAdminContext.newPage();
            await nonAdminPage.goto(serviceClusterUrl + loginUrl);
            await login(nonAdminPage, clusterAdminUser, process.env.TEST_INITIAL_PASSWORD!);
            await firstLoginChangePassword(nonAdminPage, process.env.TEST_PASSWORD!);

            await nonAdminPage.getByRole('link', { name: '集群成员' }).click();
            await nonAdminPage.getByRole('button', { name: '邀 请' }).click();
            await nonAdminPage.getByRole('dialog', { name: '邀请成员' }).getByPlaceholder('搜索成员名称').fill(invitedUser);
            await nonAdminPage.getByLabel('邀请成员').getByRole('button', { name: 'search' }).click();
            await expect(nonAdminPage.getByRole('cell', { name: invitedUser })).toBeVisible();

            await nonAdminContext.close();
        });
        await test.step("删除创建的用户", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await deleteUser(sharepage, invitedUser, "platform-regular");
            await deleteUser(sharepage, clusterAdminUser, "platform-regular");
        });      
    });

    test("151【用户管理-044】容器平台-权限管理-集群成员，按正确的成员名称搜索结果正常显示", {
            tag: ['@v24.09', '@user-management']
        }, async ( ) => {
        let invitedUser = "test-user";
        await test.step("创建新用户", async () => {
            invitedUser = await createUser(sharepage, invitedUser, process.env.TEST_INITIAL_PASSWORD!, "platform-regular");
        });
        await test.step("进入集群成员界面", async () => {
            await sharepage.getByRole('img', { name: 'home' }).locator('path').click();
            await sharepage.getByRole('link', { name: '集群成员' }).click();
            await expect(sharepage.getByRole('navigation').getByText('集群成员')).toBeVisible();
        });
        await test.step("邀请用户并绑定集群角色", async () => {
            await inviteUserWithRole(sharepage, invitedUser, "cluster-admin");
            await expect(sharepage.getByText(`邀请${invitedUser}成员成功！`)).toBeVisible();
            await expect(sharepage.getByRole('cell', { name: invitedUser, exact: true })).toBeVisible();
            await sharepage.waitForTimeout(300);
        });
        await test.step("在集群成员界面搜索用户", async () => {
            await sharepage.getByRole('textbox', { name: '搜索成员名称' }).fill(invitedUser);
            await sharepage.getByRole('button', { name: 'search' }).click();
            await expect(sharepage.getByRole('cell', { name: invitedUser })).toBeVisible();
        });
        await test.step("删除创建的用户", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await deleteUser(sharepage, invitedUser, "platform-regular");
        });      
    });

    test("300【用户管理-045】容器平台-权限管理-集群成员，按不存在的成员名称搜索结果正常显示", {
            tag: ['@v24.09', '@user-management']
        }, async ( ) => {
        let invitedUser = "test-user";
        await test.step("创建新用户", async () => {
            invitedUser = await createUser(sharepage, invitedUser, process.env.TEST_INITIAL_PASSWORD!, "platform-regular");
        });
        await test.step("进入集群成员界面", async () => {
            await sharepage.getByRole('img', { name: 'home' }).locator('path').click();
            await sharepage.getByRole('link', { name: '集群成员' }).click();
            await expect(sharepage.getByRole('navigation').getByText('集群成员')).toBeVisible();
        });
        await test.step("邀请用户并绑定集群角色", async () => {
            await inviteUserWithRole(sharepage, invitedUser, "cluster-admin");
            await expect(sharepage.getByText(`邀请${invitedUser}成员成功！`)).toBeVisible();
            await expect(sharepage.getByRole('cell', { name: invitedUser, exact: true })).toBeVisible();
            await sharepage.waitForTimeout(300);
        });
        await test.step("在集群成员界面搜索用户", async () => {
            await sharepage.getByRole('textbox', { name: '搜索成员名称' }).fill("nonexistentuser");
            await sharepage.getByRole('button', { name: 'search' }).click();
            await expect(sharepage.getByText('暂无数据')).toBeVisible();
        });
        await test.step("删除创建的用户", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await deleteUser(sharepage, invitedUser, "platform-regular");
        });      
    });

    test("301【用户管理-046】容器平台-权限管理-集群成员，按成员名称搜索支持模糊查询", {
            tag: ['@v24.09', '@user-management']
        }, async ( ) => {
        let invitedUser = "test-user";
        await test.step("创建新用户", async () => {
            invitedUser = await createUser(sharepage, invitedUser, process.env.TEST_INITIAL_PASSWORD!, "platform-regular");
        });
        await test.step("进入集群成员界面", async () => {
            await sharepage.getByRole('img', { name: 'home' }).locator('path').click();
            await sharepage.getByRole('link', { name: '集群成员' }).click();
            await expect(sharepage.getByRole('navigation').getByText('集群成员')).toBeVisible();
        });
        await test.step("邀请用户并绑定集群角色", async () => {
            await inviteUserWithRole(sharepage, invitedUser, "cluster-admin");
            await expect(sharepage.getByText(`邀请${invitedUser}成员成功！`)).toBeVisible();
            await expect(sharepage.getByRole('cell', { name: invitedUser, exact: true })).toBeVisible();
            await sharepage.waitForTimeout(300);
        });
        await test.step("在集群成员界面搜索用户(模糊搜索)", async () => {
            await sharepage.getByRole('textbox', { name: '搜索成员名称' }).fill("test-user");
            await sharepage.getByRole('button', { name: 'search' }).click();
            await expect(sharepage.getByRole('cell', { name: invitedUser })).toBeVisible();
        });
        await test.step("删除创建的用户", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await deleteUser(sharepage, invitedUser, "platform-regular");
        });      
    });

    test("315【用户管理-026】用户管理页面，表头筛选", {
            tag: ['@v24.09', '@user-management']
        }, async ( ) => {
        let invitedUser = "test-user";
        await test.step("创建新用户", async () => {
            invitedUser = await createUser(sharepage, invitedUser, process.env.TEST_INITIAL_PASSWORD!, "platform-regular");
        });
        await test.step("角色筛选", async () => {
            await sharepage.getByRole('button', { name: 'filter' }).click();
            await sharepage.getByRole('menuitem', { name: 'platform-regular' }).locator('span').first().click();
            await sharepage.getByRole('button', { name: /确\s*定|确\s*认/ }).click();
            await expect(sharepage.getByRole('link', { name: invitedUser })).toBeVisible();
            await expect(sharepage.getByRole('link', { name: "admin" })).not.toBeVisible();
        });
        await test.step("删除创建的用户", async () => {
            await deleteUser(sharepage, invitedUser, "platform-regular");
        });      
    });

     test("316【用户管理-022】用户管理页面，按正确的用户名称搜索结果正常显示" +
 	     "【用户管理-023】用户管理页面，按不存在的用户名称搜索结果展示为空", {
            tag: ['@v24.09', '@多集群多用户分权分域认证鉴权']
        }, async () => {
        let testUser = "test-user";
        const platformRole = "platform-regular";
        await test.step("进入用户管理", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await expect(sharepage.getByRole('navigation').getByText('用户管理')).toBeVisible();
        });
        await test.step("创建新用户", async () => {
            testUser = await createUser(sharepage, testUser, process.env.TEST_INITIAL_PASSWORD!, platformRole);
        });
        await test.step("按正确的用户名称搜索", async () => {
            await sharepage.getByRole('textbox', { name: '搜索用户名称' }).fill(testUser);
            await sharepage.getByRole('button', { name: 'search' }).click();
            await expect(sharepage.getByRole('link', { name: testUser })).toBeVisible();
        });
        await test.step("按不存在的用户名称搜索", async () => {
            await sharepage.getByRole('textbox', { name: '搜索用户名称' }).fill("nonexistentuser");
            await sharepage.getByRole('button', { name: 'search' }).click();
            await expect(sharepage.getByText('暂无数据')).toBeVisible();
        });
        await test.step("清空搜索条件", async () => {
            await sharepage.getByRole('textbox', { name: '搜索用户名称' }).fill("");
            await sharepage.getByRole('button', { name: 'search' }).click();
            await sharepage.waitForTimeout(300);
        });
        await test.step("删除创建的用户", async () => {
            await deleteUser(sharepage, testUser, platformRole);
        });
    });
});