import { existsSync, unlinkSync } from "fs";
import path from "path";
import { fileURLToPath } from "url";
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
    editClusterUser
} from "./utils/user-management";

const loginUrl = "/oauth2/auth/login/fuyaoPasswordProvider"
const homeUrl = "/container_platform/overview"

test.describe("用户管理", () => {
    const AUTH_FILE = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "./user-management-platform-auth.json");
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
        await expect(authpage.getByText(process.env.TEST_USERNAME!)).toBeVisible();
        await context.storageState({ path: AUTH_FILE });
        await authpage.close();
        await context.close();
    });
    test.afterAll(async ({ browser }) => {
        if (existsSync(AUTH_FILE)) {
            const context = await newContext(browser, { storageState: AUTH_FILE });
            const authpage = await context.newPage();
            await authpage.goto(serviceClusterUrl + homeUrl);
            await logout(authpage);
            await authpage.close();
            await context.close();
            unlinkSync(AUTH_FILE);
        }
    });
    test.beforeEach(async ({ browser }) => {
        const shareContext = await newContext(browser, {
            storageState: AUTH_FILE,
        });
        sharepage = await shareContext.newPage();
        await test.step("返回首页", async () => {
            await sharepage.goto(serviceClusterUrl + homeUrl);
        });
    });
    test("143【用户管理-054】平台安装时自动创建平台级的默认用户", {
            tag: ['@v24.09', '@user-management']
        }, async () => {
        await test.step("进入用户管理", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await expect(sharepage.getByRole('navigation').getByText('用户管理')).toBeVisible();
        });
        await test.step("查看默认admin用户", async () => {
            await expect(sharepage.getByRole('link', { name: 'admin', exact: true })).toBeVisible();
        });      
    });

    test("144【用户管理-055】平台安装时自动创建集群级默认角色", {
            tag: ['@v24.09', '@user-management']
        }, async () => {
        await test.step("进入集群角色", async () => {
            await sharepage.getByRole('link', { name: '集群角色' }).click();
            await expect(sharepage.getByRole('navigation').getByText('集群角色')).toBeVisible();
        });
        await test.step("查看默认集群级用户", async () => {
            await expect(sharepage.getByRole('cell', { name: 'cluster-admin', exact: true })).toBeVisible();
            await expect(sharepage.getByRole('cell', { name: 'cluster-editor', exact: true })).toBeVisible();
            await expect(sharepage.getByRole('cell', { name: 'cluster-viewer', exact: true })).toBeVisible();
        });      
    });

    test("136【用户管理-061】平台管理员权限的用户邀请用户并绑定集群角色", {
            tag: ['@v24.09', '@user-management']
        }, async ({}) => {
        let invitedUser = "test-user";
        await test.step("进入用户管理", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await expect(sharepage.getByRole('navigation').getByText('用户管理')).toBeVisible();
        });
        await test.step("创建被邀请用户", async () => {
            invitedUser = await createUser(sharepage, invitedUser, process.env.TEST_INITIAL_PASSWORD!, "platform-admin");
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
        });
        await test.step("删除创建的用户", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await deleteUser(sharepage, invitedUser, "platform-admin")
        });      
    });

    test("137【用户管理-062】集群管理员权限的用户邀请用户并绑定集群角色", {
            tag: ['@v24.09', '@user-management']
        }, async ({ browser }) => {
        let clusterAdminUser = "test-cluster-user";
        let invitedUser = "test-user";
        await test.step("创建集群管理员用户和被邀请用户", async () => {
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
        await test.step("以集群管理员在独立 context 登录并邀请用户（不登出 admin）", async () => {
            const nonAdminContext = await newContext(browser);
            const nonAdminPage = await nonAdminContext.newPage();
            await nonAdminPage.goto(serviceClusterUrl + loginUrl);
            await login(nonAdminPage, clusterAdminUser, process.env.TEST_INITIAL_PASSWORD!);
            await firstLoginChangePassword(nonAdminPage, process.env.TEST_PASSWORD!);

            await nonAdminPage.getByRole('link', { name: '集群成员' }).click();
            await inviteUserWithRole(nonAdminPage, invitedUser, "cluster-editor");
            await expect(nonAdminPage.getByText(`邀请${invitedUser}成员成功！`)).toBeVisible();
            await expect(nonAdminPage.getByRole('cell', { name: invitedUser, exact: true })).toBeVisible();

            await nonAdminContext.close();
        });
        await test.step("删除创建的用户", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await deleteUser(sharepage, invitedUser, "platform-regular");
            await deleteUser(sharepage, clusterAdminUser, "platform-regular");
        });      
    });

    test("138【用户管理-064】平台管理员权限的用户修改集群用户的集群角色", {
            tag: ['@v24.09', '@user-management']
        }, async ({ }) => {
        let targetUser = "test-user";
        await test.step("进入用户管理", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await expect(sharepage.getByRole('navigation').getByText('用户管理')).toBeVisible();
        });
        await test.step("创建用户", async () => {
            targetUser = await createUser(sharepage, targetUser, process.env.TEST_INITIAL_PASSWORD!, "platform-admin");
        });
        await test.step("进入集群成员界面", async () => {
            await sharepage.getByRole('img', { name: 'home' }).locator('path').click();
            await sharepage.getByRole('link', { name: '集群成员' }).click();
            await expect(sharepage.getByRole('navigation').getByText('集群成员')).toBeVisible();
        });
        await test.step("邀请用户并绑定集群角色", async () => {
            await inviteUserWithRole(sharepage, targetUser, "cluster-admin");
            await expect(sharepage.getByText(`邀请${targetUser}成员成功！`)).toBeVisible();
            await expect(sharepage.getByRole('cell', { name: targetUser, exact: true })).toBeVisible();
        });
        await test.step("修改用户集群角色", async () => {
            await editClusterUser(sharepage, targetUser, "cluster-admin", "cluster-editor");
            await expect(sharepage.getByText(`编辑${targetUser}成员成功！`)).toBeVisible();
        });
        await test.step("删除创建的用户", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await deleteUser(sharepage, targetUser, "platform-admin")
        });      
    });

    test("139【用户管理-065】集群管理员权限的用户修改集群用户的集群角色", {
            tag: ['@v24.09', '@user-management']
        }, async ({ browser }) => {
        let clusterAdminUser = "test-cluster-user";
        let targetUser = "test-user";
        await test.step("创建集群管理员用户和被邀请用户", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            clusterAdminUser = await createUser(sharepage, clusterAdminUser, process.env.TEST_INITIAL_PASSWORD!, "platform-regular");
            targetUser = await createUser(sharepage, targetUser, process.env.TEST_INITIAL_PASSWORD!, "platform-regular");
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
        await test.step("以集群管理员在独立 context 登录完成邀请与修改（不登出 admin）", async () => {
            const nonAdminContext = await newContext(browser);
            const nonAdminPage = await nonAdminContext.newPage();
            await nonAdminPage.goto(serviceClusterUrl + loginUrl);
            await login(nonAdminPage, clusterAdminUser, process.env.TEST_INITIAL_PASSWORD!);
            await firstLoginChangePassword(nonAdminPage, process.env.TEST_PASSWORD!);

            await nonAdminPage.getByRole('link', { name: '集群成员' }).click();
            await inviteUserWithRole(nonAdminPage, targetUser, "cluster-editor");
            await expect(nonAdminPage.getByText(`邀请${targetUser}成员成功！`)).toBeVisible();
            await expect(nonAdminPage.getByRole('cell', { name: targetUser, exact: true })).toBeVisible();

            await editClusterUser(nonAdminPage, targetUser, "cluster-editor", "cluster-admin");
            await expect(nonAdminPage.getByText(`编辑${targetUser}成员成功！`)).toBeVisible();

            await nonAdminContext.close();
        });
        await test.step("删除创建的用户", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await deleteUser(sharepage, targetUser, "platform-regular");
            await deleteUser(sharepage, clusterAdminUser, "platform-regular");
        });     
    });

    test("140【用户管理-067】平台管理员权限的用户移除集群用户", {
            tag: ['@v24.09', '@user-management']
        }, async ({}) => {
        let targetUser = "test-user";
        await test.step("进入用户管理", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await expect(sharepage.getByRole('navigation').getByText('用户管理')).toBeVisible();
        });
        await test.step("创建用户", async () => {
            targetUser = await createUser(sharepage, targetUser, process.env.TEST_INITIAL_PASSWORD!, "platform-admin");
        });
        await test.step("进入集群成员界面", async () => {
            await sharepage.getByRole('img', { name: 'home' }).locator('path').click();
            await sharepage.getByRole('link', { name: '集群成员' }).click();
            await expect(sharepage.getByRole('navigation').getByText('集群成员')).toBeVisible();
        });
        await test.step("邀请用户并绑定集群角色", async () => {
            await inviteUserWithRole(sharepage, targetUser, "cluster-admin");
            await expect(sharepage.getByText(`邀请${targetUser}成员成功！`)).toBeVisible();
            await expect(sharepage.getByRole('cell', { name: targetUser, exact: true })).toBeVisible();
        });
        await test.step("移除用户集群角色", async () => {
            await removeClusterUser(sharepage, targetUser, "cluster-admin");
        });
        await test.step("删除创建的用户", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await deleteUser(sharepage, targetUser, "platform-admin")
        });      
    });

    test("141【用户管理-068】集群管理员权限的用户移除集群用户", {
            tag: ['@v24.09', '@user-management']
        }, async ({ browser }) => {
        let clusterAdminUser = "test-cluster-user";
        let targetUser = "test-user";
        await test.step("创建集群管理员用户和被邀请用户", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            clusterAdminUser = await createUser(sharepage, clusterAdminUser, process.env.TEST_INITIAL_PASSWORD!, "platform-regular");
            targetUser = await createUser(sharepage, targetUser, process.env.TEST_INITIAL_PASSWORD!, "platform-regular");
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
        await test.step("以集群管理员在独立 context 登录并邀请/移除成员（不登出 admin）", async () => {
            const nonAdminContext = await newContext(browser);
            const nonAdminPage = await nonAdminContext.newPage();
            await nonAdminPage.goto(serviceClusterUrl + loginUrl);
            await login(nonAdminPage, clusterAdminUser, process.env.TEST_INITIAL_PASSWORD!);
            await firstLoginChangePassword(nonAdminPage, process.env.TEST_PASSWORD!);

            await nonAdminPage.getByRole('link', { name: '集群成员' }).click();
            await inviteUserWithRole(nonAdminPage, targetUser, "cluster-editor");
            await expect(nonAdminPage.getByText(`邀请${targetUser}成员成功！`)).toBeVisible();
            await expect(nonAdminPage.getByRole('cell', { name: targetUser, exact: true })).toBeVisible();

            await removeClusterUser(nonAdminPage, targetUser, "cluster-editor");

            await nonAdminContext.close();
        });
        await test.step("删除创建的用户", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await deleteUser(sharepage, targetUser, "platform-regular");
            await deleteUser(sharepage, clusterAdminUser, "platform-regular");
        });    
    });

    test("142【用户管理-052】角色为platform-regular的用户登录后会进入单独的页面，看不了容器平台任何内容", {
            tag: ['@v24.09', '@user-management']
        }, async ({ browser }) => {
        let regularUser = "test-user";
        await test.step("创建platform-regular角色用户", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            regularUser = await createUser(sharepage, regularUser, process.env.TEST_INITIAL_PASSWORD!, "platform-regular");
        });
        await test.step("以 platform-regular 用户在独立 context 登录并验证无权限（不登出 admin）", async () => {
            const nonAdminContext = await newContext(browser);
            const nonAdminPage = await nonAdminContext.newPage();
            await nonAdminPage.goto(serviceClusterUrl + loginUrl);
            await login(nonAdminPage, regularUser, process.env.TEST_INITIAL_PASSWORD!);
            await firstLoginChangePassword(nonAdminPage, process.env.TEST_PASSWORD!);

            await expect(nonAdminPage.getByText('当前用户尚无访问权限请联系管理员添加用户权限')).toBeVisible();

            await nonAdminContext.close();
        });
        await test.step("删除创建的用户", async () => {
            await deleteUser(sharepage, regularUser, "platform-regular");
        });  
    });

    test("145【用户管理-056】平台管理员权限的用户可以查看平台角色列表和集群角色列表", {
            tag: ['@v24.09', '@user-management']
        }, async () => {
        await test.step("可以查看平台角色列表", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await expect(sharepage.getByRole('table').getByText('用户名称')).toBeVisible();
        });
        await test.step("可以查看集群角色列表", async () => {
            await sharepage.getByRole('img', { name: 'home' }).locator('path').click();
            await sharepage.getByRole('link', { name: '集群成员' }).click();
            await expect(sharepage.getByText('成员名称')).toBeVisible();
        });  
    });

    test("146【用户管理-057】集群管理员权限的用户无法查看平台角色列表，可以查看集群角色列表", {
            tag: ['@v24.09', '@user-management']
        }, async ({ browser }) => {
        let clusterAdminUser = "test-cluster-user";
        await test.step("创建集群管理员用户", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            clusterAdminUser = await createUser(sharepage, clusterAdminUser, process.env.TEST_INITIAL_PASSWORD!, "platform-regular");
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
        await test.step("以集群管理员在独立 context 登录并验证权限（不登出 admin）", async () => {
            const nonAdminContext = await newContext(browser);
            const nonAdminPage = await nonAdminContext.newPage();
            await nonAdminPage.goto(serviceClusterUrl + loginUrl);
            await login(nonAdminPage, clusterAdminUser, process.env.TEST_INITIAL_PASSWORD!);
            await firstLoginChangePassword(nonAdminPage, process.env.TEST_PASSWORD!);

            await nonAdminPage.getByRole('link', { name: '用户管理' }).click();
            await expect(nonAdminPage.getByText('当前用户尚无访问权限请联系管理员添加用户权限')).toBeVisible();

            await nonAdminPage.getByRole('img', { name: 'home' }).locator('path').click();
            await nonAdminPage.getByRole('link', { name: '集群成员' }).click();
            await expect(nonAdminPage.getByText('成员名称')).toBeVisible();

            await nonAdminContext.close();
        });
        await test.step("删除创建的用户", async () => {
            await sharepage.waitForTimeout(300);
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await deleteUser(sharepage, clusterAdminUser, "platform-regular");
        });    
    });

    test("298【用户管理-058】集群editor权限的用户无法查看平台角色列表，无法查看集群角色列表", {
            tag: ['@v24.09', '@user-management']
        }, async ({ browser }) => {
        let editorUser = "test-editor-user";
        await test.step("创建集群editor用户", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            editorUser = await createUser(sharepage, editorUser, process.env.TEST_INITIAL_PASSWORD!, "platform-regular");
        });
        await test.step("进入集群成员界面", async () => {
            await sharepage.getByRole('img', { name: 'home' }).locator('path').click();
            await sharepage.getByRole('link', { name: '集群成员' }).click();
            await expect(sharepage.getByRole('navigation').getByText('集群成员')).toBeVisible();
        });
        await test.step("邀请用户并绑定集群editor角色", async () => {
            await inviteUserWithRole(sharepage, editorUser, "cluster-editor");
            await expect(sharepage.getByText(`邀请${editorUser}成员成功！`)).toBeVisible();
            await expect(sharepage.getByRole('cell', { name: editorUser, exact: true })).toBeVisible();
        });
        await test.step("以集群editor在独立 context 登录并验证权限（不登出 admin）", async () => {
            const nonAdminContext = await newContext(browser);
            const nonAdminPage = await nonAdminContext.newPage();
            await nonAdminPage.goto(serviceClusterUrl + loginUrl);
            await login(nonAdminPage, editorUser, process.env.TEST_INITIAL_PASSWORD!);
            await firstLoginChangePassword(nonAdminPage, process.env.TEST_PASSWORD!);

            await nonAdminPage.getByRole('link', { name: '用户管理' }).click();
            await expect(nonAdminPage.getByText('当前用户尚无访问权限请联系管理员添加用户权限')).toBeVisible();

            await nonAdminPage.getByRole('img', { name: 'home' }).locator('path').click();
            await nonAdminPage.getByRole('link', { name: '集群成员' }).click();
            await expect(nonAdminPage.getByText('当前用户尚无访问权限请联系管理员添加用户权限')).toBeVisible();

            await nonAdminContext.close();
        });
        await test.step("删除创建的用户", async () => {
            await sharepage.waitForTimeout(300);
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await deleteUser(sharepage, editorUser, "platform-regular");
        });    
    });

    test("299【用户管理-059】集群viewer权限的用户无法查看平台角色列表，无法查看集群角色列表", {
            tag: ['@v24.09', '@user-management']
        }, async ({ browser }) => {
        let viewerUser = "test-viewer-user";
        await test.step("创建集群viewer用户", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            viewerUser = await createUser(sharepage, viewerUser, process.env.TEST_INITIAL_PASSWORD!, "platform-regular");
        });
        await test.step("进入集群成员界面", async () => {
            await sharepage.getByRole('img', { name: 'home' }).locator('path').click();
            await sharepage.getByRole('link', { name: '集群成员' }).click();
            await expect(sharepage.getByRole('navigation').getByText('集群成员')).toBeVisible();
        });
        await test.step("邀请用户并绑定集群viewer角色", async () => {
            await inviteUserWithRole(sharepage, viewerUser, "cluster-viewer");
            await expect(sharepage.getByText(`邀请${viewerUser}成员成功！`)).toBeVisible();
            await expect(sharepage.getByRole('cell', { name: viewerUser, exact: true })).toBeVisible();
        });
        await test.step("以集群viewer在独立 context 登录并验证权限（不登出 admin）", async () => {
            const nonAdminContext = await newContext(browser);
            const nonAdminPage = await nonAdminContext.newPage();
            await nonAdminPage.goto(serviceClusterUrl + loginUrl);
            await login(nonAdminPage, viewerUser, process.env.TEST_INITIAL_PASSWORD!);
            await firstLoginChangePassword(nonAdminPage, process.env.TEST_PASSWORD!);

            await nonAdminPage.getByRole('link', { name: '用户管理' }).click();
            await expect(nonAdminPage.getByText('当前用户尚无访问权限请联系管理员添加用户权限')).toBeVisible();

            await nonAdminPage.getByRole('img', { name: 'home' }).locator('path').click();
            await nonAdminPage.getByRole('link', { name: '集群成员' }).click();
            await expect(nonAdminPage.getByText('当前用户尚无访问权限请联系管理员添加用户权限')).toBeVisible();

            await nonAdminContext.close();
        });
        await test.step("删除创建的用户", async () => {
            await sharepage.waitForTimeout(300);
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await deleteUser(sharepage, viewerUser, "platform-regular");
        });    
    });

    test("311【用户管理-036】容器平台-权限管理-集群成员，邀请成员时可修改角色功能", {
            tag: ['@v24.09', '@user-management']
        }, async ({}) => {
        let invitedUser = "test-user";
        await test.step("进入用户管理", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await expect(sharepage.getByRole('navigation').getByText('用户管理')).toBeVisible();
        });
        await test.step("创建被邀请用户", async () => {
            invitedUser = await createUser(sharepage, invitedUser, process.env.TEST_INITIAL_PASSWORD!, "platform-admin");
        });
        await test.step("进入集群成员界面", async () => {
            await sharepage.getByRole('img', { name: 'home' }).locator('path').click();
            await sharepage.getByRole('link', { name: '集群成员' }).click();
            await expect(sharepage.getByRole('navigation').getByText('集群成员')).toBeVisible();
        });
        await test.step("邀请用户时可修改集群角色", async () => {
            // 打开邀请成员对话框并添加被邀请用户（点击 plus-circle 触发角色下拉）
            await sharepage.getByRole('button', { name: '邀 请' }).click();
            await expect(sharepage.getByRole('dialog', { name: '邀请成员' })).toBeVisible();
            await sharepage.getByRole('row').filter({ hasText: invitedUser }).getByLabel('plus-circle').click();
            // 首次选择集群角色 cluster-admin：角色下拉渲染在 tooltip 中，点击 selector 打开。
            // 选项无 title 属性，且 DOM 中存在隐藏的 role=option 与遗留的 display:none 下拉，
            // 故用 :visible 作用域内的 option-content 精确定位可点击项（参见 utils/user-management）
            await sharepage.getByRole('tooltip').locator('.container-platform-select-selector').click();
            await sharepage.locator('.container-platform-select-dropdown:visible .container-platform-select-item-option-content', { hasText: 'cluster-admin' }).click();
            await expect(sharepage.getByRole('img', { name: 'check-circle' }).first()).toBeVisible();
            // 首次选择后 tooltip 仍残留、其下拉可能仍展开；此时直接点击行内下拉会与 tooltip 下拉互相 toggle（闪烁），
            // 导致重选失败。点击对话框标题（非交互元素）收起 tooltip，并等待所有下拉关闭，回到干净状态后再修改角色。
            await sharepage.getByRole('dialog', { name: '邀请成员' }).locator('.container-platform-modal-title').click();
            await expect(sharepage.locator('.container-platform-select-dropdown:visible')).toHaveCount(0);
            // 修改集群角色为 cluster-viewer：重新打开行内角色下拉并选择
            await sharepage.getByRole('row').filter({ hasText: invitedUser }).locator('.container-platform-select-selector').click();
            await sharepage.locator('.container-platform-select-dropdown:visible .container-platform-select-item-option-content', { hasText: 'cluster-viewer' }).click();
            await sharepage.getByRole('button', { name: /确\s*定|确\s*认/ }).click();
            await expect(sharepage.getByText(`邀请${invitedUser}成员成功！`)).toBeVisible();
            await sharepage.waitForTimeout(300);
            await expect(sharepage.getByRole('cell', { name: 'cluster-viewer' }).first()).toBeVisible();
        });
        await test.step("删除创建的用户", async () => {
            await sharepage.getByRole('link', { name: '用户管理' }).click();
            await deleteUser(sharepage, invitedUser, "platform-admin")
        });      
    });
})