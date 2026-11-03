import { expect, Locator, test } from "@playwright/test";
import {
    verifyOverviewSort,
    deployComponents,
    cleanupComponents,
    deployHelmReleasesByApi,
    cleanupHelmReleasesByApi,
} from "./utils/application";
import {
    totalTimeout,
    serviceClusterUrl,
    componentNameExt,
    componentNameExt2,
    appPlan,
    extPlan,
    APP_PREFIX,
    EXT_PREFIX,
    type DeployPlan,
} from "./utils/constants";

test.describe("应用管理-概览", () => {
    test.describe.configure({ timeout: totalTimeout });
    let createdApps: DeployPlan[] = [];

    test.beforeAll(async ({ browser }) => {
        test.setTimeout(totalTimeout);
        createdApps = [];
        await deployComponents(browser, [appPlan(), appPlan()], createdApps);
    });

    test.afterAll(async ({ browser }) => {
        test.setTimeout(totalTimeout);
        await cleanupComponents(browser, createdApps);
        createdApps = [];
    });

    test.beforeEach(async ({ page }) => {
        await page.goto(serviceClusterUrl + "/container_platform/applicationManageHelm");
        await expect(page.getByRole("table")).toBeVisible();
    });

    test("【应用运维-应用管理概览-007】应用管理-概览-刷新-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        const appName = createdApps[0].appName;
        await page.getByRole("textbox", { name: /搜索.*名称/ }).fill(appName);
        await page.getByRole("button", { name: "search" }).click();
        await expect(page.getByRole("link", { name: appName })).toBeVisible();
        await page.getByRole("button", { name: "sync" }).click();
        await expect(page.getByRole("link", { name: appName })).toBeVisible();
    });

    test("【应用运维-应用管理概览-008】应用管理-概览-排序-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await page.getByRole("textbox", { name: /搜索.*名称/ }).fill(APP_PREFIX);
        await page.getByRole("button", { name: "search" }).click();
        await expect(page.getByRole("table")).toBeVisible();
        await verifyOverviewSort(page, {
            nameHeader: "应用名称",
            nameColIndex: 1,
            updateTimeHeader: "更新时间",
            updateTimeColIndex: 3,
        });
    });

    test("【应用运维-应用管理概览-002/009】应用管理-概览-搜索/筛选-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        const appName = createdApps[0].appName;
        const appNamePrefix = createdApps[0].appNamePrefix ?? APP_PREFIX;

        await test.step("搜索：全称命中", async () => {
            const search = page.getByRole("textbox", { name: /搜索.*名称/ });
            await search.fill(appName);
            await search.press("Enter");
            await expect(page.getByRole("link", { name: appName })).toBeVisible();
        });
        await test.step("搜索：部分命中", async () => {
            const search = page.getByRole("textbox", { name: /搜索.*名称/ });
            await search.fill(appNamePrefix);
            await search.press("Enter");
            await expect(page.getByRole("link", { name: createdApps[0].appName })).toBeVisible();
            await expect(page.getByRole("link", { name: createdApps[1].appName })).toBeVisible();
        });
        await test.step("搜索：大小写命中", async () => {
            const search = page.getByRole("textbox", { name: /搜索.*名称/ });
            await search.fill(appName.toUpperCase());
            await search.press("Enter");
            await expect(page.getByRole("link", { name: appName })).toBeVisible();
        });
        await test.step("搜索：无匹配关键字", async () => {
            const search = page.getByRole("textbox", { name: /搜索.*名称/ });
            await search.fill("zzz-nomatch-xyz");
            await search.press("Enter");
            await expect(page.getByRole("link", { name: createdApps[0].appName })).toHaveCount(0);
            await expect(page.getByRole("link", { name: createdApps[1].appName })).toHaveCount(0);
        });
        await test.step("按状态筛选部署成功", async () => {
            const search = page.getByRole("textbox", { name: /搜索.*名称/ });
            await search.fill(appNamePrefix);
            await search.press("Enter");
            await page.getByLabel("状态").getByRole("button", { name: "filter" }).click();
            await page.getByRole("menuitem", { name: "部署成功" }).click();
            await page.getByRole("button", { name: /确\s*定/ }).click();
            await expect(page.getByRole("table")).toContainText("部署成功");
        });
    });
});

test.describe("扩展管理-概览", () => {
    test.describe.configure({ timeout: totalTimeout });
    let createdExts: DeployPlan[] = [];

    test.beforeAll(async ({ browser }) => {
        test.setTimeout(totalTimeout);
        createdExts = [];
        await deployComponents(
            browser,
            [extPlan(componentNameExt), extPlan(componentNameExt2)],
            createdExts,
        );
    });

    test.afterAll(async ({ browser }) => {
        test.setTimeout(totalTimeout);
        await cleanupComponents(browser, createdExts);
        createdExts = [];
    });

    test.beforeEach(async ({ page }) => {
        await page.goto(serviceClusterUrl + "/container_platform/extendManage");
        await expect(page.getByRole("table")).toBeVisible();
    });

    test("【应用运维-扩展组件管理概览-007】扩展管理-概览-刷新-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        const appName = createdExts[0].appName;
        await page.getByRole("textbox", { name: /搜索.*名称/ }).fill(appName);
        await page.getByRole("button", { name: "search" }).click();
        await expect(page.getByRole("link", { name: appName })).toBeVisible();
        await page.getByRole("button", { name: "sync" }).click();
        await expect(page.getByRole("link", { name: appName })).toBeVisible();
        await expect(page.getByRole("table")).toContainText("部署成功");
    });

    test("【应用运维-扩展管理概览-008】扩展管理-概览-排序-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await page.getByRole("textbox", { name: /搜索.*名称/ }).fill(EXT_PREFIX);
        await page.getByRole("button", { name: "search" }).click();
        await expect(page.getByRole("table")).toBeVisible();
        await verifyOverviewSort(page, {
            nameHeader: "扩展组件名称",
            nameColIndex: 1,
            updateTimeHeader: "更新时间",
            updateTimeColIndex: 3,
        });
    });

    test("【应用运维-扩展管理概览-002/009】扩展管理-概览-搜索/筛选-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        const extName = createdExts[0].appName;
        const extNamePrefix = createdExts[0].appNamePrefix ?? EXT_PREFIX;

        await test.step("搜索：全称命中", async () => {
            const search = page.getByRole("textbox", { name: /搜索.*名称/ });
            await search.fill(extName);
            await search.press("Enter");
            await expect(page.getByRole("link", { name: extName })).toBeVisible();
        });
        await test.step("搜索：部分命中", async () => {
            const search = page.getByRole("textbox", { name: /搜索.*名称/ });
            await search.fill(extNamePrefix);
            await search.press("Enter");
            await expect(page.getByRole("link", { name: createdExts[0].appName })).toBeVisible();
            await expect(page.getByRole("link", { name: createdExts[1].appName })).toBeVisible();
        });
        await test.step("搜索：大小写命中", async () => {
            const search = page.getByRole("textbox", { name: /搜索.*名称/ });
            await search.fill(extName.toUpperCase());
            await search.press("Enter");
            await expect(page.getByRole("link", { name: extName })).toBeVisible();
        });
        await test.step("搜索：无匹配关键字", async () => {
            const search = page.getByRole("textbox", { name: /搜索.*名称/ });
            await search.fill("zzz-nomatch-xyz");
            await search.press("Enter");
            await expect(page.getByRole("link", { name: createdExts[0].appName })).toHaveCount(0);
            await expect(page.getByRole("link", { name: createdExts[1].appName })).toHaveCount(0);
        });
        await test.step("按状态筛选部署成功", async () => {
            const search = page.getByRole("textbox", { name: /搜索.*名称/ });
            await search.fill(extNamePrefix);
            await search.press("Enter");
            await page.getByLabel("状态").getByRole("button", { name: "filter" }).click();
            await page.getByRole("menuitem", { name: "部署成功" }).click();
            await page.getByRole("button", { name: /确\s*定/ }).click();
            await expect(page.getByRole("table")).toContainText("部署成功");
        });
    });
});

test.describe("应用管理-概览-跳转", () => {
    test.describe.configure({ timeout: totalTimeout });
    let apps: DeployPlan[] = [];

    test.beforeAll(async ({ browser }) => {
        test.setTimeout(totalTimeout);
        apps = [];
        await deployHelmReleasesByApi(
            browser,
            Array.from({ length: 12 }, () => appPlan()),
            apps,
        );
    });
    test.afterAll(async ({ browser }) => {
        test.setTimeout(totalTimeout);
        await cleanupHelmReleasesByApi(browser, apps);
        apps = [];
    });

    test("【应用运维-应用管理概览-010】应用管理-概览-页面跳转-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await page.goto(serviceClusterUrl + "/container_platform/applicationManageHelm");
        await page.getByRole("textbox", { name: /搜索.*名称/ }).fill(APP_PREFIX);
        await page.getByRole("button", { name: "search" }).click();
        await expect(page.getByRole("table")).toBeVisible();

        const extractFirstRowAppName = async (): Promise<string> => {
            const rowText = (await page.locator("table tbody tr").first().textContent()) ?? "";
            const m = rowText.match(new RegExp(`${APP_PREFIX}-[0-9a-f]+-\\d+`));
            return m?.[0] ?? rowText.trim();
        };

        await expect(page.locator('.container-platform-pagination-total-text')).toContainText(/共\s*12\s*条/, { timeout: 60_000 });

        const clickPageIndex = async (pagination: Locator, index: number) => {
            const pageIndexBtn = pagination.locator('.container-platform-pagination-item');
            await pageIndexBtn.nth(index - 1).click();
            const activePageIndexBtn = pagination.locator('.container-platform-pagination-item-active');
            const pageRegex = new RegExp(`^${index}$`);
            await expect(activePageIndexBtn).toHaveText(pageRegex);
        };
        const togglePageSize = async (pagination: Locator, size: 10 | 20 | 50) => {
            const pageSizeSelector = pagination.locator('.container-platform-select-selector');
            await pageSizeSelector.click();
            await pagination.locator('.container-platform-select-item-option').getByText(`${size} 条/页`).click();
            await expect(pageSizeSelector).toContainText(`${size} 条/页`);
        };

        const pagination = page.locator('.container-platform-pagination');
        await togglePageSize(pagination, 10);
        const firstRowPage1 = await extractFirstRowAppName();
        await clickPageIndex(pagination, 2); // 点击第2页
        await expect.poll(async () => extractFirstRowAppName(), { timeout: 10_000 }).not.toBe(firstRowPage1);
        const firstRowPage2 = await extractFirstRowAppName();
        expect(firstRowPage2).not.toBe(firstRowPage1);
        await togglePageSize(pagination, 20);
    });
});
