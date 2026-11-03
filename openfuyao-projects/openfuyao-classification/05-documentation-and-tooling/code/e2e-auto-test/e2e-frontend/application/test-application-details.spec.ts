import { expect, test, type Page } from "@playwright/test";
import {
    selectLog,
    waitForLogReady,
    withKubectlSsh,
    verifyUiLogWithKubectl,
    verifyLogExport,
    verifyResourceSortInResourceTab,
    gotoManagedAppDetail,
    deployComponents,
    cleanupComponents,
} from "./utils/application";
import {
    totalTimeout,
    chartSidecarContainer,
    chartResourceName,
    componentNameApp,
    componentNameExt,
    extSidecarContainer,
    appPlan,
    extPlan,
    applicationTestNamespace,
    type DeployPlan,
} from "./utils/constants";

const appResourceName = (app: DeployPlan) => chartResourceName(app.appName, componentNameApp);
const extResourceName = (app: DeployPlan) => chartResourceName(app.appName, componentNameExt);

async function verifyResourceInfoDisplay(page: Page) {
    const table = page.getByRole("table").first();
    await expect(table).toBeVisible();
    const empty = page.getByText(/暂无(内容|数据)/).first();
    await expect(table.locator("tbody tr").first().or(empty)).toBeVisible();
}

async function filterEvents(
    page: Page,
    options: { resourceName: string; resourceType: string; level: string; namespace: string },
) {
    const selects = page.locator(".container-platform-select-selector");
    await selects.nth(1).click();
    await page.keyboard.type(options.resourceName);
    await page.getByTitle(options.resourceName, { exact: true }).locator("div").click();
    await selects.nth(2).click();
    await page.getByTitle(options.resourceType, { exact: true }).locator("div").click();
    await selects.nth(3).click();
    await page.getByTitle(options.level, { exact: true }).locator("div").click();
    await selects.nth(4).click();
    await page.getByTitle(options.namespace, { exact: true }).locator("div").click();
    await page.getByRole("button", { name: /查\s*询/ }).click();
}

test.describe("应用管理-详情", () => {
    test.describe.configure({ timeout: totalTimeout });
    test.use({ acceptDownloads: true });
    let createdApps: DeployPlan[] = [];

    test.beforeAll(async ({ browser }) => {
        createdApps = [];
        await deployComponents(browser, [appPlan()], createdApps);
    });
    test.afterAll(async ({ browser }) => {
        await cleanupComponents(browser, createdApps);
        createdApps = [];
    });

    test.beforeEach(async ({ page }) => {
        await gotoManagedAppDetail(page, createdApps[0]);
    });

    test("【应用运维-应用详情-017】应用管理-详情-日志标签-筛选-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        const instancePrefix = appResourceName(createdApps[0]);
        await test.step("筛选主容器：应有日志", async () => {
            await selectLog(page, instancePrefix, componentNameApp);
            await expect(page.getByText("暂无日志")).toBeHidden({ timeout: 60_000 });
        });
        await test.step("筛选 sidecar：应显示暂无日志", async () => {
            await selectLog(page, instancePrefix, chartSidecarContainer);
            await expect(page.getByText("暂无日志")).toBeVisible({ timeout: 60_000 });
        });
        await test.step("重置筛选", async () => {
            await page.getByRole("button", { name: "重 置" }).click();
        });
    });

    test("【应用运维-应用详情-020】应用管理-详情-事件标签-筛选-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await test.step("进入事件标签页并筛选", async () => {
            await page.getByRole("tab", { name: "事件" }).click();
            await filterEvents(page, {
                resourceName: appResourceName(createdApps[0]),
                resourceType: "Deployment",
                level: "正常",
                namespace: applicationTestNamespace,
            });
        });
        await test.step("验证筛选结果并重置", async () => {
            await page.getByRole("button", { name: "重 置" }).click();
            await expect(page.locator(".container-platform-select-selector").nth(3)).toHaveText("全部");
        });
    });

    test("【应用运维-应用详情-008】应用管理-概览-排序-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await test.step("进入应用管理-Helm详情页面-资源标签页", async () => {
            await page.getByRole("tab", { name: "资源" }).click();
            await expect(page.getByRole("table")).toBeVisible();
        });
        await test.step("校验资源信息展示正确（创建时间）", async () => {
            await verifyResourceInfoDisplay(page);
        });
    });

    test("【应用运维-应用详情-014】应用管理-详情-资源标签-排序-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await test.step("进入应用管理-Helm详情页面-资源标签页", async () => {
            await page.getByRole("tab", { name: "资源" }).click();
            await expect(page.getByRole("table")).toBeVisible();
        });
        await verifyResourceSortInResourceTab(page);
    });

    test("【应用运维-应用详情-015】应用管理-详情-资源标签-筛选-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await test.step("进入应用管理-Helm详情页面-资源标签页", async () => {
            await page.getByRole("tab", { name: "资源" }).click();
            await expect(page.getByRole("table")).toBeVisible();
        });
        await test.step("按资源名称搜索并校验结果", async () => {
            await page.getByRole("textbox", { name: "资源名称" }).fill(appResourceName(createdApps[0]));
            await page.getByRole("button", { name: "search" }).click();
            await expect(page.getByRole("table")).toContainText(appResourceName(createdApps[0]));
        });
        await test.step("按类型与状态筛选并校验", async () => {
            await page.getByLabel("类型").getByRole("button", { name: "filter" }).click();
            await page.getByRole("menuitem", { name: "Deployment", exact: true }).click();
            await page.getByRole("button", { name: /确\s*定/ }).click();
            await expect(page.getByRole("table")).toContainText("Deployment");
            await page.getByLabel("状态").getByRole("button", { name: "filter" }).click();
            await page.getByText("运行中失败创建中删除中").click();
            await page.getByRole("menuitem", { name: "运行中", exact: true }).click();
            await page.getByRole("button", { name: /确\s*定/ }).click();
            await expect(page.getByRole("table")).toContainText("运行中");
        });
    });

    test("【应用运维-应用详情-016】应用管理-详情-日志标签-信息展示-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await test.step("对比 kubectl logs 与 UI 日志内容", async () => {
            await selectLog(page, appResourceName(createdApps[0]), componentNameApp);
            await waitForLogReady(page);
            await withKubectlSsh(async (ssh) => {
                await verifyUiLogWithKubectl(page, ssh, {
                    kubectlNamespace: applicationTestNamespace,
                    kubectlPodNamePrefix: appResourceName(createdApps[0]),
                    kubectlContainerName: componentNameApp,
                });
            });
        });
    });

    test("【应用运维-应用详情-019】应用管理-详情-事件标签-信息展示-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await test.step("进入应用管理-Helm详情页面-事件标签页", async () => {
            await page.getByRole("tab", { name: "事件" }).click();
            await filterEvents(page, {
                resourceName: appResourceName(createdApps[0]),
                resourceType: "Deployment",
                level: "正常",
                namespace: applicationTestNamespace,
            });
        });
        await test.step("查询后确认展示时间线", async () => {
            const eventTabPanel = page.getByRole("tabpanel", { name: "事件" });
            const timelineFirstItem = eventTabPanel
                .locator(".helm_event_timeLine .container-platform-timeline-item, .event_timeLine li")
                .first();
            await expect(timelineFirstItem).toBeVisible({ timeout: 30_000 });
        });
    });

    test("【应用运维-应用详情-018】应用管理-详情-日志标签-导出-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await test.step("进入应用管理-Helm详情页面-日志标签页并筛选主容器", async () => {
            await selectLog(page, appResourceName(createdApps[0]), componentNameApp);
        });
        await test.step("等待日志就绪并导出校验下载文件", async () => {
            await verifyLogExport(page, test.info().outputDir);
        });
    });
});

test.describe("扩展管理-详情", () => {
    test.describe.configure({ timeout: totalTimeout });
    test.use({ acceptDownloads: true });
    let createdExts: DeployPlan[] = [];

    test.beforeAll(async ({ browser }) => {
        createdExts = [];
        await deployComponents(browser, [extPlan()], createdExts);
    });

    test.afterAll(async ({ browser }) => {
        await cleanupComponents(browser, createdExts);
        createdExts = [];
    });

    test.beforeEach(async ({ page }) => {
        await gotoManagedAppDetail(page, createdExts[0]);
    });

    test("【应用运维-扩展详情-017】扩展管理-详情-日志标签-筛选-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        const instancePrefix = extResourceName(createdExts[0]);
        await test.step("筛选主容器：应有日志", async () => {
            await selectLog(page, instancePrefix, componentNameExt);
            await expect(page.getByText("暂无日志")).toBeHidden({ timeout: 60_000 });
        });
        await test.step("筛选 sidecar：应显示暂无日志", async () => {
            await selectLog(page, instancePrefix, extSidecarContainer);
            await expect(page.getByText("暂无日志")).toBeVisible({ timeout: 60_000 });
        });
        await test.step("重置筛选", async () => {
            await page.getByRole("button", { name: "重 置" }).click();
        });
    });

    test("【应用运维-扩展详情-020】扩展管理-详情-事件标签-筛选-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await test.step("进入事件标签页并筛选", async () => {
            await page.getByRole("tab", { name: "事件" }).click();
            await filterEvents(page, {
                resourceName: extResourceName(createdExts[0]),
                resourceType: "Deployment",
                level: "正常",
                namespace: applicationTestNamespace,
            });
        });
        await test.step("验证筛选成功并重置", async () => {
            await page.getByRole("button", { name: "重 置" }).click();
            await expect(page.locator(".container-platform-select-selector").nth(3)).toHaveText("全部");
        });
    });

    test("【应用运维-扩展详情-008】扩展管理-概览-排序-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await test.step("进入扩展组件管理-详情页面-资源标签页", async () => {
            await page.getByRole("tab", { name: "资源" }).click();
            await expect(page.getByRole("table")).toBeVisible();
        });
        await test.step("校验资源信息展示正确（创建时间）", async () => {
            await verifyResourceInfoDisplay(page);
        });
    });

    test("【应用运维-扩展详情-014】扩展管理-详情-资源标签-排序-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await test.step("进入扩展组件管理-详情页面-资源标签页", async () => {
            await page.getByRole("tab", { name: "资源" }).click();
            await expect(page.getByRole("table")).toBeVisible();
        });
        await verifyResourceSortInResourceTab(page);
    });

    test("【应用运维-扩展详情-015】扩展管理-详情-资源标签-筛选-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await test.step("进入扩展组件管理-详情页面-资源标签页", async () => {
            await page.getByRole("tab", { name: "资源" }).click();
            await expect(page.getByRole("table")).toBeVisible();
        });
        await test.step("按资源名称搜索并校验结果", async () => {
            await page.getByRole("textbox", { name: "资源名称" }).fill(extResourceName(createdExts[0]));
            await page.getByRole("button", { name: "search" }).click();
            await expect(page.getByRole("table")).toContainText(extResourceName(createdExts[0]));
        });
        await test.step("按类型与状态筛选并校验", async () => {
            await page.getByLabel("类型").getByRole("button", { name: "filter" }).click();
            await page.getByRole("menuitem", { name: "Deployment", exact: true }).click();
            await page.getByRole("button", { name: /确\s*定/ }).click();
            await expect(page.getByRole("table")).toContainText("Deployment");
            await page.getByLabel("状态").getByRole("button", { name: "filter" }).click();
            await page.getByText("运行中失败创建中删除中").click();
            await page.getByRole("menuitem", { name: "运行中", exact: true }).click();
            await page.getByRole("button", { name: /确\s*定/ }).click();
            await expect(page.getByRole("table")).toContainText("运行中");
        });
    });

    test("【应用运维-扩展详情-016】扩展管理-详情-日志标签-信息展示-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await test.step("进入扩展组件管理-详情页面-日志标签页并筛选主容器", async () => {
            await selectLog(page, extResourceName(createdExts[0]), componentNameExt);
            await page.waitForTimeout(3000);
        });
        await test.step("对比 kubectl logs 与 UI 日志内容", async () => {
            await withKubectlSsh(async (ssh) => {
                await verifyUiLogWithKubectl(page, ssh, {
                    kubectlNamespace: applicationTestNamespace,
                    kubectlPodNamePrefix: extResourceName(createdExts[0]),
                    kubectlContainerName: componentNameExt,
                });
            });
        });
    });

    test("【应用运维-扩展详情-024】扩展管理-详情-详情标签-扩展界面编辑-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await test.step("进入详情标签页", async () => {
            await page.getByRole("tab", { name: "详情" }).click();
            await expect(page.getByText("扩展界面启停")).toBeVisible();
        });
        await test.step("编辑扩展界面并保存", async () => {
            await page.getByRole("img", { name: "edit" }).locator("svg").click();
            await page.getByRole("button", { name: "保 存" }).click();
            const refreshHint = page.getByText("扩展组件发生更新，请刷新以展示最新版本的页面。", { exact: true }).first();
            await expect(refreshHint).toBeVisible({ timeout: 60_000 });
            await refreshHint.click();
            await page.locator("div").filter({ hasText: new RegExp(`^${componentNameExt}$`) }).first().click();
            const switchBtn = page.getByRole("switch");
            const beforeChecked = (await switchBtn.getAttribute("aria-checked")) ?? "";
            await page.getByRole("img", { name: "edit" }).locator("svg").click();
            await switchBtn.click();
            await page.getByRole("button", { name: "保 存" }).click();
            await expect(refreshHint).toBeVisible({ timeout: 60_000 });
            await refreshHint.click();
            await page.locator("div").filter({ hasText: new RegExp(`^${componentNameExt}$`) }).click();
            const afterChecked = (await switchBtn.getAttribute("aria-checked")) ?? "";
            await expect(afterChecked).not.toBe(beforeChecked);
        });
    });

    test("【应用运维-扩展详情-019】扩展管理-详情-事件标签-信息展示-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await test.step("进入扩展组件管理-详情页面-事件标签页", async () => {
            await page.getByRole("tab", { name: "事件" }).click();
            await expect(page.getByRole("tab", { name: "事件" })).toBeVisible();
            await filterEvents(page, {
                resourceName: extResourceName(createdExts[0]),
                resourceType: "Deployment",
                level: "正常",
                namespace: applicationTestNamespace,
            });
        });
        await test.step("查询后确认展示时间线", async () => {
            const eventTabPanel = page.getByRole("tabpanel", { name: "事件" });
            const timelineFirstItem = eventTabPanel
                .locator(".helm_event_timeLine .container-platform-timeline-item, .event_timeLine li")
                .first();
            await expect(timelineFirstItem).toBeVisible({ timeout: 30_000 });
        });
    });

    test("【应用运维-扩展详情-018】扩展管理-详情-日志标签-导出-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await test.step("进入扩展组件管理-详情页面-日志标签页并筛选主容器", async () => {
            await selectLog(page, extResourceName(createdExts[0]), componentNameExt);
        });
        await test.step("等待日志就绪并导出校验下载文件", async () => {
            await verifyLogExport(page, test.info().outputDir);
        });
    });
});
