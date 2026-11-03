import { expect, test } from "@playwright/test";
import {
    changePullPolicyLineWithWait,
    openUpgradeEditor,
    selectUpgradeChartVersion,
    upgradeByChartVersion,
    revertComponent,
    checkPrefixResourceNum,
    gotoManagedAppDetail,
    deployComponents,
    cleanupComponents,
} from "./utils/application";
import {
    totalTimeout,
    upgradeTime,
    revertTime,
    componentNameApp,
    componentNameExt,
    chartResourceName,
    packageVersion,
    packageVersionNext,
    appPlan,
    extPlan,
    type DeployPlan,
} from "./utils/constants";

test.describe("应用管理-升级回退", () => {
    let apps: DeployPlan[] = [];

    test.beforeEach(async ({ browser }) => {
        apps = await deployComponents(browser, [appPlan(undefined, "app")]);
    });
    test.afterEach(async ({ browser }) => {
        await cleanupComponents(browser, apps);
        apps = [];
    });

    test("【应用运维-应用升级回退-005】应用管理-升级-详情页面-yaml导出查找复制-取消重置-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await gotoManagedAppDetail(page, apps[0]);
        await openUpgradeEditor(page);
        await test.step("导出、复制、变化校验、重置", async () => {
            const downloadPromise = page.waitForEvent("download", { timeout: 60_000 });
            await page.getByRole("img", { name: "export" }).locator("svg").click();
            const download = await downloadPromise;
            expect(download.suggestedFilename().toLowerCase()).toContain(".yaml");
            await page.getByRole("img", { name: "copy" }).locator("svg").click();
            await expect(page.getByText("复制成功")).toBeVisible();
            await page.getByRole("tab", { name: "Values.yaml" }).click();
            const { oldLine, newLine } = await changePullPolicyLineWithWait(page);
            await page.getByRole("tab", { name: "变化" }).click();
            await expect(page.getByRole("cell", { name: "- " + oldLine }).first()).toBeVisible({ timeout: 60_000 });
            await expect(page.getByRole("cell", { name: "+ " + newLine }).first()).toBeVisible();
            await page.getByRole("tab", { name: "Values.yaml" }).click();
            await page.getByRole("button", { name: "重 置" }).click();
            await expect(page.getByText(oldLine).first()).toBeVisible();
            await expect(page.getByText(newLine).first()).toHaveCount(0);
        });
        await test.step("取消并返回上一页", async () => {
            await page.getByRole("button", { name: "取 消" }).click();
            await expect(page.getByText("导出")).toHaveCount(0);
        });
    });

    test("【应用运维-应用升级回退-004】应用管理-升级-详情页面-yaml修改-变化-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        let newPullPolicyLine = "";
        await gotoManagedAppDetail(page, apps[0]);
        await openUpgradeEditor(page);
        await test.step("修改版本与 yaml 并升级", async () => {
            await selectUpgradeChartVersion(page, packageVersion, packageVersionNext);
            const { oldLine, newLine } = await changePullPolicyLineWithWait(page);
            newPullPolicyLine = newLine;
            await page.getByRole("tab", { name: "变化" }).click();
            await expect(page.getByRole("cell", { name: "- " + oldLine }).first()).toBeVisible({ timeout: 60_000 });
            await expect(page.getByRole("cell", { name: "+ " + newLine }).first()).toBeVisible();
            await page.getByRole("button", { name: /确\s*定/ }).click();
            await expect(page.getByText("升级成功")).toBeVisible({ timeout: upgradeTime });
            await expect(page.getByText("模板版本：" + packageVersionNext)).toBeVisible({ timeout: 60_000 });
        });
        await test.step("检查 yaml 是否修改", async () => {
            await expect(page.getByText("基本信息")).toBeVisible({ timeout: 60_000 });
            await page.getByRole("tab", { name: "YAML" }).click();
            await expect(page.getByText(/---#\s*Source:/)).toBeVisible({ timeout: 120_000 });
            await page.getByRole("checkbox", { name: "只显示Values" }).check({ timeout: 60_000 });
            await expect(page.getByText(newPullPolicyLine).first()).toBeVisible({ timeout: 60_000 });
        });
    });

    test("【应用运维-应用升级回退-022】应用管理-升级回退-升级回退", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await gotoManagedAppDetail(page, apps[0]);
        await test.step("升级和回退操作", async () => {
            await upgradeByChartVersion(page, packageVersion, packageVersionNext);
            const revertName = `${componentNameApp} 1 ${componentNameApp}`;
            await revertComponent(page, revertName);
            await expect(page.getByText("模板版本：" + packageVersion)).toBeVisible({ timeout: revertTime });
            await upgradeByChartVersion(page, packageVersion, packageVersionNext);
            await revertComponent(page, revertName);
            await expect(page.getByText("模板版本：" + packageVersion)).toBeVisible({ timeout: revertTime });
        });
        await test.step("检查Pod正常", async () => {
            const name = chartResourceName(apps[0].appName, componentNameApp);
            await checkPrefixResourceNum(page, "Pod", name, "Running", 1);
        });
    });
});

test.describe("扩展管理-升级回退", () => {
    let apps: DeployPlan[] = [];

    test.beforeEach(async ({ browser }) => {
        apps = await deployComponents(browser, [extPlan(undefined, "ext")]);
    });
    test.afterEach(async ({ browser }) => {
        await cleanupComponents(browser, apps);
        apps = [];
    });

    test("【应用运维-扩展升级回退-005】扩展管理-升级-详情页面-yaml导出查找复制-取消重置-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await gotoManagedAppDetail(page, apps[0]);
        await openUpgradeEditor(page);
        await test.step("导出、复制、重置并校验", async () => {
            const downloadPromise = page.waitForEvent("download", { timeout: 60_000 });
            await page.getByRole("img", { name: "export" }).locator("svg").click();
            const download = await downloadPromise;
            expect(download.suggestedFilename().toLowerCase()).toContain(".yaml");
            await page.getByRole("img", { name: "copy" }).locator("svg").click();
            await expect(page.getByText("复制成功")).toBeVisible({ timeout: 60_000 });
            const { oldLine, newLine } = await changePullPolicyLineWithWait(page);
            await page.getByRole("button", { name: "重 置" }).click();
            await expect(page.getByText(oldLine).first()).toBeVisible();
            await expect(page.getByText(newLine).first()).toHaveCount(0);
        });
        await test.step("取消并返回上一页", async () => {
            await page.getByRole("button", { name: "取 消" }).click();
            await expect(page.getByText("导出")).toHaveCount(0);
        });
    });

    test("【应用运维-扩展升级回退-004】扩展管理-升级-详情页面-yaml修改-变化-基础功能验证", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await gotoManagedAppDetail(page, apps[0]);
        await openUpgradeEditor(page);
        await test.step("修改版本与 yaml 并升级", async () => {
            await selectUpgradeChartVersion(page, packageVersion, packageVersionNext);
            const { oldLine, newLine } = await changePullPolicyLineWithWait(page);
            await page.getByRole("tab", { name: "变化" }).click();
            await expect(page.getByRole("cell", { name: "- " + oldLine }).first()).toBeVisible();
            await expect(page.getByRole("cell", { name: "+ " + newLine }).first()).toBeVisible();
            await page.getByRole("button", { name: /确\s*定/ }).click();
            await expect(page.getByText("升级成功")).toBeVisible({ timeout: upgradeTime });
            await expect(page.getByText("模板版本：" + packageVersionNext)).toBeVisible({ timeout: 60_000 });
        });
        await test.step("检查 yaml 是否修改", async () => {
            await expect(page.getByText("基本信息")).toBeVisible({ timeout: 60_000 });
            await page.getByRole("tab", { name: "YAML" }).click();
            await expect(page.getByText(/---#\s*Source:/)).toBeVisible({ timeout: 120_000 });
            await page.getByRole("checkbox", { name: "只显示Values" }).check({ timeout: 60_000 });
            await expect(page.getByText("pullPolicy: Always").first()).toBeVisible({ timeout: 60_000 });
        });
    });

    test("【应用运维-扩展升级回退-022】扩展管理-升级回退-升级回退", { tag: ["@v24.06", "@application"] }, async ({ page }) => {
        await gotoManagedAppDetail(page, apps[0]);
        await test.step("升级和回退操作", async () => {
            await upgradeByChartVersion(page, packageVersion, packageVersionNext);
            const revertName = `${componentNameExt} 1 ${componentNameExt}`;
            await revertComponent(page, revertName);
            await expect(page.getByText("模板版本：" + packageVersion)).toBeVisible({ timeout: revertTime });
            await upgradeByChartVersion(page, packageVersion, packageVersionNext);
            await revertComponent(page, revertName);
            await expect(page.getByText("模板版本：" + packageVersion)).toBeVisible({ timeout: revertTime });
        });
        await test.step("检查Pod正常", async () => {
            const name = chartResourceName(apps[0].appName, componentNameExt);
            await checkPrefixResourceNum(page, "Pod", name, "Running", 1);
        });
    });
});
