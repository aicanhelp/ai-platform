import { expect, type Locator, type Page } from "@playwright/test";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { TEST_FUYAOURL } from "../consts";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export const applicationListUrl = "/container_platform/appMarket/marketCategory";
export const repositoryConfigUrl = "/container_platform/appMarket/stash/wareHouse";
export const packageManagementUrl = "/container_platform/appMarket/stash/applicationPackage";

export const coreRepositoryName = "core";
export const coreRepositoryUrl = "https://helm.openfuyao.cn/_core";
export const chartsUploadUrl = "/clusters/host/rest/marketplace/v1beta1/helm-charts";
export const chartsListUrl = "/clusters/host/rest/marketplace/v1beta1/helm-charts";

export const generatedPackagesDir = path.resolve(__dirname, "../assets/charts/generated-packages");
export const extensionPackagesDir = path.resolve(__dirname, "../assets/charts/extension-components");
const generatedManifestPath = path.join(generatedPackagesDir, "manifest.json");

export type PackageFixture = {
    filePath: string,
    fileName: string,
    packageName: string,
    version: string,
};

const generatedManifest = JSON.parse(fs.readFileSync(generatedManifestPath, "utf8")) as {
    packages: Array<{ name: string, version: string, packageFile: string }>
};

const generatedPackageFixtures: PackageFixture[] = generatedManifest.packages.map((item) => ({
    filePath: path.join(generatedPackagesDir, item.packageFile),
    fileName: item.packageFile,
    packageName: item.name,
    version: item.version,
}));

const extensionPackageFixtures: PackageFixture[] = fs.readdirSync(extensionPackagesDir)
    .filter((fileName) => fileName.endsWith(".tgz"))
    .map((fileName) => {
        const trimmedName = fileName.slice(0, -4);
        const versionSplitIndex = trimmedName.search(/-\d/);
        const packageName = versionSplitIndex > 0 ? trimmedName.slice(0, versionSplitIndex) : trimmedName;
        const version = versionSplitIndex > 0 ? trimmedName.slice(versionSplitIndex + 1) : "latest";
        return {
            filePath: path.join(extensionPackagesDir, fileName),
            fileName,
            packageName,
            version,
        };
    });

export const packageFixtures: PackageFixture[] = [...generatedPackageFixtures, ...extensionPackageFixtures];
export const fixturePackageNames = [...new Set(packageFixtures.map((item) => item.packageName))];

export function getLocalDateText(date: Date) {
    return new Intl.DateTimeFormat("en-CA", {
        timeZone: "Asia/Shanghai",
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
    }).format(date);
}

export function normalizeApplicationTitles(titles: string[]) {
    return titles
        .map((title) => title.trim())
        .filter((title) => title && !title.includes("暂无数据"));
}

export async function waitForRealApplicationTitles(applicationTitles: Locator) {
    await expect.poll(async () => {
        return normalizeApplicationTitles(await applicationTitles.allTextContents()).length;
    }, {
        timeout: 30000,
    }).toBeGreaterThan(0);

    return normalizeApplicationTitles(await applicationTitles.allTextContents());
}

export async function waitForApplicationListReady(page: Page) {
    const searchInput = page.getByPlaceholder("搜索应用名称");
    const applicationTitles = page.locator(".application_list .application_item .application_item_title");

    await expect(page).toHaveURL(/\/container_platform\/appMarket\/marketCategory$/);
    await expect(searchInput).toBeVisible({ timeout: 30000 });
    await expect.poll(async () => {
        return normalizeApplicationTitles(await applicationTitles.allTextContents()).length;
    }, {
        timeout: 30000,
    }).toBeGreaterThan(0);
}

export async function openApplicationListPageByUrl(page: Page) {
    await page.goto(`${TEST_FUYAOURL}${applicationListUrl}`);
    await page.waitForURL(new RegExp(`${applicationListUrl}$`));
    await page.waitForLoadState("networkidle");
}

export async function openRepositoryConfigPageByUrl(page: Page) {
    await page.goto(`${TEST_FUYAOURL}${repositoryConfigUrl}`);
    await page.waitForURL(new RegExp(`${repositoryConfigUrl}$`));
    await page.waitForLoadState("networkidle");
}

export async function openPackageManagementPageByUrl(page: Page) {
    await page.goto(`${TEST_FUYAOURL}${packageManagementUrl}`);
    await page.waitForURL(new RegExp(`${packageManagementUrl}$`));
    await page.waitForLoadState("networkidle");
}

export async function ensureFirstCheckboxChecked(dialog: Locator) {
    const checkbox = dialog.locator('input[type="checkbox"]').first();
    await expect(checkbox).toBeVisible({ timeout: 30000 });
    if (!await checkbox.isChecked()) {
        await dialog.locator("label").first().click();
        await expect(checkbox).toBeChecked({ timeout: 30000 });
    }
}

export async function syncAllRepositories(page: Page) {
    const repositoryRows = page.locator("tbody tr");
    const repositoryCount = await repositoryRows.count();

    for (let index = 0; index < repositoryCount; index++) {
        const currentRow = repositoryRows.nth(index);
        const repositoryName = ((await currentRow.locator("td").first().textContent()) || "").trim();
        const syncSuccessText = `${repositoryName}仓库同步成功！`;
        await currentRow.getByRole("img", { name: "more" }).click();
        await page.getByText("同步仓库", { exact: true }).last().click();
        await expect(page.getByText(syncSuccessText)).toBeVisible({ timeout: 30000 });
        await page.locator("body").click({
            position: { x: 10, y: 10 },
        });
        await page.waitForTimeout(300);
    }
}

export async function syncLocalRepository(page: Page) {
    await openRepositoryConfigPageByUrl(page);
    const localRepositoryRow = page.locator("tbody tr")
        .filter({ hasText: "local" })
        .filter({ hasText: "http://local-harbor." })
        .first();
    await expect(localRepositoryRow).toBeVisible({ timeout: 30000 });
    await localRepositoryRow.getByRole("img", { name: "more" }).click();
    await page.getByText("同步仓库", { exact: true }).last().click();
    await expect(page.getByText(/local仓库同步成功[！!]/).last()).toBeVisible({ timeout: 60000 });
}

export async function syncCurrentRepositoryFromPackageManagement(page: Page) {
    const syncButton = page.locator("button").filter({ hasText: "同 步" }).first();
    await expect(syncButton).toBeEnabled({ timeout: 30000 });
    await syncButton.click();
    await expect(page.getByText(/仓库同步成功！/).last()).toBeVisible({ timeout: 30000 });
}

export async function ensureCoreRepositoryExists(page: Page) {
    const coreRepositoryRow = page.locator("tbody tr")
        .filter({ hasText: coreRepositoryName })
        .filter({ hasText: coreRepositoryUrl });

    if (await coreRepositoryRow.count()) {
        await expect(coreRepositoryRow.first()).toBeVisible();
        return;
    }

    await page.getByRole("button", { name: "添加仓库" }).click();
    const addRepositoryDialog = page.getByRole("dialog").filter({ hasText: "添加仓库" }).last();
    await expect(addRepositoryDialog).toBeVisible();
    await addRepositoryDialog.locator("#form_in_modal_name").fill(coreRepositoryName);
    await addRepositoryDialog.locator("#form_in_modal_url").fill(coreRepositoryUrl);
    await addRepositoryDialog.locator("label").filter({ hasText: "NONE" }).click();

    const securityRiskDialog = page.getByRole("dialog").filter({ hasText: "安全风险" }).last();
    await expect(securityRiskDialog).toBeVisible({ timeout: 30000 });
    await securityRiskDialog.locator("label").first().click();
    await securityRiskDialog.locator("button").filter({ hasText: /确\s*定/ }).click();

    await addRepositoryDialog.locator("button").filter({ hasText: /确\s*定/ }).click();

    const certificateRiskDialog = page.getByRole("dialog").filter({ hasText: "证书配置提示" }).last();
    await expect(certificateRiskDialog).toBeVisible({ timeout: 30000 });
    await ensureFirstCheckboxChecked(certificateRiskDialog);
    await certificateRiskDialog.locator("button").filter({ hasText: "继 续" }).click();

    await page.waitForLoadState("networkidle");
    await expect(coreRepositoryRow.first()).toBeVisible({ timeout: 30000 });
}

export async function deleteCoreRepositoryIfExists(page: Page) {
    await openRepositoryConfigPageByUrl(page);

    const coreRepositoryRow = page.locator("tbody tr")
        .filter({ hasText: coreRepositoryName })
        .filter({ hasText: coreRepositoryUrl });

    if (!await coreRepositoryRow.count()) {
        return;
    }

    await coreRepositoryRow.first().getByRole("img", { name: "more" }).click();
    await page.getByText("删除", { exact: true }).click();

    const deleteDialog = page.getByText(`确定删除仓库 ${coreRepositoryName} 吗？`);
    await expect(deleteDialog).toBeVisible({ timeout: 30000 });
    await page.locator("label").filter({ hasText: "已了解删除操作无法恢复" }).click();
    await page.getByRole("button", { name: "删 除" }).click();

    await page.waitForLoadState("networkidle");
    await expect(coreRepositoryRow).toHaveCount(0, { timeout: 30000 });
}

export async function listAllLocalCharts(page: Page) {
    const items: Array<{
        metadata: {
            name: string,
            version: string,
            description?: string,
        }
    }> = [];
    let currentPage = 1;
    let totalPage = 1;

    do {
        const response = await page.request.get(
            `${TEST_FUYAOURL}${chartsListUrl}?repo=local&page=${currentPage}&limit=100`,
        );
        expect(response.ok()).toBeTruthy();
        const responseJson = await response.json() as {
            data?: {
                items?: Array<{
                    metadata: {
                        name: string,
                        version: string,
                        description?: string,
                    }
                }>,
                totalPage?: number,
            }
        };
        const responseItems = responseJson.data?.items || [];
        totalPage = responseJson.data?.totalPage || 1;
        items.push(...responseItems);
        currentPage += 1;
    } while (currentPage <= totalPage);

    return items;
}

export async function uploadPackageFixtureOnPage(targetPage: Page, fixture: PackageFixture) {
    await targetPage.getByRole("button", { name: "添加包" }).click();
    const addPackageDialog = targetPage.getByRole("dialog").filter({ hasText: "添加包" }).last();
    await expect(addPackageDialog).toBeVisible({ timeout: 30000 });
    const fileInput = addPackageDialog.locator('input[type="file"]');
    await expect(fileInput).toHaveCount(1, { timeout: 30000 });
    await expect(fileInput).toBeAttached({ timeout: 30000 });
    await fileInput.setInputFiles(fixture.filePath);
    await expect(addPackageDialog.getByText(fixture.fileName)).toBeVisible({ timeout: 30000 });
    await addPackageDialog.locator("button").filter({ hasText: /确\s*定/ }).click();
}

export async function ensurePackageFixtureUploaded(page: Page, fixture: PackageFixture) {
    const existingCharts = await listAllLocalCharts(page);
    const exists = existingCharts.some((item) => (
        item.metadata.name === fixture.packageName && item.metadata.version === fixture.version
    ));

    if (exists) {
        return;
    }

    await uploadPackageFixtureOnPage(page, fixture);

    await expect.poll(async () => {
        const charts = await listAllLocalCharts(page);
        return charts.some((item) => (
            item.metadata.name === fixture.packageName && item.metadata.version === fixture.version
        ));
    }, {
        timeout: 60000,
        intervals: [1000, 2000, 3000],
    }).toBe(true);
}

export async function ensurePackageFixturesUploaded(page: Page) {
    await openPackageManagementPageByUrl(page);

    for (const fixture of packageFixtures) {
        await ensurePackageFixtureUploaded(page, fixture);
    }
}

export async function deletePackageFixtureByName(page: Page, packageName: string) {
    const charts = await listAllLocalCharts(page);
    if (!charts.some((item) => item.metadata.name === packageName)) {
        return;
    }

    const deleteResponse = await page.request.delete(`${TEST_FUYAOURL}${chartsUploadUrl}/${packageName}`);
    expect(deleteResponse.ok()).toBeTruthy();

    await expect.poll(async () => {
        const remainingCharts = await listAllLocalCharts(page);
        return remainingCharts.some((item) => item.metadata.name === packageName);
    }, {
        timeout: 60000,
        intervals: [1000, 2000, 3000],
    }).toBe(false);
}

export async function deletePackageFixturesIfExist(page: Page) {
    for (const packageName of fixturePackageNames) {
        await deletePackageFixtureByName(page, packageName);
    }
}

// 对应在应用市场安装组件（默认配置）
async function installComponent(page: Page, componentName: string, appName: string) {
    await page.getByRole("link", { name: "应用列表" }).click();
    await page.getByRole("textbox", { name: "搜索应用名称" }).fill(componentName);
    await page.getByRole("button", { name: "search" }).click();
    await page.locator("#root_container").getByText(componentName).click();
    await page.getByRole("button", { name: "部 署" }).click();
    await page.getByRole("textbox", { name: "* 应用名称" }).fill(appName);
    await page.getByRole("button", { name: /确\s*定/ }).click();
}

// 对应在扩展组件管理页面检查组件部署成功
async function checkComponent(page: Page, appName: string) {
    await expect(page.getByText("部署成功，即将跳转扩展组件管理页面")).toBeVisible();
    await expect(page.getByRole("navigation").getByText("扩展组件管理")).toBeVisible();
    await page.getByRole("textbox", { name: "搜索Extend名称" }).fill(appName);
    await page.getByRole("button", { name: "search" }).click();
    await expect(page.getByRole("cell", { name: "check-circle 部署成功" })).toBeVisible();
}

// 对应将yaml文件中的单一的 oldStr 修改为 newStr（修改整行）
async function changeString(page: Page, oldStr: string, newStr: string) {
    await page.getByText(oldStr).click();
    await page.keyboard.press("Shift+Home");
    await page.keyboard.press("Backspace");
    await page.keyboard.type(newStr);
    await expect(page.getByText(newStr)).toBeVisible();
}

// 对应在日志界面进行筛选操作
async function selectLog(page: Page, defaultContainer: string, selectContainer: string) {
    await page.getByRole("tab", { name: "日志" }).click();
    await page.getByText(defaultContainer + "-").click();
    await page.getByTitle(selectContainer + "-").locator("div").click();
    await page.getByRole("button", { name: /查\s*询/ }).click();
    await page.waitForTimeout(1000);
}

// 对应在扩展组件管理中卸载组件
async function uninstallComponent(page: Page, appName: string) {
    await page.getByRole("textbox", { name: "搜索Extend名称" }).fill(appName);
    await page.getByRole("img", { name: "more" }).locator("svg").click();
    await page.getByRole("button", { name: "卸载" }).click();
    await expect(page.getByText("卸载扩展组件", { exact: true })).toBeVisible();
    await page.getByRole("checkbox", { name: "已了解卸载操作无法恢复" }).check();
    await page.getByRole("button", { name: "卸 载" }).click();
    await expect(page.getByText("扩展组件" + appName + "卸载成功")).toBeVisible();
    await expect(page.getByRole("cell", { name: "Simple Empty 暂无数据" })).toBeVisible();
}

// 对应在扩展组件详情页面进行升级操作
async function upgradeComponent(page: Page, initVersion: string, upgradeVersion: string) {
    await page.getByRole("list").getByText("详情").click();
    await page.getByRole("button", { name: "操作 down" }).click();
    await page.getByRole("button", { name: "升级" }).click();
    await page.getByTitle(initVersion).click();
    await page.getByTitle(upgradeVersion).locator("div").click();
}

// 对应在扩展组件详情页面进行回退操作(在升级成功界面后操作)
async function revertComponent(page: Page, revertName: string) {
    await page.getByRole("link", { name: "详情" }).click();
    await page.getByRole("button", { name: "操作 down" }).click();
    await page.getByRole("button", { name: "回退" }).click();
    await expect(page.getByText("应用回退")).toBeVisible();
    await page.getByRole("row", { name: revertName }).getByRole("cell").first().click();
    await page.getByRole("button", { name: /确\s*定/ }).click();
}

// 对应检查pod (检查指定前缀 resource 为 state 的个数)
async function checkPrefixResourceNum(page: Page, resource: string, prefix: string, state: string, stateNum: number) {
    await page.getByText("工作负载").click();
    await page.getByRole("link", { name: resource }).click();
    await page.getByRole("textbox", { name: "搜索Pod名称" }).fill(prefix);
    await page.getByRole("button", { name: "search" }).click();
    await expect(page.getByText(state)).toHaveCount(stateNum);
}

export {
    installComponent,
    checkComponent,
    changeString,
    uninstallComponent,
    selectLog,
    upgradeComponent,
    revertComponent,
    checkPrefixResourceNum,
};
