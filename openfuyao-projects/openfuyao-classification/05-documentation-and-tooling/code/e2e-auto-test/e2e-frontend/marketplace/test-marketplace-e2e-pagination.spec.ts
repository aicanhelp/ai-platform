import { expect, test, type Locator, type Page } from "@playwright/test";
import path from "path";
import { fileURLToPath } from "url";
import { TEST_FUYAOURL } from "./consts";
import { newAuthorizedContext } from "@/utils/common";
import {
    type PackageFixture,
    deleteCoreRepositoryIfExists,
    deletePackageFixtureByName,
    deletePackageFixturesIfExist,
    ensureCoreRepositoryExists,
    ensureFirstCheckboxChecked,
    ensurePackageFixtureUploaded,
    ensurePackageFixturesUploaded,
    fixturePackageNames,
    generatedPackagesDir,
    getLocalDateText,
    listAllLocalCharts,
    normalizeApplicationTitles,
    openApplicationListPageByUrl,
    openPackageManagementPageByUrl,
    openRepositoryConfigPageByUrl,
    packageFixtures,
    repositoryConfigUrl,
    syncAllRepositories,
    syncCurrentRepositoryFromPackageManagement,
    uploadPackageFixtureOnPage,
    waitForRealApplicationTitles,
} from "./utils/marketplace";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

test.describe("应用市场（带分页）E2E", () => {
    let packageFixtureUploadDateText: string;
    test.beforeAll(async ({ browser }) => {
        const context = await newAuthorizedContext(browser);
        const page = await context.newPage();
        packageFixtureUploadDateText = getLocalDateText(new Date());

        try {
            await openRepositoryConfigPageByUrl(page);
            await ensureCoreRepositoryExists(page);
            await syncAllRepositories(page);
            await ensurePackageFixturesUploaded(page);

            await openPackageManagementPageByUrl(page);
            const charts = await listAllLocalCharts(page);
            const packageNames = new Set(charts.map((item) => item.metadata.name));

            for (const packageName of fixturePackageNames) {
                expect(packageNames.has(packageName)).toBe(true);
            }

            expect(packageNames.has("multi-version-demo")).toBe(true);
            expect(packageNames.has("pkg-2026-demo")).toBe(true);
            expect(packageNames.has("logging-package")).toBe(true);
            expect(packageNames.has("special-long-name-abcdefghijklmnopqrstuvwxyz0123456789abc")).toBe(true);
        } finally {
            await context.close();
        }
    });

    test.afterAll(async ({ browser }) => {
        const context = await newAuthorizedContext(browser);
        const page = await context.newPage();
        try {
            await deletePackageFixturesIfExist(page);
        } finally {
            try {
                await deleteCoreRepositoryIfExists(page);
            } finally {
                await context.close();
            }
        }
    });

    test("【Marketplace-013】平台管理-仓库-仓库配置(概览)-添加仓库-无需认证-基础功能验证", async ({ page }) => {
        const repositoryName = `pagination-add-none-${Date.now()}`;
        const repositoryUrl = "https://helm.openfuyao.cn/_dev";
        const repositoryRow = page.locator("tbody tr")
            .filter({ hasText: repositoryName })
            .filter({ hasText: repositoryUrl });

        try {
            await openRepositoryConfigPageByUrl(page);

            await page.getByRole("button", { name: "添加仓库" }).click();
            const addRepositoryDialog = page.getByRole("dialog").filter({ hasText: "添加仓库" }).last();
            await expect(addRepositoryDialog).toBeVisible();
            await addRepositoryDialog.locator("#form_in_modal_name").fill(repositoryName);
            await addRepositoryDialog.locator("#form_in_modal_url").fill(repositoryUrl);
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
            await expect(repositoryRow.first()).toBeVisible({ timeout: 30000 });
        } finally {
            await openRepositoryConfigPageByUrl(page);

            if (await repositoryRow.count()) {
                await repositoryRow.first().getByRole("img", { name: "more" }).click();
                await page.locator(".container-platform-popover button").filter({ hasText: "删除" }).click();

                const deleteDialog = page.getByRole("dialog").filter({ hasText: `确定删除仓库 ${repositoryName} 吗？` }).last();
                await expect(deleteDialog).toBeVisible({ timeout: 30000 });
                await deleteDialog.locator("label").filter({ hasText: "已了解删除操作无法恢复" }).click();
                await deleteDialog.locator("button").filter({ hasText: "删 除" }).click();
                await page.waitForLoadState("networkidle");
                await expect(repositoryRow).toHaveCount(0, { timeout: 30000 });
            }
        }
    });

    test("【Marketplace-015】平台管理-仓库-仓库配置-仓库详情-操作-删除(仅远端仓)-基础功能验证", async ({ page }) => {
        const repositoryName = `pagination-delete-${Date.now()}`;
        const repositoryUrl = "https://helm.openfuyao.cn/_dev";
        const repositoryRow = page.locator("tbody tr")
            .filter({ hasText: repositoryName })
            .filter({ hasText: repositoryUrl });

        try {
            await openRepositoryConfigPageByUrl(page);

            await page.getByRole("button", { name: "添加仓库" }).click();
            const addRepositoryDialog = page.getByRole("dialog").filter({ hasText: "添加仓库" }).last();
            await expect(addRepositoryDialog).toBeVisible();
            await addRepositoryDialog.locator("#form_in_modal_name").fill(repositoryName);
            await addRepositoryDialog.locator("#form_in_modal_url").fill(repositoryUrl);
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
            await expect(repositoryRow.first()).toBeVisible({ timeout: 30000 });

            await repositoryRow.first().locator("a").first().click();
            await page.waitForLoadState("networkidle");

            await expect(page).toHaveURL(new RegExp(`/container_platform/appMarket/stash/wareHouse/stashDetails/${repositoryName}$`));

            await page.getByRole("button", { name: "操作" }).click();
            await page.locator(".container-platform-popover button").filter({ hasText: "删除" }).click();

            const deleteDialog = page.getByRole("dialog").filter({ hasText: `确定删除仓库 ${repositoryName} 吗？` }).last();
            await expect(deleteDialog).toContainText(repositoryName);
            await deleteDialog.locator("label").filter({ hasText: "已了解删除操作无法恢复" }).click();
            await deleteDialog.locator("button").filter({ hasText: "删 除" }).click();

            await page.waitForLoadState("networkidle");
            await openRepositoryConfigPageByUrl(page);
            await expect(repositoryRow).toHaveCount(0, { timeout: 30000 });
        } finally {
            await openRepositoryConfigPageByUrl(page);

            if (await repositoryRow.count()) {
                await repositoryRow.first().getByRole("img", { name: "more" }).click();
                await page.locator(".container-platform-popover button").filter({ hasText: "删除" }).click();

                const deleteDialog = page.getByRole("dialog").filter({ hasText: `确定删除仓库 ${repositoryName} 吗？` }).last();
                await expect(deleteDialog).toBeVisible({ timeout: 30000 });
                await deleteDialog.locator("label").filter({ hasText: "已了解删除操作无法恢复" }).click();
                await deleteDialog.locator("button").filter({ hasText: "删 除" }).click();
                await page.waitForLoadState("networkidle");
                await expect(repositoryRow).toHaveCount(0, { timeout: 30000 });
            }
        }
    });

    test("【Marketplace-016】平台管理-仓库-仓库配置(概览)-操作-删除-基础功能验证", async ({ page }) => {
        const repositoryName = `pagination-overview-delete-${Date.now()}`;
        const repositoryUrl = "https://helm.openfuyao.cn/_dev";
        const repositoryRow = page.locator("tbody tr")
            .filter({ hasText: repositoryName })
            .filter({ hasText: repositoryUrl });

        try {
            await openRepositoryConfigPageByUrl(page);

            await page.getByRole("button", { name: "添加仓库" }).click();
            const addRepositoryDialog = page.getByRole("dialog").filter({ hasText: "添加仓库" }).last();
            await expect(addRepositoryDialog).toBeVisible();
            await addRepositoryDialog.locator("#form_in_modal_name").fill(repositoryName);
            await addRepositoryDialog.locator("#form_in_modal_url").fill(repositoryUrl);
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
            await expect(repositoryRow.first()).toBeVisible({ timeout: 30000 });

            await repositoryRow.first().getByRole("img", { name: "more" }).click();
            await page.locator(".container-platform-popover button").filter({ hasText: "删除" }).click();

            const deleteDialog = page.getByRole("dialog").filter({ hasText: `确定删除仓库 ${repositoryName} 吗？` }).last();
            await expect(deleteDialog).toContainText(repositoryName);
            await deleteDialog.locator("label").filter({ hasText: "已了解删除操作无法恢复" }).click();
            await deleteDialog.locator("button").filter({ hasText: "删 除" }).click();

            await page.waitForLoadState("networkidle");
            await expect(repositoryRow).toHaveCount(0, { timeout: 30000 });
        } finally {
            await openRepositoryConfigPageByUrl(page);

            if (await repositoryRow.count()) {
                await repositoryRow.first().getByRole("img", { name: "more" }).click();
                await page.locator(".container-platform-popover button").filter({ hasText: "删除" }).click();

                const deleteDialog = page.getByRole("dialog").filter({ hasText: `确定删除仓库 ${repositoryName} 吗？` }).last();
                await expect(deleteDialog).toBeVisible({ timeout: 30000 });
                await deleteDialog.locator("label").filter({ hasText: "已了解删除操作无法恢复" }).click();
                await deleteDialog.locator("button").filter({ hasText: "删 除" }).click();
                await page.waitForLoadState("networkidle");
                await expect(repositoryRow).toHaveCount(0, { timeout: 30000 });
            }
        }
    });

    test("【Marketplace-019】平台管理-仓库-仓库配置-仓库详情-操作-修改(仅远端仓)-基础功能验证", async ({ page }) => {
        const repositoryName = `pagination-detail-update-${Date.now()}`;
        const originalRepositoryUrl = "https://helm.openfuyao.cn/_dev";
        const updatedRepositoryUrl = "https://charts.bitnami.com/bitnami";
        const originalRepositoryRow = page.locator("tbody tr")
            .filter({ hasText: repositoryName })
            .filter({ hasText: originalRepositoryUrl });
        const updatedRepositoryRow = page.locator("tbody tr")
            .filter({ hasText: repositoryName })
            .filter({ hasText: updatedRepositoryUrl });

        try {
            await openRepositoryConfigPageByUrl(page);

            await page.getByRole("button", { name: "添加仓库" }).click();
            const addRepositoryDialog = page.getByRole("dialog").filter({ hasText: "添加仓库" }).last();
            await expect(addRepositoryDialog).toBeVisible();
            await addRepositoryDialog.locator("#form_in_modal_name").fill(repositoryName);
            await addRepositoryDialog.locator("#form_in_modal_url").fill(originalRepositoryUrl);
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
            await expect(originalRepositoryRow.first()).toBeVisible({ timeout: 30000 });

            await originalRepositoryRow.first().locator("a").first().click();
            await page.waitForLoadState("networkidle");

            await page.getByRole("button", { name: "操作" }).click();
            await page.locator(".container-platform-popover button").filter({ hasText: "修改" }).click();

            const updateDialog = page.getByRole("dialog").filter({ hasText: "修改仓库" }).last();
            await expect(updateDialog).toBeVisible();
            await updateDialog.locator("#form_in_modal_url").fill(updatedRepositoryUrl);
            await updateDialog.locator("label").filter({ hasText: "NONE" }).click();

            const updateSecurityRiskDialog = page.getByRole("dialog").filter({ hasText: "安全风险" }).last();
            await expect(updateSecurityRiskDialog).toBeVisible({ timeout: 30000 });
            await updateSecurityRiskDialog.locator("label").first().click();
            await updateSecurityRiskDialog.locator("button").filter({ hasText: /确\s*定/ }).click();

            await updateDialog.locator("button").filter({ hasText: /确\s*定/ }).click();

            await page.waitForLoadState("networkidle");
            await expect(page.locator("body")).toContainText(updatedRepositoryUrl);

            await openRepositoryConfigPageByUrl(page);
            await expect(updatedRepositoryRow.first()).toBeVisible({ timeout: 30000 });
        } finally {
            await openRepositoryConfigPageByUrl(page);

            if (await updatedRepositoryRow.count()) {
                await updatedRepositoryRow.first().getByRole("img", { name: "more" }).click();
                await page.locator(".container-platform-popover button").filter({ hasText: "删除" }).click();

                const deleteDialog = page.getByRole("dialog").filter({ hasText: `确定删除仓库 ${repositoryName} 吗？` }).last();
                await expect(deleteDialog).toBeVisible({ timeout: 30000 });
                await deleteDialog.locator("label").filter({ hasText: "已了解删除操作无法恢复" }).click();
                await deleteDialog.locator("button").filter({ hasText: "删 除" }).click();
                await page.waitForLoadState("networkidle");
                await expect(updatedRepositoryRow).toHaveCount(0, { timeout: 30000 });
            } else if (await originalRepositoryRow.count()) {
                await originalRepositoryRow.first().getByRole("img", { name: "more" }).click();
                await page.locator(".container-platform-popover button").filter({ hasText: "删除" }).click();

                const deleteDialog = page.getByRole("dialog").filter({ hasText: `确定删除仓库 ${repositoryName} 吗？` }).last();
                await expect(deleteDialog).toBeVisible({ timeout: 30000 });
                await deleteDialog.locator("label").filter({ hasText: "已了解删除操作无法恢复" }).click();
                await deleteDialog.locator("button").filter({ hasText: "删 除" }).click();
                await page.waitForLoadState("networkidle");
                await expect(originalRepositoryRow).toHaveCount(0, { timeout: 30000 });
            }
        }
    });

    test("【Marketplace-026】平台管理-仓库-仓库配置(概览)-操作-修改(仅远端仓)-基础功能验证", async ({ page }) => {
        const repositoryName = `pagination-overview-update-${Date.now()}`;
        const originalRepositoryUrl = "https://helm.openfuyao.cn/_dev";
        const updatedRepositoryUrl = "https://charts.bitnami.com/bitnami";
        const originalRepositoryRow = page.locator("tbody tr")
            .filter({ hasText: repositoryName })
            .filter({ hasText: originalRepositoryUrl });
        const updatedRepositoryRow = page.locator("tbody tr")
            .filter({ hasText: repositoryName })
            .filter({ hasText: updatedRepositoryUrl });

        try {
            await openRepositoryConfigPageByUrl(page);

            await page.getByRole("button", { name: "添加仓库" }).click();
            const addRepositoryDialog = page.getByRole("dialog").filter({ hasText: "添加仓库" }).last();
            await expect(addRepositoryDialog).toBeVisible();
            await addRepositoryDialog.locator("#form_in_modal_name").fill(repositoryName);
            await addRepositoryDialog.locator("#form_in_modal_url").fill(originalRepositoryUrl);
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
            await expect(originalRepositoryRow.first()).toBeVisible({ timeout: 30000 });

            await originalRepositoryRow.first().getByRole("img", { name: "more" }).click();
            await page.locator(".container-platform-popover button").filter({ hasText: "修改" }).click();

            const updateDialog = page.getByRole("dialog").filter({ hasText: "修改仓库" }).last();
            await expect(updateDialog).toBeVisible();
            await updateDialog.locator("#form_in_modal_url").fill(updatedRepositoryUrl);
            await updateDialog.locator("label").filter({ hasText: "NONE" }).click();

            const updateSecurityRiskDialog = page.getByRole("dialog").filter({ hasText: "安全风险" }).last();
            await expect(updateSecurityRiskDialog).toBeVisible({ timeout: 30000 });
            await updateSecurityRiskDialog.locator("label").first().click();
            await updateSecurityRiskDialog.locator("button").filter({ hasText: /确\s*定/ }).click();

            await updateDialog.locator("button").filter({ hasText: /确\s*定/ }).click();

            const updateCertificateRiskDialog = page.getByRole("dialog").filter({ hasText: "证书配置提示" }).last();
            await expect(updateCertificateRiskDialog).toBeVisible({ timeout: 30000 });
            await ensureFirstCheckboxChecked(updateCertificateRiskDialog);
            await updateCertificateRiskDialog.locator("button").filter({ hasText: "继 续" }).click();

            await page.waitForLoadState("networkidle");
            await expect.poll(async () => {
                await openRepositoryConfigPageByUrl(page);
                return await updatedRepositoryRow.count();
            }, {
                timeout: 60000,
                intervals: [1000, 2000, 3000],
            }).toBe(1);
        } finally {
            await openRepositoryConfigPageByUrl(page);

            if (await updatedRepositoryRow.count()) {
                await updatedRepositoryRow.first().getByRole("img", { name: "more" }).click();
                await page.locator(".container-platform-popover button").filter({ hasText: "删除" }).click();

                const deleteDialog = page.getByRole("dialog").filter({ hasText: `确定删除仓库 ${repositoryName} 吗？` }).last();
                await expect(deleteDialog).toBeVisible({ timeout: 30000 });
                await deleteDialog.locator("label").filter({ hasText: "已了解删除操作无法恢复" }).click();
                await deleteDialog.locator("button").filter({ hasText: "删 除" }).click();
                await page.waitForLoadState("networkidle");
                await expect(updatedRepositoryRow).toHaveCount(0, { timeout: 30000 });
            } else if (await originalRepositoryRow.count()) {
                await originalRepositoryRow.first().getByRole("img", { name: "more" }).click();
                await page.locator(".container-platform-popover button").filter({ hasText: "删除" }).click();

                const deleteDialog = page.getByRole("dialog").filter({ hasText: `确定删除仓库 ${repositoryName} 吗？` }).last();
                await expect(deleteDialog).toBeVisible({ timeout: 30000 });
                await deleteDialog.locator("label").filter({ hasText: "已了解删除操作无法恢复" }).click();
                await deleteDialog.locator("button").filter({ hasText: "删 除" }).click();
                await page.waitForLoadState("networkidle");
                await expect(originalRepositoryRow).toHaveCount(0, { timeout: 30000 });
            }
        }
    });

    test("【Marketplace-027】应用市场-首页-页面跳转-基础功能验证", async ({ page }) => {
        await openApplicationListPageByUrl(page);
        const applicationTitles = page.locator(".application_list .application_item .application_item_title");
        const pagination = page.locator(".container-platform-pagination");
        await expect(pagination).toBeVisible();
        const totalApplicationsText = ((await page.getByText(/共\d+个应用/).first().innerText()) || "").trim();
        const totalPaginationText = ((await pagination.locator(".container-platform-pagination-total-text").innerText()) || "").trim();
        const totalApplicationsCount = Number(totalApplicationsText.match(/\d+/)?.[0] || 0);
        const totalPaginationCount = Number(totalPaginationText.match(/\d+/)?.[0] || 0);
        const pageOneButton = page.locator(".container-platform-pagination-item-1");
        const pageTwoButton = page.locator(".container-platform-pagination-item-2");
        const prevButton = page.locator(".container-platform-pagination-prev");
        const nextButton = page.locator(".container-platform-pagination-next");
        const jumpInput = page.locator(".container-platform-pagination-options-quick-jumper input");

        const getVisibleTitles = async () => normalizeApplicationTitles(await applicationTitles.allTextContents());
        const getActivePageText = async () => ((await page.locator(".container-platform-pagination-item-active").textContent()) || "").trim();

        const pageOneTitles = await waitForRealApplicationTitles(applicationTitles);
        expect(totalApplicationsCount).toBeGreaterThan(pageOneTitles.length);
        expect(pageOneTitles.length).toBeGreaterThan(0);

        await expect(pageTwoButton).toBeVisible();
        await pageTwoButton.click();
        await expect.poll(async () => await getActivePageText()).toBe("2");
        const pageTwoTitles = await getVisibleTitles();
        expect(pageTwoTitles.length).toBeGreaterThan(0);
        expect(pageTwoTitles).not.toEqual(pageOneTitles);
        expect(totalApplicationsCount).toBe(pageOneTitles.length + pageTwoTitles.length);
        expect(totalPaginationCount).toBe(pageOneTitles.length + pageTwoTitles.length);

        await prevButton.click();
        await expect.poll(async () => await getActivePageText()).toBe("1");
        await expect.poll(async () => await getVisibleTitles()).toEqual(pageOneTitles);

        await nextButton.click();
        await expect.poll(async () => await getActivePageText()).toBe("2");
        await expect.poll(async () => await getVisibleTitles()).toEqual(pageTwoTitles);

        await jumpInput.fill("1");
        await jumpInput.press("Enter");
        await expect.poll(async () => await getActivePageText()).toBe("1");
        await expect(jumpInput).toHaveValue("");
        await expect.poll(async () => await getVisibleTitles()).toEqual(pageOneTitles);

        await jumpInput.fill("18");
        await jumpInput.press("Enter");
        await expect.poll(async () => await getActivePageText()).toBe("2");
        await expect(jumpInput).toHaveValue("");
        await expect.poll(async () => await getVisibleTitles()).toEqual(pageTwoTitles);
    });

    test("【Marketplace-004,005,025】平台管理-仓库-包管理-概览与详情展示、刷新-基础功能验证", async ({ page }) => {
        const displayPackageName = "logging-package";
        const displayPackageVersion = "0.0.0-latest";
        const displayPackageDescription = "openFuyao logging extension that integrates the logging operator, Loki stack, and proxy frontend to deliver a complete logging solution in Kubernetes clusters.";
        const refreshPackageName = "logging-package";
        const refreshPackageVersion = "0.0.0-latest";
        const refreshFixture = packageFixtures.find((item) => (
            item.packageName === refreshPackageName && item.version === refreshPackageVersion
        ));
        expect(refreshFixture).toBeTruthy();

        const enterPackageManagement = async (targetPage: Page) => {
            await targetPage.goto(`${TEST_FUYAOURL}${repositoryConfigUrl}`);
            await targetPage.waitForLoadState("networkidle");
            await expect(targetPage).toHaveURL(new RegExp(`${repositoryConfigUrl}$`));
            const packageManagementTab = targetPage.locator(".container-platform-tabs-tab-btn").filter({ hasText: "包管理" });
            await expect(packageManagementTab).toBeVisible({ timeout: 30000 });
            await packageManagementTab.click();
            await targetPage.waitForLoadState("networkidle");
            await expect(targetPage.getByRole("button", { name: "添加包" })).toBeVisible({ timeout: 30000 });
        };
        const getPackageCard = (targetPage: Page, packageName: string) =>
            targetPage.locator(".package_item").filter({ hasText: packageName }).first();

        let pageB: Page | undefined;

        try {
            await enterPackageManagement(page);
            const packageCards = page.locator(".package_item");
            await expect(packageCards.first()).toBeVisible({ timeout: 30000 });

            const displayPackageCard = getPackageCard(page, displayPackageName);
            await expect(displayPackageCard).toBeVisible({ timeout: 30000 });
            const displayCardText = await displayPackageCard.innerText();
            expect(displayCardText).toContain(displayPackageName);
            expect(displayCardText).toContain(packageFixtureUploadDateText);
            expect(displayCardText).toContain(displayPackageDescription);

            await displayPackageCard.click();
            await page.waitForLoadState("networkidle");
            await expect(page).toHaveURL(new RegExp(`/container_platform/appMarket/stash/packageManageDetails/.*/${displayPackageName}$`));

            await expect(page.locator("body")).toContainText(displayPackageName);
            await expect(page.locator("body")).toContainText(displayPackageVersion);
            await expect(page.locator("body")).toContainText(displayPackageDescription);
            await expect(page.locator("body")).toContainText("添加时间");
            await expect(page.locator("body")).toContainText("版本管理");

            await enterPackageManagement(page);
            const refreshPackageCard = getPackageCard(page, refreshPackageName);
            await expect(refreshPackageCard).toBeVisible({ timeout: 30000 });

            pageB = await page.context().newPage();
            await enterPackageManagement(pageB);
            const refreshPackageCardOnPageB = getPackageCard(pageB, refreshPackageName);
            await expect(refreshPackageCardOnPageB).toBeVisible({ timeout: 30000 });
            await refreshPackageCardOnPageB.locator("button").filter({ hasText: "删 除" }).click();

            const deleteDialog = pageB.getByRole("dialog").filter({ hasText: `确定删除包 ${refreshPackageName} 吗？` }).last();
            await expect(deleteDialog).toBeVisible({ timeout: 30000 });
            await deleteDialog.locator("label").filter({ hasText: "已了解删除操作无法恢复" }).click();
            await deleteDialog.locator("button").filter({ hasText: "删 除" }).click();
            await pageB.waitForLoadState("networkidle");

            await expect.poll(async () => {
                await syncCurrentRepositoryFromPackageManagement(page);
                const charts = await listAllLocalCharts(page);
                return charts.some((item) => item.metadata.name === refreshPackageName);
            }, {
                timeout: 90000,
                intervals: [3000, 5000, 5000],
            }).toBe(false);
            await expect(refreshPackageCard).toHaveCount(0, { timeout: 30000 });

            await uploadPackageFixtureOnPage(pageB, refreshFixture!);

            await expect.poll(async () => {
                await syncCurrentRepositoryFromPackageManagement(page);
                const charts = await listAllLocalCharts(page);
                const packageRestored = charts.some((item) => (
                    item.metadata.name === refreshPackageName && item.metadata.version === refreshPackageVersion
                ));
                const packageVisible = await refreshPackageCard.count() > 0;
                return packageRestored && packageVisible;
            }, {
                timeout: 90000,
                intervals: [3000, 5000, 5000],
            }).toBe(true);
        } finally {
            if (pageB) {
                await pageB.close();
            }
        }
    });

    test("【Marketplace-012,023,024】平台管理-仓库-包管理-添加与删除包-基础功能验证", async ({ page }) => {
        const m07PackagesDir = path.resolve(__dirname, "assets/charts/012,023,024-packages");
        const applicationTgzFixture: PackageFixture = {
            filePath: path.join(m07PackagesDir, "m07-app-tgz-0.1.0.tgz"),
            fileName: "m07-app-tgz-0.1.0.tgz",
            packageName: "m07-app-tgz",
            version: "0.1.0",
        };
        const applicationGzFixture: PackageFixture = {
            filePath: path.join(m07PackagesDir, "m07-app-gz-0.1.0.gz"),
            fileName: "m07-app-gz-0.1.0.gz",
            packageName: "m07-app-gz",
            version: "0.1.0",
        };
        const extensionFixture: PackageFixture = {
            filePath: path.join(m07PackagesDir, "npu-operator-0.0.0-latest.tgz"),
            fileName: "npu-operator-0.0.0-latest.tgz",
            packageName: "npu-operator",
            version: "0.0.0-latest",
        };
        const ensurePackageExists = async (fixture: PackageFixture) => {
            await expect.poll(async () => {
                const charts = await listAllLocalCharts(page);
                return charts.some((item) => (
                    item.metadata.name === fixture.packageName && item.metadata.version === fixture.version
                ));
            }, {
                timeout: 60000,
                intervals: [1000, 2000, 3000],
            }).toBe(true);
        };
        const searchPackageOnCurrentPage = async (packageName: string) => {
            const searchInput = page.locator('input[placeholder*="搜索"]').first();
            await expect(searchInput).toBeVisible({ timeout: 30000 });
            await searchInput.fill(packageName);
            await searchInput.press("Enter");
            await page.waitForLoadState("networkidle");
            await page.waitForTimeout(1000);
        };
        const getPackageCard = (packageName: string) =>
            page.locator(".package_item").filter({ hasText: packageName }).first();

        try {
            await deletePackageFixtureByName(page, applicationTgzFixture.packageName);
            await deletePackageFixtureByName(page, applicationGzFixture.packageName);
            await deletePackageFixtureByName(page, extensionFixture.packageName);

            await openPackageManagementPageByUrl(page);

            await uploadPackageFixtureOnPage(page, applicationTgzFixture);
            await ensurePackageExists(applicationTgzFixture);

            await uploadPackageFixtureOnPage(page, applicationGzFixture);
            await ensurePackageExists(applicationGzFixture);

            await uploadPackageFixtureOnPage(page, extensionFixture);
            await ensurePackageExists(extensionFixture);

            await searchPackageOnCurrentPage(extensionFixture.packageName);
            const extensionPackageCard = getPackageCard(extensionFixture.packageName);
            await expect(extensionPackageCard).toBeVisible({ timeout: 30000 });
            await extensionPackageCard.locator("button").filter({ hasText: "删 除" }).click();

            const deleteDialog = page.getByRole("dialog").filter({ hasText: `确定删除包 ${extensionFixture.packageName} 吗？` }).last();
            await expect(deleteDialog).toBeVisible({ timeout: 30000 });
            await deleteDialog.locator("label").filter({ hasText: "已了解删除操作无法恢复" }).click();
            await deleteDialog.locator("button").filter({ hasText: "删 除" }).click();

            await expect.poll(async () => {
                const charts = await listAllLocalCharts(page);
                return charts.some((item) => (
                    item.metadata.name === extensionFixture.packageName && item.metadata.version === extensionFixture.version
                ));
            }, {
                timeout: 60000,
                intervals: [1000, 2000, 3000],
            }).toBe(false);
            await expect(extensionPackageCard).toHaveCount(0, { timeout: 30000 });
        } finally {
            await deletePackageFixtureByName(page, applicationTgzFixture.packageName);
            await deletePackageFixtureByName(page, applicationGzFixture.packageName);
            await deletePackageFixtureByName(page, extensionFixture.packageName);
        }
    });

    test("【Marketplace-010,011,022】平台管理-仓库-包管理-包详情版本新增与删除-基础功能验证", async ({ page }) => {
        const m08PackagesDir = path.resolve(__dirname, "assets/charts/010,011,022-packages");
        const applicationPackageName = "multi-version-demo";
        const applicationVersionToDelete = "0.2.0";
        const applicationReAddFixture: PackageFixture = {
            filePath: path.join(generatedPackagesDir, "multi-version-demo-0.2.0.tgz"),
            fileName: "multi-version-demo-0.2.0.tgz",
            packageName: applicationPackageName,
            version: applicationVersionToDelete,
        };
        const extensionPackageName = "logging-package";
        const extensionVersionToAdd = "0.11.0";
        const extensionAddFixture: PackageFixture = {
            filePath: path.join(m08PackagesDir, "logging-package-0.11.0.tgz"),
            fileName: "logging-package-0.11.0.tgz",
            packageName: extensionPackageName,
            version: extensionVersionToAdd,
        };
        const searchPackageOnCurrentPage = async (packageName: string) => {
            const searchInput = page.locator('input[placeholder*="搜索"]').first();
            await expect(searchInput).toBeVisible({ timeout: 30000 });
            await searchInput.fill(packageName);
            await searchInput.press("Enter");
            await page.waitForLoadState("networkidle");
            await page.waitForTimeout(1000);
        };
        const getPackageCard = (packageName: string) =>
            page.locator(".package_item").filter({ hasText: packageName }).first();
        const getVersionRows = () => page.locator("tbody tr");
        const waitForVersionTableLoaded = async () => {
            await expect(page.getByRole("button", { name: "添加版本" })).toBeVisible({ timeout: 30000 });
            await expect.poll(async () => {
                return await getVersionRows().count();
            }, {
                timeout: 60000,
                intervals: [1000, 2000, 3000],
            }).toBeGreaterThan(0);
        };
        const openPackageDetail = async (packageName: string) => {
            await openPackageManagementPageByUrl(page);
            await searchPackageOnCurrentPage(packageName);
            const packageCard = getPackageCard(packageName);
            await expect(packageCard).toBeVisible({ timeout: 30000 });
            await packageCard.click();
            await page.waitForLoadState("networkidle");
            await expect(page).toHaveURL(new RegExp(`/container_platform/appMarket/stash/packageManageDetails/.*/${packageName}$`));
            await waitForVersionTableLoaded();
        };
        const getVersionRow = (version: string) =>
            getVersionRows().filter({ hasText: version }).first();
        const ensureVersionExists = async (version: string) => {
            const waitForVersionRow = async () => {
                await expect.poll(async () => {
                    return await getVersionRow(version).count();
                }, {
                    timeout: 60000,
                    intervals: [1000, 2000, 3000],
                }).toBe(1);
                await expect(getVersionRow(version)).toBeVisible({ timeout: 30000 });
            };

            try {
                await waitForVersionRow();
            } catch {
                await page.reload();
                await page.waitForLoadState("networkidle");
                await waitForVersionTableLoaded();
                await waitForVersionRow();
            }
        };
        const addVersionByFixture = async (fixture: PackageFixture) => {
            await page.getByRole("button", { name: "添加版本" }).click();
            const addVersionDialog = page.getByRole("dialog").last();
            await expect(addVersionDialog).toBeVisible({ timeout: 30000 });
            await addVersionDialog.locator('input[type="file"]').setInputFiles(fixture.filePath);
            await expect(addVersionDialog.getByText(fixture.fileName)).toBeVisible({ timeout: 30000 });
            await addVersionDialog.locator("button").filter({ hasText: /确\s*定/ }).click();
            await ensureVersionExists(fixture.version);
        };
        const deleteVersionIfExists = async (version: string, packageName: string) => {
            const versionRow = getVersionRow(version);
            if (!await versionRow.count()) {
                return;
            }

            await versionRow.getByRole("img", { name: "more" }).click();
            await page.locator("button").filter({ hasText: "删除" }).last().click();

            const deleteDialog = page.getByRole("dialog").last();
            await expect(deleteDialog).toBeVisible({ timeout: 30000 });
            await expect(deleteDialog).toContainText(packageName);
            await deleteDialog.locator("label").filter({ hasText: "已了解删除操作无法恢复" }).click();
            await deleteDialog.locator("button").filter({ hasText: "删 除" }).click();
            await page.waitForLoadState("networkidle");
            await page.waitForTimeout(2000);
            try {
                await expect.poll(async () => {
                    return await getVersionRow(version).count();
                }, {
                    timeout: 60000,
                    intervals: [1000, 2000, 3000],
                }).toBe(0);
            } catch {
                await page.reload();
                await page.waitForLoadState("networkidle");
                await waitForVersionTableLoaded();
                await expect(getVersionRow(version)).toHaveCount(0);
            }
            await expect(versionRow).toHaveCount(0, { timeout: 30000 });
        };

        try {
            await openPackageDetail(applicationPackageName);
            if (!await getVersionRow(applicationVersionToDelete).count()) {
                await addVersionByFixture(applicationReAddFixture);
            }

            await deleteVersionIfExists(applicationVersionToDelete, applicationPackageName);
            await addVersionByFixture(applicationReAddFixture);

            await openPackageDetail(extensionPackageName);
            await deleteVersionIfExists(extensionVersionToAdd, extensionPackageName);
            await addVersionByFixture(extensionAddFixture);
        } finally {
            await openPackageDetail(applicationPackageName);
            if (!await getVersionRow(applicationVersionToDelete).count()) {
                await addVersionByFixture(applicationReAddFixture);
            }

            await openPackageDetail(extensionPackageName);
            await ensureVersionExists(extensionVersionToAdd);
            await deleteVersionIfExists(extensionVersionToAdd, extensionPackageName);
        }
    });

    test("【Marketplace-032,033】平台管理-仓库-包管理-包详情版本管理排序与分页-基础功能验证", async ({ page }) => {
        const packageName = "multi-version-demo";
        const deletedVersionFixtures: PackageFixture[] = ["0.10.0", "0.11.0"].map((version) => ({
            filePath: path.join(generatedPackagesDir, `multi-version-demo-${version}.tgz`),
            fileName: `multi-version-demo-${version}.tgz`,
            packageName,
            version,
        }));
        const allDefaultVersions = ["0.1.0", "0.10.0", "0.11.0", "0.2.0", "0.3.0", "0.4.0", "0.5.0", "0.6.0", "0.7.0", "0.8.0", "0.9.0"];
        const defaultVersions = allDefaultVersions.slice(0, 10);
        const secondPageVersions = allDefaultVersions.slice(10);
        const versionAscVersions = [...allDefaultVersions].slice(0, 10);
        const versionDescVersions = [...allDefaultVersions].reverse().slice(0, 10);
        const searchPackageOnCurrentPage = async (targetPackageName: string) => {
            const searchInput = page.locator('input[placeholder*="搜索"]').first();
            await expect(searchInput).toBeVisible({ timeout: 30000 });
            await searchInput.fill(targetPackageName);
            await searchInput.press("Enter");
            await page.waitForLoadState("networkidle");
            await page.waitForTimeout(1000);
        };
        const getPackageCard = (targetPackageName: string) =>
            page.locator(".package_item").filter({ hasText: targetPackageName }).first();
        const getVersionRows = () => page.locator("tbody tr");
        const getVersionRecords = async () => {
            const rows = getVersionRows();
            const records: Array<{ version: string, createTime: string }> = [];
            for (let index = 0; index < await rows.count(); index++) {
                const cells = rows.nth(index).locator("td");
                records.push({
                    version: ((await cells.nth(0).innerText()) || "").trim(),
                    createTime: ((await cells.nth(1).innerText()) || "").trim(),
                });
            }
            return records;
        };
        const getVisibleVersions = async () => {
            return (await getVersionRecords()).map((item) => item.version);
        };
        const waitForVersionTableLoaded = async () => {
            await expect(page.getByRole("button", { name: "添加版本" })).toBeVisible({ timeout: 30000 });
            await expect.poll(async () => {
                return await getVersionRows().count();
            }, {
                timeout: 60000,
                intervals: [1000, 2000, 3000],
            }).toBeGreaterThan(0);
        };
        const openPackageDetail = async () => {
            await openPackageManagementPageByUrl(page);
            await searchPackageOnCurrentPage(packageName);
            const packageCard = getPackageCard(packageName);
            await expect(packageCard).toBeVisible({ timeout: 30000 });
            await packageCard.click();
            await page.waitForLoadState("networkidle");
            await expect(page).toHaveURL(new RegExp(`/container_platform/appMarket/stash/packageManageDetails/.*/${packageName}$`));
            await waitForVersionTableLoaded();
        };
        const ensureVersionExists = async (version: string) => {
            await expect.poll(async () => {
                return await getVersionRows().filter({ hasText: version }).count();
            }, {
                timeout: 60000,
                intervals: [1000, 2000, 3000],
            }).toBe(1);
        };
        const addVersionByFixture = async (fixture: PackageFixture) => {
            await page.getByRole("button", { name: "添加版本" }).click();
            const addVersionDialog = page.getByRole("dialog").last();
            await expect(addVersionDialog).toBeVisible({ timeout: 30000 });
            await addVersionDialog.locator('input[type="file"]').setInputFiles(fixture.filePath);
            await expect(addVersionDialog.getByText(fixture.fileName)).toBeVisible({ timeout: 30000 });
            await addVersionDialog.locator("button").filter({ hasText: /确\s*定/ }).click();
            await ensureVersionExists(fixture.version);
        };
        const ensureDeletedVersionsRestored = async () => {
            await openPackageDetail();
            for (const fixture of deletedVersionFixtures) {
                if (!await getVersionRows().filter({ hasText: fixture.version }).count()) {
                    await addVersionByFixture(fixture);
                    await openPackageDetail();
                }
            }
        };
        const deleteVersionAndReload = async (version: string) => {
            const versionRow = getVersionRows().filter({ hasText: version }).first();
            await expect(versionRow).toBeVisible({ timeout: 30000 });
            await versionRow.getByRole("img", { name: "more" }).click();
            await page.locator("button").filter({ hasText: "删除" }).last().click();
            const deleteDialog = page.getByRole("dialog").last();
            await expect(deleteDialog).toBeVisible({ timeout: 30000 });
            await deleteDialog.locator("label").filter({ hasText: "已了解删除操作无法恢复" }).click();
            await deleteDialog.locator("button").filter({ hasText: "删 除" }).click();
            await page.waitForLoadState("networkidle");
            await page.waitForTimeout(2000);
            await openPackageDetail();
            try {
                await expect.poll(async () => {
                    return await getVersionRows().filter({ hasText: version }).count();
                }, {
                    timeout: 60000,
                    intervals: [1000, 2000, 3000],
                }).toBe(0);
            } catch {
                await page.reload();
                await page.waitForLoadState("networkidle");
                await waitForVersionTableLoaded();
                await expect(getVersionRows().filter({ hasText: version })).toHaveCount(0);
            }
        };
        const setPageSize = async (optionIndex: number, expectedRowCount: number) => {
            await page.locator(".container-platform-pagination-options-size-changer").click();
            await page.waitForTimeout(1000);
            await page.locator('[role="option"]').nth(optionIndex).click();
            await expect.poll(async () => {
                return await getVersionRows().count();
            }, {
                timeout: 60000,
                intervals: [1000, 2000, 3000],
            }).toBe(expectedRowCount);
        };
        const compareVersion = (left: string, right: string) => left.localeCompare(right);

        try {
            await ensureDeletedVersionsRestored();

            await openPackageDetail();
            await expect(page.locator(".container-platform-pagination-total-text")).toContainText("11");
            await expect.poll(async () => await getVisibleVersions()).toEqual(defaultVersions);

            await setPageSize(1, 11);
            await expect.poll(async () => await getVisibleVersions()).toEqual(allDefaultVersions);

            const allDefaultRecords = await getVersionRecords();

            await setPageSize(0, 10);
            await expect.poll(async () => await getVisibleVersions()).toEqual(defaultVersions);

            const timeAscVersions = [...allDefaultRecords]
                .sort((left, right) => left.createTime.localeCompare(right.createTime) || compareVersion(left.version, right.version))
                .map((item) => item.version);
            const timeDescVersions = [...timeAscVersions].reverse().slice(0, 10);
            const timeAscFirstPageVersions = timeAscVersions.slice(0, 10);
            await expect.poll(async () => await getVisibleVersions()).toEqual(defaultVersions);

            await page.locator("thead th").nth(0).click();
            await expect.poll(async () => await getVisibleVersions()).toEqual(versionAscVersions);

            await page.locator("thead th").nth(0).click();
            await expect.poll(async () => await getVisibleVersions()).toEqual(versionDescVersions);

            await openPackageDetail();
            await page.locator("thead th").nth(1).click();
            await expect.poll(async () => await getVisibleVersions()).toEqual(timeAscFirstPageVersions);

            await page.locator("thead th").nth(1).click();
            await expect.poll(async () => await getVisibleVersions()).toEqual(timeDescVersions);

            await openPackageDetail();
            await page.locator(".container-platform-pagination-item-2").click();
            await expect.poll(async () => await getVisibleVersions()).toEqual(secondPageVersions);

            await page.locator(".container-platform-pagination-prev").click();
            await expect.poll(async () => await getVisibleVersions()).toEqual(defaultVersions);

            await page.locator(".container-platform-pagination-next").click();
            await expect.poll(async () => await getVisibleVersions()).toEqual(secondPageVersions);

            await page.locator(".container-platform-pagination-item-1").click();
            await expect.poll(async () => await getVisibleVersions()).toEqual(defaultVersions);

            const jumpInput = page.locator(".container-platform-pagination-options-quick-jumper input");
            await jumpInput.fill("2");
            await jumpInput.press("Enter");
            await expect.poll(async () => await getVisibleVersions()).toEqual(secondPageVersions);

            await jumpInput.fill("1");
            await jumpInput.press("Enter");
            await expect.poll(async () => await getVisibleVersions()).toEqual(defaultVersions);

            await setPageSize(1, 11);
            await expect.poll(async () => await getVisibleVersions()).toEqual(allDefaultVersions);

            await setPageSize(0, 10);
            await openPackageDetail();
            await deleteVersionAndReload("0.10.0");
            await deleteVersionAndReload("0.11.0");

            await expect(getVersionRows()).toHaveCount(9, { timeout: 30000 });
            await expect(page.locator(".container-platform-pagination")).toHaveCount(0, { timeout: 30000 });
        } finally {
            await ensureDeletedVersionsRestored();
        }
    });

    test("【Marketplace-037,060】平台管理-仓库-包管理-搜索-基础功能验证", async ({ page }) => {
        const exactPackageName = "alpha-demo";
        const fuzzyKeyword = "demo";
        const upperCaseKeyword = "DEMO";
        const numericKeyword = "2026";
        const emptyKeyword = "not-exist-package-name";
        const expectedFuzzyNames = ["alpha-demo", "multi-version-demo", "pkg-2026-demo"];
        const getSearchInput = () => page.locator('input[placeholder="搜索应用名称"]').first();
        const getPackageCards = () => page.locator(".package_item");
        const getVisiblePackageNames = async () => {
            const cards = getPackageCards();
            const names: string[] = [];
            for (let index = 0; index < await cards.count(); index++) {
                const cardText = ((await cards.nth(index).innerText()) || "").trim();
                names.push(cardText.split("\n")[0]);
            }
            return names;
        };
        const searchPackages = async (keyword: string) => {
            const searchInput = getSearchInput();
            await expect(searchInput).toBeVisible({ timeout: 30000 });
            await searchInput.fill(keyword);
            await searchInput.press("Enter");
            await page.waitForLoadState("networkidle");
            await page.waitForTimeout(1000);
        };
        const clearSearchAndExpectDefault = async (defaultVisibleNames: string[]) => {
            const searchInput = getSearchInput();
            await expect(searchInput).toBeVisible({ timeout: 30000 });
            await searchInput.fill("");
            await searchInput.press("Enter");
            await page.waitForLoadState("networkidle");
            await page.waitForTimeout(1000);
            await expect.poll(async () => await getVisiblePackageNames()).toEqual(defaultVisibleNames);
        };

        await openPackageManagementPageByUrl(page);

        const allCharts = await listAllLocalCharts(page);
        const packageNames = [...new Set(allCharts.map((item) => item.metadata.name))];
        expect(packageNames.length).toBeGreaterThan(5);
        expect(packageNames.some((name) => /\d/.test(name))).toBe(true);
        await expect(getPackageCards().first()).toBeVisible({ timeout: 30000 });
        const defaultVisibleNames = await getVisiblePackageNames();
        expect(defaultVisibleNames.length).toBeGreaterThan(0);

        await searchPackages(exactPackageName);
        await expect.poll(async () => await getVisiblePackageNames()).toEqual([exactPackageName]);
        await clearSearchAndExpectDefault(defaultVisibleNames);

        await searchPackages(fuzzyKeyword);
        await expect.poll(async () => await getVisiblePackageNames()).toEqual(expectedFuzzyNames);
        await clearSearchAndExpectDefault(defaultVisibleNames);

        await searchPackages(upperCaseKeyword);
        await expect.poll(async () => await getVisiblePackageNames()).toEqual(expectedFuzzyNames);
        await clearSearchAndExpectDefault(defaultVisibleNames);

        await searchPackages(numericKeyword);
        await expect.poll(async () => await getVisiblePackageNames()).toEqual(["pkg-2026-demo"]);
        await clearSearchAndExpectDefault(defaultVisibleNames);

        await searchPackages(emptyKeyword);
        await expect(getPackageCards()).toHaveCount(0, { timeout: 30000 });
        await clearSearchAndExpectDefault(defaultVisibleNames);
    });

    test("【Marketplace-034,059】平台管理-仓库-包管理-概览分页与超长信息显示-基础功能验证", async ({ page }) => {
        const specialPackageName = "special-long-name-abcdefghijklmnopqrstuvwxyz0123456789abc";
        const specialPackageDescription = "This package is intentionally generated for marketplace package-management tests. It exists to cover long-description display, search, pagination, and upload flows. The content is synthetic but packaged with Helm and valid for smoke-level upload checks. This package is intentionally generated for marketplace package-management tests. It exists to cover long-description display, search, pagination, and upload flows. The content is synthetic but packaged with Helm and valid for smoke-level upload checks.";
        const getPackagePagination = () => page.locator(".container-platform-pagination").last();
        const getPackageCards = () => page.locator(".package_item");
        const getVisiblePackageNames = async () => {
            const cards = getPackageCards();
            const names: string[] = [];
            for (let index = 0; index < await cards.count(); index++) {
                const cardText = ((await cards.nth(index).innerText()) || "").trim();
                names.push(cardText.split("\n")[0]);
            }
            return names;
        };
        const setPageSize = async (optionIndex: number, expectedCount: number) => {
            await getPackagePagination().locator(".container-platform-pagination-options-size-changer").click();
            await page.waitForTimeout(1000);
            await page.locator('[role="option"]').nth(optionIndex).click();
            await expect.poll(async () => {
                return await getPackageCards().count();
            }, {
                timeout: 60000,
                intervals: [1000, 2000, 3000],
            }).toBe(expectedCount);
        };

        await openPackageManagementPageByUrl(page);

        const allCharts = await listAllLocalCharts(page);
        const distinctPackageNames = [...new Set(allCharts.map((item) => item.metadata.name))];
        const totalPackageCount = distinctPackageNames.length;
        expect(totalPackageCount).toBeGreaterThan(15);
        expect(distinctPackageNames).toContain(specialPackageName);

        const firstPageCount = Math.min(10, totalPackageCount);
        const secondPageCount = Math.max(totalPackageCount - firstPageCount, 0);

        await expect(getPackagePagination().locator(".container-platform-pagination-total-text")).toContainText(String(totalPackageCount));
        await expect(getPackageCards().first()).toBeVisible({ timeout: 30000 });
        await expect(getPackageCards()).toHaveCount(firstPageCount, { timeout: 30000 });

        await getPackagePagination().locator(".container-platform-pagination-item-2").click();
        await expect(getPackageCards()).toHaveCount(secondPageCount, { timeout: 30000 });

        await getPackagePagination().locator(".container-platform-pagination-prev").click();
        await expect(getPackageCards()).toHaveCount(firstPageCount, { timeout: 30000 });

        await getPackagePagination().locator(".container-platform-pagination-next").click();
        await expect(getPackageCards()).toHaveCount(secondPageCount, { timeout: 30000 });

        await getPackagePagination().locator(".container-platform-pagination-item-1").click();
        await expect(getPackageCards()).toHaveCount(firstPageCount, { timeout: 30000 });

        const jumpInput = getPackagePagination().locator(".container-platform-pagination-options-quick-jumper input");
        await jumpInput.fill("2");
        await jumpInput.press("Enter");
        await expect(getPackageCards()).toHaveCount(secondPageCount, { timeout: 30000 });

        await jumpInput.fill("1");
        await jumpInput.press("Enter");
        await expect(getPackageCards()).toHaveCount(firstPageCount, { timeout: 30000 });

        await setPageSize(1, totalPackageCount);
        await expect.poll(async () => await getVisiblePackageNames()).toEqual(distinctPackageNames);

        const specialPackageCard = getPackageCards().filter({ hasText: specialPackageName }).first();
        await expect(specialPackageCard).toBeVisible({ timeout: 30000 });
        await expect(specialPackageCard).toContainText(specialPackageName);
        await expect(specialPackageCard).toContainText(specialPackageDescription);

        await setPageSize(0, firstPageCount);
        await expect(getPackageCards()).toHaveCount(firstPageCount, { timeout: 30000 });
    });

    test("【Marketplace-035】平台管理-仓库-仓库配置-仓库详情-筛选-基础功能验证", async ({ page }) => {
        const repositoryName = "openFuyao";
        const waitForRepositoryDetailTableLoaded = async () => {
            await expect.poll(async () => {
                const rows = page.locator(".container-platform-table-tbody tr");
                const rowCount = await rows.count();
                if (!rowCount) {
                    return false;
                }

                const firstRowText = ((await rows.first().innerText()) || "").trim();
                if (!firstRowText || firstRowText.includes("暂无数据")) {
                    return false;
                }

                const firstRowType = ((await rows.first().locator("td").nth(1).innerText()) || "").trim();
                return firstRowType === "应用" || firstRowType === "扩展组件";
            }, {
                timeout: 60000,
                intervals: [1000, 2000, 3000],
            }).toBe(true);
        };
        const openOfficialRepositoryDetail = async () => {
            await openRepositoryConfigPageByUrl(page);
            const officialRepositoryRow = page.locator("tbody tr").filter({ hasText: repositoryName }).last();
            const officialRepositoryLink = officialRepositoryRow.locator("td").first().locator("a").first();
            await expect(officialRepositoryLink).toBeVisible({ timeout: 30000 });
            await officialRepositoryLink.click();
            await page.waitForLoadState("networkidle");
            await expect(page).toHaveURL(new RegExp(`/container_platform/appMarket/stash/wareHouse/stashDetails/${repositoryName}$`));
            await waitForRepositoryDetailTableLoaded();
        };
        const getDetailRows = () => page.locator(".container-platform-table-tbody tr");
        const getVisibleTypes = async () => {
            const rows = getDetailRows();
            const types: string[] = [];
            for (let index = 0; index < await rows.count(); index++) {
                const typeText = ((await rows.nth(index).locator("td").nth(1).innerText()) || "").trim();
                types.push(typeText);
            }
            return types;
        };
        const openTypeFilter = async () => {
            const typeHeader = page.locator("thead th").nth(1);
            await expect(typeHeader).toBeVisible({ timeout: 30000 });
            const trigger = typeHeader.locator(".container-platform-table-filter-trigger");
            await expect(trigger).toBeVisible({ timeout: 30000 });
            await trigger.click();
            const dropdown = page.locator(".container-platform-table-filter-dropdown").last();
            await expect(dropdown).toBeVisible({ timeout: 30000 });
            return dropdown;
        };
        const applyTypeFilter = async (typeText: string) => {
            const dropdown = await openTypeFilter();
            const option = dropdown.locator('[role="menuitem"]').filter({ hasText: typeText }).first();
            await expect(option).toBeVisible({ timeout: 30000 });
            await option.click();
            const confirmButton = dropdown.locator("button").filter({ hasText: /确\s*定/ }).last();
            await expect(confirmButton).toBeVisible({ timeout: 30000 });
            await confirmButton.click();
            await page.waitForLoadState("networkidle");
            await page.waitForTimeout(1000);
        };
        const resetTypeFilter = async () => {
            const dropdown = await openTypeFilter();
            const resetButton = dropdown.locator("button").filter({ hasText: "重置" }).last();
            await expect(resetButton).toBeVisible({ timeout: 30000 });
            await resetButton.click();
            const confirmButton = dropdown.locator("button").filter({ hasText: /确\s*定/ }).last();
            await expect(confirmButton).toBeVisible({ timeout: 30000 });
            await confirmButton.click();
            await page.waitForLoadState("networkidle");
            await page.waitForTimeout(1000);
        };

        await openOfficialRepositoryDetail();
        const totalText = ((await page.locator(".container-platform-pagination-total-text").last().innerText()) || "").trim();
        const totalCount = Number(totalText.match(/\d+/)?.[0] || 0);
        expect(totalCount).toBeGreaterThan(5);
        const initialVisibleTypes = await getVisibleTypes();
        expect(initialVisibleTypes.length).toBeGreaterThan(0);
        expect(initialVisibleTypes).toContain("应用");
        expect(initialVisibleTypes).toContain("扩展组件");

        await applyTypeFilter("应用");
        await expect(getDetailRows().first()).toBeVisible({ timeout: 30000 });
        expect((await getVisibleTypes()).every((type) => type === "应用")).toBe(true);

        await resetTypeFilter();
        await expect(getDetailRows().first()).toBeVisible({ timeout: 30000 });
        await expect.poll(async () => await getVisibleTypes()).toEqual(initialVisibleTypes);
        await applyTypeFilter("扩展组件");
        await expect(getDetailRows().first()).toBeVisible({ timeout: 30000 });
        expect((await getVisibleTypes()).every((type) => type === "扩展组件")).toBe(true);
    });

    test("【Marketplace-036】平台管理-仓库-仓库配置-仓库详情-页面跳转-基础功能验证", async ({ page }) => {
        const repositoryName = "local";
        const packageNamesToDelete = [
            "alpha-demo",
            "beta-ops",
            "delta-service",
            "epsilon-tools",
            "eta-runner",
            "gamma-stack",
            "iota-api",
            "kappa-worker",
        ];
        const restoreFixtures = packageNamesToDelete.map((packageName) => {
            const fixture = packageFixtures.find((item) => item.packageName === packageName);
            expect(fixture).toBeTruthy();
            return fixture!;
        });
        const waitForRepositoryDetailTableLoaded = async () => {
            await expect.poll(async () => {
                const rows = page.locator(".container-platform-table-tbody tr");
                const rowCount = await rows.count();
                if (!rowCount) {
                    return false;
                }

                const firstRowText = ((await rows.first().innerText()) || "").trim();
                if (!firstRowText || firstRowText.includes("暂无数据")) {
                    return false;
                }

                const firstRowName = ((await rows.first().locator("td").first().innerText()) || "").trim();
                return firstRowName !== "";
            }, {
                timeout: 60000,
                intervals: [1000, 2000, 3000],
            }).toBe(true);
        };
        const openLocalRepositoryDetail = async () => {
            await openRepositoryConfigPageByUrl(page);
            const localRepositoryRow = page.locator("tbody tr").filter({ hasText: repositoryName }).first();
            const localRepositoryLink = localRepositoryRow.locator("td").first().locator("a").first();
            await expect(localRepositoryLink).toBeVisible({ timeout: 30000 });
            await localRepositoryLink.click();
            await page.waitForLoadState("networkidle");
            await expect(page).toHaveURL(new RegExp(`/container_platform/appMarket/stash/wareHouse/stashDetails/${repositoryName}$`));
            await waitForRepositoryDetailTableLoaded();
        };
        const getDetailRows = () => page.locator(".container-platform-table-tbody tr");
        const getDetailPagination = () => page.locator(".container-platform-pagination").last();
        const getVisiblePackageNames = async () => {
            const rows = getDetailRows();
            const names: string[] = [];
            for (let index = 0; index < await rows.count(); index++) {
                const nameText = ((await rows.nth(index).locator("td").first().innerText()) || "").trim();
                names.push(nameText.split("\n")[0]);
            }
            return names;
        };
        const setPageSize = async (optionIndex: number, expectedCount: number) => {
            await getDetailPagination().locator(".container-platform-pagination-options-size-changer").click();
            await page.waitForTimeout(1000);
            await page.locator('[role="option"]').nth(optionIndex).click();
            await expect.poll(async () => {
                return await getDetailRows().count();
            }, {
                timeout: 60000,
                intervals: [1000, 2000, 3000],
            }).toBe(expectedCount);
        };
        const ensureDeletedPackagesRestored = async () => {
            await openPackageManagementPageByUrl(page);
            for (const fixture of restoreFixtures) {
                await ensurePackageFixtureUploaded(page, fixture);
            }
        };

        try {
            await ensureDeletedPackagesRestored();
            await openLocalRepositoryDetail();

            const allCharts = await listAllLocalCharts(page);
            const distinctPackageNames = [...new Set(allCharts.map((item) => item.metadata.name))];
            const totalPackageCount = distinctPackageNames.length;
            expect(totalPackageCount).toBeGreaterThan(15);

            const firstPageCount = Math.min(10, totalPackageCount);
            const secondPageCount = Math.max(totalPackageCount - firstPageCount, 0);
            await expect(getDetailPagination().locator(".container-platform-pagination-total-text")).toContainText(String(totalPackageCount));
            await expect(getDetailRows()).toHaveCount(firstPageCount, { timeout: 30000 });

            const firstPageNames = await getVisiblePackageNames();
            expect(firstPageNames.length).toBe(firstPageCount);

            await getDetailPagination().locator(".container-platform-pagination-item-2").click();
            await expect(getDetailRows()).toHaveCount(secondPageCount, { timeout: 30000 });
            const secondPageNames = await getVisiblePackageNames();
            expect(secondPageNames.length).toBe(secondPageCount);
            expect(secondPageNames).not.toEqual(firstPageNames);

            await getDetailPagination().locator(".container-platform-pagination-prev").click();
            await expect.poll(async () => await getVisiblePackageNames()).toEqual(firstPageNames);

            await getDetailPagination().locator(".container-platform-pagination-next").click();
            await expect.poll(async () => await getVisiblePackageNames()).toEqual(secondPageNames);

            await getDetailPagination().locator(".container-platform-pagination-item-1").click();
            await expect.poll(async () => await getVisiblePackageNames()).toEqual(firstPageNames);

            const jumpInput = getDetailPagination().locator(".container-platform-pagination-options-quick-jumper input");
            await jumpInput.fill("2");
            await jumpInput.press("Enter");
            await expect.poll(async () => await getVisiblePackageNames()).toEqual(secondPageNames);

            await jumpInput.fill("1");
            await jumpInput.press("Enter");
            await expect.poll(async () => await getVisiblePackageNames()).toEqual(firstPageNames);

            await setPageSize(1, totalPackageCount);
            await expect.poll(async () => await getVisiblePackageNames()).toEqual(distinctPackageNames);

            await setPageSize(0, firstPageCount);
            await expect(getDetailRows()).toHaveCount(firstPageCount, { timeout: 30000 });

            for (const packageName of packageNamesToDelete) {
                await deletePackageFixtureByName(page, packageName);
            }

            await openLocalRepositoryDetail();
            await expect(getDetailRows()).toHaveCount(9, { timeout: 30000 });
            await expect(getDetailPagination().locator(".container-platform-pagination-total-text")).toContainText("9");
            await expect(getDetailPagination().locator(".container-platform-pagination-item-2")).toHaveCount(0, { timeout: 30000 });
            await expect(
                getDetailPagination().locator(".container-platform-pagination-options-quick-jumper input"),
            ).toHaveCount(0, { timeout: 30000 });
        } finally {
            await ensureDeletedPackagesRestored();
        }
    });

    test("【Marketplace-038】平台管理-仓库-仓库配置-仓库详情-排序-基础功能验证", async ({ page }) => {
        const repositoryName = "openFuyao";
        const waitForRepositoryDetailTableLoaded = async () => {
            await expect.poll(async () => {
                const rows = page.locator(".container-platform-table-tbody tr");
                const rowCount = await rows.count();
                if (!rowCount) {
                    return false;
                }

                const firstRowText = ((await rows.first().innerText()) || "").trim();
                if (!firstRowText || firstRowText.includes("暂无数据")) {
                    return false;
                }

                const firstRowName = ((await rows.first().locator("td").first().innerText()) || "").trim();
                return firstRowName !== "";
            }, {
                timeout: 60000,
                intervals: [1000, 2000, 3000],
            }).toBe(true);
        };
        const openOfficialRepositoryDetail = async () => {
            await openRepositoryConfigPageByUrl(page);
            const officialRepositoryRow = page.locator("tbody tr").filter({ hasText: repositoryName }).last();
            const officialRepositoryLink = officialRepositoryRow.locator("td").first().locator("a").first();
            await expect(officialRepositoryLink).toBeVisible({ timeout: 30000 });
            await officialRepositoryLink.click();
            await page.waitForLoadState("networkidle");
            await expect(page).toHaveURL(new RegExp(`/container_platform/appMarket/stash/wareHouse/stashDetails/${repositoryName}$`));
            await waitForRepositoryDetailTableLoaded();
        };
        const getDetailRows = () => page.locator(".container-platform-table-tbody tr");
        const getDetailPagination = () => page.locator(".container-platform-pagination").last();
        const getRowRecords = async () => {
            const rows = getDetailRows();
            const records: Array<{ name: string, type: string, version: string, createTime: string }> = [];
            for (let index = 0; index < await rows.count(); index++) {
                const cells = rows.nth(index).locator("td");
                const nameText = ((await cells.nth(0).innerText()) || "").trim().split("\n")[0];
                const typeText = ((await cells.nth(1).innerText()) || "").trim();
                const versionText = ((await cells.nth(2).innerText()) || "").trim();
                const createTimeText = ((await cells.nth(3).innerText()) || "").trim();
                records.push({
                    name: nameText,
                    type: typeText,
                    version: versionText,
                    createTime: createTimeText,
                });
            }
            return records;
        };
        const getVisibleNames = async () => (await getRowRecords()).map((item) => item.name);
        const getVisibleCreateTimes = async () => (await getRowRecords()).map((item) => item.createTime);
        const setPageSize = async (pageSizeText: string, expectedCount: number) => {
            await getDetailPagination().locator(".container-platform-pagination-options-size-changer").click();
            await page.waitForTimeout(1000);
            await page.locator('[role="option"]').filter({ hasText: pageSizeText }).first().click();
            await expect.poll(async () => {
                return await getDetailRows().count();
            }, {
                timeout: 60000,
                intervals: [1000, 2000, 3000],
            }).toBe(expectedCount);
        };

        await openOfficialRepositoryDetail();

        const totalText = ((await getDetailPagination().locator(".container-platform-pagination-total-text").innerText()) || "").trim();
        const totalCount = Number(totalText.match(/\d+/)?.[0] || 0);
        expect(totalCount).toBeGreaterThan(5);

        await setPageSize("20 条/页", totalCount);

        const defaultRecords = await getRowRecords();
        const defaultNames = defaultRecords.map((item) => item.name);
        const defaultCreateTimes = defaultRecords.map((item) => item.createTime);

        expect(defaultNames).toEqual([...defaultNames].sort((left, right) => left.localeCompare(right)));

        await page.locator("thead th").nth(0).click();
        await expect.poll(async () => await getVisibleNames()).toEqual(defaultNames);

        const nameDesc = [...defaultNames].reverse();
        await page.locator("thead th").nth(0).click();
        await expect.poll(async () => await getVisibleNames()).toEqual(nameDesc);

        await openOfficialRepositoryDetail();
        await setPageSize("20 条/页", totalCount);
        await page.locator("thead th").nth(3).click();
        const timeAsc = [...defaultCreateTimes].sort((left, right) => left.localeCompare(right));
        await expect.poll(async () => await getVisibleCreateTimes()).toEqual(timeAsc);

        await page.locator("thead th").nth(3).click();
        await expect.poll(async () => await getVisibleCreateTimes()).toEqual([...timeAsc].reverse());
    });

    test("【Marketplace-042】平台管理-仓库-仓库配置-仓库详情-搜索-基础功能验证", async ({ page }) => {
        const repositoryName = "local";
        const exactPackageName = "alpha-demo";
        const fuzzyKeyword = "demo";
        const upperCaseKeyword = "DEMO";
        const numericKeyword = "2026";
        const expectedFuzzyNames = ["alpha-demo", "multi-version-demo", "pkg-2026-demo"];
        const waitForRepositoryDetailTableLoaded = async () => {
            await expect.poll(async () => {
                const rows = page.locator(".container-platform-table-tbody tr");
                const rowCount = await rows.count();
                if (!rowCount) {
                    return false;
                }

                const firstRowText = ((await rows.first().innerText()) || "").trim();
                if (!firstRowText || firstRowText.includes("暂无数据")) {
                    return false;
                }

                const firstRowName = ((await rows.first().locator("td").first().innerText()) || "").trim();
                return firstRowName !== "";
            }, {
                timeout: 60000,
                intervals: [1000, 2000, 3000],
            }).toBe(true);
        };
        const openLocalRepositoryDetail = async () => {
            await openRepositoryConfigPageByUrl(page);
            const localRepositoryRow = page.locator("tbody tr").filter({ hasText: repositoryName }).first();
            const localRepositoryLink = localRepositoryRow.locator("td").first().locator("a").first();
            await expect(localRepositoryLink).toBeVisible({ timeout: 30000 });
            await localRepositoryLink.click();
            await page.waitForLoadState("networkidle");
            await expect(page).toHaveURL(new RegExp(`/container_platform/appMarket/stash/wareHouse/stashDetails/${repositoryName}$`));
            await waitForRepositoryDetailTableLoaded();
        };
        const getDetailRows = () => page.locator(".container-platform-table-tbody tr");
        const getSearchInput = () => page.getByPlaceholder("搜索包名称");
        const getVisiblePackageNames = async () => {
            const rows = getDetailRows();
            const names: string[] = [];
            for (let index = 0; index < await rows.count(); index++) {
                const nameText = ((await rows.nth(index).locator("td").first().innerText()) || "").trim();
                names.push(nameText.split("\n")[0]);
            }
            return names;
        };
        const searchPackages = async (keyword: string) => {
            const searchInput = getSearchInput();
            await expect(searchInput).toBeVisible({ timeout: 30000 });
            await searchInput.fill(keyword);
            await searchInput.press("Enter");
            await page.waitForLoadState("networkidle");
            await page.waitForTimeout(1000);
        };
        const clearSearchAndExpectDefault = async (defaultVisibleNames: string[]) => {
            const searchInput = getSearchInput();
            await expect(searchInput).toBeVisible({ timeout: 30000 });
            await searchInput.fill("");
            await searchInput.press("Enter");
            await page.waitForLoadState("networkidle");
            await page.waitForTimeout(1000);
            await expect.poll(async () => await getVisiblePackageNames()).toEqual(defaultVisibleNames);
        };

        await openLocalRepositoryDetail();

        const totalText = ((await page.locator(".container-platform-pagination-total-text").last().innerText()) || "").trim();
        const totalCount = Number(totalText.match(/\d+/)?.[0] || 0);
        expect(totalCount).toBeGreaterThan(5);

        const defaultVisibleNames = await getVisiblePackageNames();
        expect(defaultVisibleNames.length).toBeGreaterThan(5);

        await searchPackages(exactPackageName);
        await expect.poll(async () => await getVisiblePackageNames()).toEqual([exactPackageName]);
        await clearSearchAndExpectDefault(defaultVisibleNames);

        await searchPackages(fuzzyKeyword);
        await expect.poll(async () => await getVisiblePackageNames()).toEqual(expectedFuzzyNames);
        await clearSearchAndExpectDefault(defaultVisibleNames);

        await searchPackages(upperCaseKeyword);
        await expect.poll(async () => await getVisiblePackageNames()).toEqual(expectedFuzzyNames);
        await clearSearchAndExpectDefault(defaultVisibleNames);

        await searchPackages(numericKeyword);
        await expect.poll(async () => await getVisiblePackageNames()).toEqual(["pkg-2026-demo"]);
        await clearSearchAndExpectDefault(defaultVisibleNames);
    });

    test("【Marketplace-061】平台管理-仓库-仓库配置-仓库详情-搜索-构造异常-目标不存在", async ({ page }) => {
        const repositoryName = "openFuyao";
        const nonexistentKeyword = "not-exist-package-name-23456";
        const waitForRepositoryDetailTableLoaded = async () => {
            await expect.poll(async () => {
                const rows = page.locator(".container-platform-table-tbody tr");
                const rowCount = await rows.count();
                if (!rowCount) {
                    return false;
                }

                const firstRowText = ((await rows.first().innerText()) || "").trim();
                if (!firstRowText || firstRowText.includes("暂无数据")) {
                    return false;
                }

                const firstRowName = ((await rows.first().locator("td").first().innerText()) || "").trim();
                return firstRowName !== "";
            }, {
                timeout: 60000,
                intervals: [1000, 2000, 3000],
            }).toBe(true);
        };
        const openOfficialRepositoryDetail = async () => {
            await openRepositoryConfigPageByUrl(page);
            const officialRepositoryRow = page.locator("tbody tr").filter({ hasText: repositoryName }).last();
            const officialRepositoryLink = officialRepositoryRow.locator("td").first().locator("a").first();
            await expect(officialRepositoryLink).toBeVisible({ timeout: 30000 });
            await officialRepositoryLink.click();
            await page.waitForLoadState("networkidle");
            await expect(page).toHaveURL(new RegExp(`/container_platform/appMarket/stash/wareHouse/stashDetails/${repositoryName}$`));
            await waitForRepositoryDetailTableLoaded();
        };

        await openOfficialRepositoryDetail();

        const totalText = ((await page.locator(".container-platform-pagination-total-text").last().innerText()) || "").trim();
        const totalCount = Number(totalText.match(/\d+/)?.[0] || 0);
        expect(totalCount).toBeGreaterThan(5);

        const searchInput = page.getByPlaceholder("搜索包名称");
        await expect(searchInput).toBeVisible({ timeout: 30000 });
        await searchInput.fill(nonexistentKeyword);
        await searchInput.press("Enter");
        await page.waitForLoadState("networkidle");
        await page.waitForTimeout(1000);

        await expect(page.locator("tbody")).toContainText("暂无数据");
    });
});
