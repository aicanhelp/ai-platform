import { expect, test, type Locator } from "@playwright/test";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { newAuthorizedContext } from "@/utils/common";
import {
    openApplicationListPageByUrl,
    openRepositoryConfigPageByUrl,
    syncAllRepositories,
    waitForApplicationListReady,
    waitForRealApplicationTitles,
} from "./utils/marketplace";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

test.describe("应用市场 E2E", () => {
    test.beforeAll(async ({ browser }) => {
        const context = await newAuthorizedContext(browser);
        const page = await context.newPage();
        try {
            await openRepositoryConfigPageByUrl(page);
            await syncAllRepositories(page);
        } finally {
            await context.close();
        }
    });

    test("【Marketplace-001】应用市场-首页-应用概览-基础功能验证", async ({ page }) => {
        await openApplicationListPageByUrl(page);
        const searchInput = page.getByPlaceholder("搜索应用名称");
        const applicationItems = page.locator(".application_list .application_item");
        const applicationTitles = page.locator(".application_list .application_item .application_item_title");

        await expect(searchInput).toBeVisible({ timeout: 30000 });
        const initialTitles = await waitForRealApplicationTitles(applicationTitles);
        expect(initialTitles.length).toBeGreaterThan(0);

        const applicationCountText = await page.getByText(/共\d+个应用/).textContent();
        const expectedCount = Number(applicationCountText?.match(/\d+/)?.[0] || 0);
        const actualCount = await applicationItems.count();

        expect(expectedCount).toBeGreaterThan(0);
        expect(actualCount).toBe(expectedCount);

        for (let index = 0; index < actualCount; index++) {
            const currentItem = applicationItems.nth(index);
            const title = (await currentItem.locator(".application_item_title").textContent())?.trim() || "";
            const source = (await currentItem.locator(".application_item_header .application_item_text").textContent())?.trim() || "";
            const description = (await currentItem.locator(".application_item_text.clamp_3").textContent())?.trim() || "";
            const logo = currentItem.locator(".application_item_logo img");

            expect(title).not.toBe("");
            expect(source).not.toBe("");
            expect(description).not.toBe("");
            await expect(logo).toBeVisible();
            await expect(logo).toHaveAttribute("src", /.+/);
        }

        const extensionLabels = page.locator(".application_list .application_item .extension_label");
        const extensionLabelCount = await extensionLabels.count();
        expect(extensionLabelCount).toBeGreaterThan(0);

        for (let index = 0; index < extensionLabelCount; index++) {
            const currentExtensionLabel = extensionLabels.nth(index);
            const extensionIcon = currentExtensionLabel.locator("img");

            await expect(currentExtensionLabel).toContainText("扩展组件");
            await expect(extensionIcon).toBeVisible();
            await expect(extensionIcon).toHaveAttribute("src", /.+/);
        }
    });

    test("【Marketplace-002】应用市场-首页-市场管理按钮跳转-基础功能验证", async ({ page }) => {
        await openApplicationListPageByUrl(page);
        await page.getByRole("link", { name: "仓库配置" }).click();
        await page.waitForLoadState("networkidle");
        await page.waitForURL(/\/container_platform\/appMarket\/stash\/wareHouse$/, { timeout: 30000 });
        await expect(page.getByRole("tab", { name: "仓库配置" })).toBeVisible();
        await expect(page.getByRole("button", { name: "添加仓库" })).toBeVisible();
    });

    test("【Marketplace-003】应用市场-应用详情-信息展示-基础功能验证", async ({ page }) => {
        await openApplicationListPageByUrl(page);
        await waitForApplicationListReady(page);

        const targetApplication = page.locator(".application_list .application_item")
            .filter({ has: page.locator(".extension_label") })
            .filter({ has: page.locator(".application_item_header .application_item_text", { hasText: "官方仓库提供" }) })
            .first();

        await expect(targetApplication).toBeVisible();

        const outerApplicationName = (await targetApplication.locator(".application_item_title").textContent())?.trim() || "";
        const outerSupplierType = (await targetApplication.locator(".application_item_header .application_item_text").textContent())?.trim() || "";
        const outerDescription = (await targetApplication.locator(".application_item_text.clamp_3").textContent())?.trim() || "";
        const outerLabel = (await targetApplication.locator(".extension_label").textContent())?.trim() || "";

        expect(outerApplicationName).not.toBe("");
        expect(outerSupplierType).toBe("官方仓库提供");
        expect(outerDescription).not.toBe("");
        expect(outerLabel).toContain("扩展组件");

        await targetApplication.click();
        await page.waitForLoadState("networkidle");

        await expect(page).toHaveURL(/\/container_platform\/appMarket\/marketCategory\/ApplicationDetails\//);

        const detailHeader = page.locator(".detail_container .details_header");
        const detailContent = page.locator(".detail_container .application_details_content");
        const detailTagContainer = detailHeader.locator("div").filter({ hasText: /^扩展组件$/ }).last().locator("xpath=..");
        const detailTagIcon = detailTagContainer.locator("img");
        const detailInformationPane = page.locator(".container-platform-tabs-tabpane-active");

        const detailApplicationName = detailHeader.locator(".clamp_1").first();
        const detailVersion = detailHeader.locator(".container-platform-select-selection-item");
        const detailDescription = detailHeader.locator(".clamp_3").first();
        const detailRepositoryName = detailContent.locator("> div")
            .filter({ has: page.locator(".tips_color", { hasText: "仓库名称：" }) })
            .first()
            .locator(".clamp_1");
        const detailAddedTime = detailContent.locator("> div")
            .filter({ has: page.locator(".tips_color", { hasText: "添加时间：" }) })
            .first()
            .locator(".clamp_1");
        const detailApplicationVersion = detailContent.locator("> div")
            .filter({ has: page.locator(".tips_color", { hasText: "应用版本：" }) })
            .first()
            .locator(".clamp_1");
        const detailSupplierType = detailContent.locator("> div")
            .filter({ has: page.locator(".tips_color", { hasText: "供应商类型：" }) })
            .first()
            .locator(".clamp_1");

        await expect(detailApplicationName).toHaveText(outerApplicationName);
        await expect(detailDescription).toHaveText(outerDescription);
        await expect(detailSupplierType).toHaveText("官方仓库");
        await expect(detailVersion).not.toHaveText("");
        await expect(detailRepositoryName).not.toHaveText("");
        await expect(detailAddedTime).not.toHaveText("");
        await expect(detailApplicationVersion).not.toHaveText("");

        await expect(detailTagContainer).toContainText("扩展组件");
        await expect(detailTagIcon).toBeVisible();
        await expect(detailTagIcon).toHaveAttribute("src", /.+/);
        await expect.poll(async () => ((await detailInformationPane.innerText()) || "").trim().length).toBeGreaterThan(0);
    });

    test("【Marketplace-006】平台管理-仓库-仓库配置-仓库详情-详情展示-远端仓库-基础功能验证", async ({ page }) => {
        await openRepositoryConfigPageByUrl(page);

        const targetRepositoryRow = page.locator("tbody tr")
            .filter({ has: page.locator('a[href*="/stashDetails/openFuyao"]') })
            .first();
        const targetRepositoryLink = targetRepositoryRow.locator('a[href*="/stashDetails/openFuyao"]').first();
        const outerRepositoryName = (await targetRepositoryLink.textContent())?.trim() || "";
        const outerRepositoryUrl = (await targetRepositoryRow.locator("td").nth(1).textContent())?.trim() || "";

        expect(outerRepositoryName).toBe("openFuyao");
        expect(outerRepositoryUrl).not.toBe("");

        await targetRepositoryLink.click();
        await page.waitForLoadState("networkidle");

        await expect(page).toHaveURL(/\/container_platform\/appMarket\/stash\/wareHouse\/stashDetails\/openFuyao$/);

        const detailsContainer = page.locator(".stash_details_container");
        const detailsHeader = page.locator(".ware_house_detail_header");
        const detailsContent = page.locator(".details_content");
        const detailInfoRows = detailsContent.locator(':scope > div');
        const detailRepositoryType = detailInfoRows.filter({ hasText: "仓库类型：" }).first().locator(".clamp_1");
        const detailRepositoryUrl = detailInfoRows.filter({ hasText: "仓库地址：" }).first().locator(".clamp_1");
        const detailLastSyncTime = detailInfoRows.filter({ hasText: "上次同步时间：" }).first().locator(".clamp_1");
        const detailCreateTime = detailInfoRows.filter({ hasText: "创建时间：" }).first().locator(".clamp_1");

        await expect(detailsHeader).toContainText(outerRepositoryName);
        await expect(detailRepositoryUrl).toHaveText(outerRepositoryUrl);
        await expect(detailRepositoryType).not.toHaveText("");
        await expect(detailLastSyncTime).not.toHaveText("");
        await expect(detailCreateTime).not.toHaveText("");
        await expect(detailsContainer).toContainText("应用/扩展组件");

        const componentRows = page.locator(".container-platform-table-tbody tr");
        const componentCount = await componentRows.count();
        expect(componentCount).toBeGreaterThan(0);

        const firstComponentRow = componentRows.first();
        const firstComponentCells = firstComponentRow.locator("td");
        const firstComponentName = (await firstComponentCells.nth(0).textContent())?.trim() || "";
        const firstComponentType = (await firstComponentCells.nth(1).textContent())?.trim() || "";
        const firstComponentVersion = (await firstComponentCells.nth(2).textContent())?.trim() || "";
        const firstComponentCreateTime = (await firstComponentCells.nth(3).textContent())?.trim() || "";

        expect(firstComponentName).not.toBe("");
        expect(firstComponentType).not.toBe("");
        expect(["应用", "扩展组件"]).toContain(firstComponentType);
        expect(firstComponentVersion).not.toBe("");
        expect(firstComponentCreateTime).not.toBe("");
    });

    test("【Marketplace-007】平台管理-仓库-仓库配置(概览)-基础功能验证", async ({ page }) => {
        await openRepositoryConfigPageByUrl(page);

        const repositoryRows = page.locator("tbody tr");
        const repositoryCount = await repositoryRows.count();

        expect(repositoryCount).toBeGreaterThan(0);

        for (let index = 0; index < repositoryCount; index++) {
            const currentRow = repositoryRows.nth(index);
            const repositoryName = (await currentRow.locator("td").first().textContent())?.trim() || "";
            const repositoryUrl = (await currentRow.locator("td").nth(1).textContent())?.trim() || "";

            expect(repositoryName).not.toBe("");
            expect(repositoryUrl).not.toBe("");
        }
    });

    test("【Marketplace-008】平台管理-仓库-仓库配置(概览)-默认展示本地仓库-基础功能验证", async ({ page }) => {
        await openRepositoryConfigPageByUrl(page);

        const localRepositoryRow = page.locator("tbody tr").filter({ hasText: "local" }).first();
        const localRepositoryIcon = localRepositoryRow.locator("td").first().locator("img").first();
        const localRepositoryName = (await localRepositoryRow.locator("td").first().locator("a").textContent())?.trim() || "";
        const localRepositoryUrl = (await localRepositoryRow.locator("td").nth(1).textContent())?.trim() || "";

        await expect(localRepositoryRow).toBeVisible();
        await expect(localRepositoryIcon).toBeVisible();
        await expect(localRepositoryIcon).toHaveAttribute("src", /.+/);
        expect(localRepositoryName).toBe("local");
        expect(localRepositoryUrl).not.toBe("");
    });

    test("【Marketplace-009】平台管理-仓库-仓库配置(概览)-默认展示官方仓库-基础功能验证", async ({ page }) => {
        await openRepositoryConfigPageByUrl(page);

        const officialRepositoryRow = page.locator("tbody tr").filter({ hasText: "openFuyao" }).last();
        const officialRepositoryIcon = officialRepositoryRow.locator("td").first().locator("img").first();
        const officialRepositoryName = (await officialRepositoryRow.locator("td").first().locator("a").textContent())?.trim() || "";
        const officialRepositoryUrl = (await officialRepositoryRow.locator("td").nth(1).textContent())?.trim() || "";

        await expect(officialRepositoryRow).toBeVisible();
        await expect(officialRepositoryIcon).toBeVisible();
        await expect(officialRepositoryIcon).toHaveAttribute("src", /.+/);
        expect(officialRepositoryName).toBe("openFuyao");
        expect(officialRepositoryUrl).not.toBe("");
    });

    test("【Marketplace-028】应用市场-首页-排序-基础功能验证", async ({ page }) => {
        await openApplicationListPageByUrl(page);
        await expect(page.getByPlaceholder("搜索应用名称")).toBeVisible({ timeout: 30000 });

        const sortSelect = page.locator(".pod_search .container-platform-select").first();
        const applicationTitles = page.locator(".application_list .application_item .application_item_title");
        const applicationCards = page.locator(".application_list .application_item");

        const getCardOrder = async () => {
            return await applicationCards.evaluateAll((nodes) => nodes.map((node) => ({
                title: node.querySelector(".application_item_title")?.textContent?.trim() || "",
                source: node.querySelector(".application_item_header .application_item_text")?.textContent?.trim() || "",
            })));
        };

        const defaultSortText = ((await sortSelect.innerText()) || "").replace(/\s+/g, "");
        expect(defaultSortText).toContain("按名称排序");

        const defaultNameOrder = await waitForRealApplicationTitles(applicationTitles);
        const expectedNameOrder = [...defaultNameOrder].sort((left, right) =>
            left.localeCompare(right, undefined, { sensitivity: "base" }),
        );
        expect(defaultNameOrder).toEqual(expectedNameOrder);

        await sortSelect.click();
        await page.locator(".container-platform-select-dropdown .container-platform-select-item-option").nth(1).click();
        await page.waitForTimeout(2000);

        const timeSortText = ((await sortSelect.innerText()) || "").replace(/\s+/g, "");
        expect(timeSortText).toContain("按时间排序");

        const timeSortedCards = await getCardOrder();
        expect(timeSortedCards.length).toBeGreaterThan(0);
        const response = await page.evaluate(async () => {
            const result = await fetch("/clusters/host/rest/marketplace/v1beta1/helm-charts?page=1&limit=50");
            return await result.json();
        });
        const createdAtByTitle = new Map<string, number>(
            (response.data.items as Array<{
                metadata?: { name?: string; created?: string };
            }>).map((item) => [
                item.metadata?.name?.trim() || "",
                Date.parse(item.metadata?.created || ""),
            ]),
        );
        // 遍历后端返回顺序，用原始数据中的创建时间校验相邻项非递减（时间相同允许任意相对顺序）
        for (let index = 0; index < timeSortedCards.length - 1; index++) {
            const currentTitle = timeSortedCards[index].title;
            const nextTitle = timeSortedCards[index + 1].title;
            const currentTime = createdAtByTitle.get(currentTitle);
            const nextTime = createdAtByTitle.get(nextTitle);
            expect(currentTime, `获取「${currentTitle}」的创建时间失败`).toEqual(expect.any(Number));
            expect(nextTime, `获取「${nextTitle}」的创建时间失败`).toEqual(expect.any(Number));
            expect(currentTime!, `时间顺序 broken: ${currentTitle}(${currentTime}) > ${nextTitle}(${nextTime})`).toBeLessThanOrEqual(nextTime!);
        }
        expect(timeSortedCards.every(({ source }) => source.trim() !== "")).toBeTruthy();

        await sortSelect.click();
        await page.locator(".container-platform-select-dropdown .container-platform-select-item-option").nth(0).click();
        await page.waitForTimeout(2000);

        const nameSortText = ((await sortSelect.innerText()) || "").replace(/\s+/g, "");
        expect(nameSortText).toContain("按名称排序");

        const restoredNameOrder = (await applicationTitles.allTextContents()).map((title) => title.trim());
        expect(restoredNameOrder).toEqual(expectedNameOrder);
    });

    test("【Marketplace-029】应用市场-首页-搜索-基础功能验证", async ({ page }) => {
        await openApplicationListPageByUrl(page);
        await expect(page.getByPlaceholder("搜索应用名称")).toBeVisible({ timeout: 30000 });

        const searchInput = page.getByPlaceholder("搜索应用名称");
        const applicationCards = page.locator(".application_list .application_item");
        const applicationTitles = page.locator(".application_list .application_item .application_item_title");
        const initialTitles = await waitForRealApplicationTitles(applicationTitles);

        await expect(searchInput).toBeVisible();
        expect(initialTitles.length).toBeGreaterThan(0);

        await searchInput.fill("npu-operator");
        await page.waitForTimeout(2000);
        const exactSearchCount = await applicationCards.count();
        expect(exactSearchCount).toBeGreaterThan(0);
        const exactSearchTitles = (await applicationTitles.allTextContents()).map((title) => title.trim());
        expect(exactSearchTitles.every((title) => title === "npu-operator")).toBeTruthy();
        await searchInput.clear();
        await expect(searchInput).toHaveValue("");
        await expect.poll(async () => (await applicationTitles.allTextContents()).map((title) => title.trim())).toEqual(initialTitles);

        await searchInput.fill("npu");
        await page.waitForTimeout(2000);
        const partialSearchCount = await applicationCards.count();
        expect(partialSearchCount).toBeGreaterThan(0);
        const partialSearchTitles = (await applicationTitles.allTextContents()).map((title) => title.trim());
        expect(partialSearchTitles.every((title) => title.toLowerCase().includes("npu"))).toBeTruthy();
        await searchInput.clear();
        await expect(searchInput).toHaveValue("");
        await expect.poll(async () => (await applicationTitles.allTextContents()).map((title) => title.trim())).toEqual(initialTitles);

        await searchInput.fill("Npu-Operator");
        await page.waitForTimeout(2000);
        const caseInsensitiveSearchCount = await applicationCards.count();
        expect(caseInsensitiveSearchCount).toBeGreaterThan(0);
        const caseInsensitiveSearchTitles = (await applicationTitles.allTextContents()).map((title) => title.trim());
        expect(caseInsensitiveSearchTitles.every((title) => title.toLowerCase() === "npu-operator")).toBeTruthy();
    });

    test("【Marketplace-030】应用市场-应用详情-默认参数(YAML)-导出查找复制-基础功能验证", async ({ page }) => {
        const context = page.context();
        await context.grantPermissions(["clipboard-read", "clipboard-write"]);

        await openApplicationListPageByUrl(page);
        await waitForApplicationListReady(page);

        const targetApplication = page.locator(".application_list .application_item")
            .filter({ has: page.locator(".extension_label") })
            .first();
        const applicationName = (await targetApplication.locator(".application_item_title").textContent())?.trim() || "";

        await expect(targetApplication).toBeVisible();
        expect(applicationName).not.toBe("");

        await targetApplication.click();
        await page.waitForLoadState("networkidle");
        await expect(page).toHaveURL(/\/container_platform\/appMarket\/marketCategory\/ApplicationDetails\//);

        const detailTabs = page.locator("[role='tab']");
        await detailTabs.nth(1).click();
        await page.waitForTimeout(2000);

        const yamlContent = page.locator(".cm-content");
        await expect(yamlContent).toBeVisible();

        const visibleYamlText = ((await yamlContent.innerText()) || "")
            .replace(/\r\n/g, "\n")
            .trim();
        expect(visibleYamlText.length).toBeGreaterThan(0);

        const downloadPromise = page.waitForEvent("download");
        await page.locator(".defaultParameterExport").click();
        const download = await downloadPromise;
        const downloadPath = path.resolve(__dirname, `./downloads/${applicationName}-default-values.yaml`);
        await download.saveAs(downloadPath);

        const downloadedYamlText = fs.readFileSync(downloadPath, "utf8").replace(/\r\n/g, "\n").trim();
        expect(downloadedYamlText.length).toBeGreaterThan(0);
        if (fs.existsSync(downloadPath)) {
            fs.unlinkSync(downloadPath);
        }
        expect(fs.existsSync(downloadPath)).toBeFalsy();

        await page.locator(".defaultParameterCopy").click();
        await expect(page.getByText("复制成功！")).toBeVisible();

        await expect.poll(async () => {
            const clipboardText = await page.evaluate(async () => {
                return await navigator.clipboard.readText();
            });
            return clipboardText.replace(/\r\n/g, "\n").trim();
        }).toBe(downloadedYamlText);

        await context.clearPermissions();
    });

    test("【Marketplace-031】应用市场-应用详情-版本切换-基础功能验证", async ({ page }) => {
        await openApplicationListPageByUrl(page);
        await waitForApplicationListReady(page);

        const applicationCards = page.locator(".application_list .application_item");
        const applicationCount = await applicationCards.count();

        const detailTabs = page.locator("[role='tab']");
        const versionSelect = page.locator(".container-platform-select").nth(1);
        const yamlContent = page.locator(".cm-content");
        const versionOptions = page.locator(".container-platform-select-dropdown .container-platform-select-item-option");

        const normalizeText = (text: string) => (text || "").replace(/\s+/g, "");
        // 提取大版本系：如 "版本：26.9.0-rc.1" -> "26.9"，0.0.0-* 重打包/最新构建归为 0.0 系
        const versionSeriesOf = (normalizedText: string) => (normalizedText.match(/^版本：(\d+\.\d+)/) || [])[1] || "";
        const getVersionText = async () => normalizeText((await versionSelect.innerText()) || "");
        const getYamlText = async () => ((await yamlContent.innerText()) || "").replace(/\r\n/g, "\n").trim();
        // 下拉关闭状态下调用：打开下拉后按归一化文本全等匹配选项下标点击，并轮询等待切换完成
        const switchToVersion = async (targetVersionText: string) => {
            await versionSelect.click();
            await page.waitForTimeout(1000);
            const optionTexts = (await versionOptions.allTextContents()).map(normalizeText);
            const optionIndex = optionTexts.findIndex((text) => text === targetVersionText);
            expect(optionIndex, `下拉中未找到版本「${targetVersionText}」`).toBeGreaterThanOrEqual(0);
            await versionOptions.nth(optionIndex).click();
            // 选中后 select 文本会呈现为"版本：X版本：X"的重复展示，用包含匹配判断切换完成
            await expect.poll(getVersionText, { timeout: 30000 }).toContain(targetVersionText);
            await page.waitForTimeout(1000);
            await expect(yamlContent).toBeVisible();
        };

        let multiVersionApplicationFound = false;
        let initialVersionText = "";
        let targetVersionText = "";

        for (let index = 0; index < applicationCount; index++) {
            const currentApplication = applicationCards.nth(index);
            await expect(currentApplication).toBeVisible();

            await currentApplication.click();
            await page.waitForLoadState("networkidle");
            await expect(page).toHaveURL(/\/container_platform\/appMarket\/marketCategory\/ApplicationDetails\//);

            await detailTabs.nth(1).click();
            await expect(yamlContent).toBeVisible({ timeout: 30000 });

            initialVersionText = await getVersionText();
            const initialYamlText = await getYamlText();
            expect(initialVersionText).toContain("版本：");

            await versionSelect.click();
            await page.waitForTimeout(1000);
            const optionTexts = (await versionOptions.allTextContents()).map(normalizeText);
            await page.keyboard.press("Escape");

            // 只选与初始版本跨大版本系的候选，0.0.0-* 这类重打包版本排到最后作为兜底
            const initialSeries = versionSeriesOf(initialVersionText);
            const crossSeriesCandidates = optionTexts.filter((text) => {
                const series = versionSeriesOf(text);
                return series !== "" && series !== initialSeries;
            }).sort((left, right) => Number(left.startsWith("版本：0.0.0")) - Number(right.startsWith("版本：0.0.0")));

            for (const candidate of crossSeriesCandidates) {
                await switchToVersion(candidate);
                if ((await getYamlText()) !== initialYamlText) {
                    multiVersionApplicationFound = true;
                    targetVersionText = candidate;
                    break;
                }
            }

            if (multiVersionApplicationFound) {
                // 切回初始版本，保证最终断言从初始状态开始
                await switchToVersion(initialVersionText);
                break;
            }

            await page.keyboard.press("Escape");
            await openApplicationListPageByUrl(page);
            await waitForApplicationListReady(page);
        }

        expect(multiVersionApplicationFound, "所有应用的跨大版本默认参数均相同，未找到可校验版本切换的样本").toBeTruthy();

        await detailTabs.nth(1).click();
        await expect(yamlContent).toBeVisible({ timeout: 30000 });

        const finalInitialVersionText = await getVersionText();
        const finalInitialYamlText = await getYamlText();
        // 切换操作后 select 文本为重复展示，与首次读取的单行文本不逐字相等，用包含匹配确认已切回
        expect(finalInitialVersionText).toContain(initialVersionText);

        await switchToVersion(targetVersionText);

        const switchedVersionText = await getVersionText();
        const switchedYamlText = await getYamlText();

        expect(switchedVersionText).toContain(targetVersionText);
        expect(switchedVersionText).not.toBe(initialVersionText);
        expect(switchedYamlText.length).toBeGreaterThan(0);
        expect(switchedYamlText).not.toBe(finalInitialYamlText);
    });

    test("【Marketplace-039】平台管理-仓库-仓库配置-仓库详情-详情展示-本地仓库-基础功能验证", async ({ page }) => {
        await openRepositoryConfigPageByUrl(page);

        const localRepositoryRow = page.locator("tbody tr").filter({ hasText: "local" }).first();
        const localRepositoryLink = localRepositoryRow.locator("td").first().locator("a");
        const outerRepositoryName = (await localRepositoryLink.textContent())?.trim() || "";
        const outerRepositoryUrl = (await localRepositoryRow.locator("td").nth(1).textContent())?.trim() || "";

        expect(outerRepositoryName).toBe("local");
        expect(outerRepositoryUrl).not.toBe("");

        await localRepositoryLink.click();
        await page.waitForLoadState("networkidle");

        await expect(page).toHaveURL(/\/container_platform\/appMarket\/stash\/wareHouse\/stashDetails\/local$/);

        const detailsContainer = page.locator(".stash_details_container");
        const detailsHeader = page.locator(".ware_house_detail_header");
        const detailsContent = page.locator(".details_content");
        const detailInfoRows = detailsContent.locator(":scope > div");
        const detailRepositoryType = detailInfoRows.filter({ hasText: "仓库类型：" }).first().locator(".clamp_1");
        const detailRepositoryUrl = detailInfoRows.filter({ hasText: "仓库地址：" }).first().locator(".clamp_1");
        const detailLastSyncTime = detailInfoRows.filter({ hasText: "上次同步时间：" }).first().locator(".clamp_1");
        const detailCreateTime = detailInfoRows.filter({ hasText: "创建时间：" }).first().locator(".clamp_1");

        await expect(detailsHeader).toContainText(outerRepositoryName);
        await expect(detailRepositoryType).toHaveText("本地仓库");
        await expect(detailRepositoryUrl).toHaveText(outerRepositoryUrl);
        await expect(detailLastSyncTime).not.toHaveText("");
        await expect(detailCreateTime).not.toHaveText("");
        await expect(detailsContainer).toContainText("应用/扩展组件");
        const tableHeader = page.locator(".container-platform-table-thead");
        await expect(tableHeader).toContainText("名称");
        await expect(tableHeader).toContainText("类型");
        await expect(tableHeader).toContainText("版本");
        await expect(tableHeader).toContainText("创建时间");

        const componentRows = page.locator(".container-platform-table-tbody tr");
        const componentCount = await componentRows.count();
        expect(componentCount).toBeGreaterThan(0);
        await expect(page.getByText("暂无数据")).toBeVisible();
        await expect(page.getByText("共0条")).toBeVisible();
    });

    test("【Marketplace-040】平台管理-仓库-仓库配置-仓库详情-详情展示-官方仓库-基础功能验证", async ({ page }) => {
        await openRepositoryConfigPageByUrl(page);

        const officialRepositoryRow = page.locator("tbody tr").filter({ hasText: /openFuyao/ });
        const officialRepositoryLink = officialRepositoryRow.locator("td").first().locator("a");
        const outerRepositoryName = (await officialRepositoryLink.textContent())?.trim() || "";
        const outerRepositoryUrl = (await officialRepositoryRow.locator("td").nth(1).textContent())?.trim() || "";

        expect(outerRepositoryName).toBe("openFuyao");
        expect(outerRepositoryUrl).not.toBe("");

        await officialRepositoryLink.click();
        await page.waitForLoadState("networkidle");

        await expect(page).toHaveURL(/\/container_platform\/appMarket\/stash\/wareHouse\/stashDetails\/openFuyao$/);

        const detailsContainer = page.locator(".stash_details_container");
        const detailsHeader = page.locator(".ware_house_detail_header");
        const detailsContent = page.locator(".details_content");
        const detailInfoRows = detailsContent.locator(":scope > div");
        const detailRepositoryType = detailInfoRows.filter({ hasText: "仓库类型：" }).first().locator(".clamp_1");
        const detailRepositoryUrl = detailInfoRows.filter({ hasText: "仓库地址：" }).first().locator(".clamp_1");
        const detailLastSyncTime = detailInfoRows.filter({ hasText: "上次同步时间：" }).first().locator(".clamp_1");
        const detailCreateTime = detailInfoRows.filter({ hasText: "创建时间：" }).first().locator(".clamp_1");

        await expect(detailsHeader).toContainText(outerRepositoryName);
        await expect(detailRepositoryType).toHaveText("官方仓库");
        await expect(detailRepositoryUrl).toHaveText(outerRepositoryUrl);
        await expect(detailLastSyncTime).not.toHaveText("");
        await expect(detailCreateTime).not.toHaveText("");
        await expect(detailsContainer).toContainText("应用/扩展组件");
        const tableHeader = page.locator(".container-platform-table-thead");
        await expect(tableHeader).toContainText("名称");
        await expect(tableHeader).toContainText("类型");
        await expect(tableHeader).toContainText("版本");
        await expect(tableHeader).toContainText("创建时间");

        const componentRows = page.locator(".container-platform-table-tbody tr");
        const componentCount = await componentRows.count();
        expect(componentCount).toBeGreaterThan(0);
        const componentTotalText = ((await page.getByText(/共\d+条/).first().textContent()) || "").trim();
        const componentTotalCount = Number(componentTotalText.match(/\d+/)?.[0] || 0);
        expect(componentTotalCount).toBeGreaterThan(0);

        const firstComponentRow = componentRows.first();
        const firstComponentCells = firstComponentRow.locator("td");
        const firstComponentName = (await firstComponentCells.nth(0).textContent())?.trim() || "";
        const firstComponentType = (await firstComponentCells.nth(1).textContent())?.trim() || "";
        const firstComponentVersion = (await firstComponentCells.nth(2).textContent())?.trim() || "";
        const firstComponentCreateTime = (await firstComponentCells.nth(3).textContent())?.trim() || "";

        expect(firstComponentName).not.toBe("");
        expect(firstComponentType).not.toBe("");
        expect(["应用", "扩展组件"]).toContain(firstComponentType);
        expect(firstComponentVersion).not.toBe("");
        expect(firstComponentCreateTime).not.toBe("");
    });

    test("【Marketplace-044】平台管理-仓库-仓库配置(概览)-搜索-基础功能验证", async ({ page }) => {
        await openRepositoryConfigPageByUrl(page);

        const searchInput = page.getByPlaceholder("请输入仓库");
        const repositoryRows = page.locator("tbody tr");
        const initialRepositoryRows = await repositoryRows.evaluateAll((rows) => rows.map((row) => ({
            name: row.querySelector("td:nth-child(1) a")?.textContent?.trim() || "",
            url: row.querySelector("td:nth-child(2)")?.textContent?.trim() || "",
        })));

        await expect(searchInput).toBeVisible();
        expect(initialRepositoryRows.length).toBeGreaterThan(0);

        await searchInput.fill("local");
        await page.waitForTimeout(2000);
        const exactSearchRows = await repositoryRows.evaluateAll((rows) => rows.map((row) => ({
            name: row.querySelector("td:nth-child(1) a")?.textContent?.trim() || "",
            url: row.querySelector("td:nth-child(2)")?.textContent?.trim() || "",
        })));
        expect(exactSearchRows.length).toBeGreaterThan(0);
        expect(exactSearchRows.every((row) => row.name === "local")).toBeTruthy();

        await searchInput.clear();
        await expect(searchInput).toHaveValue("");
        await expect.poll(async () => await repositoryRows.evaluateAll((rows) => rows.map((row) => ({
            name: row.querySelector("td:nth-child(1) a")?.textContent?.trim() || "",
            url: row.querySelector("td:nth-child(2)")?.textContent?.trim() || "",
        })))).toEqual(initialRepositoryRows);

        await searchInput.fill("o");
        await page.waitForTimeout(2000);
        const partialSearchRows = await repositoryRows.evaluateAll((rows) => rows.map((row) => ({
            name: row.querySelector("td:nth-child(1) a")?.textContent?.trim() || "",
            url: row.querySelector("td:nth-child(2)")?.textContent?.trim() || "",
        })));
        expect(partialSearchRows.length).toBeGreaterThan(1);
        expect(partialSearchRows.every((row) => row.name.toLowerCase().includes("o"))).toBeTruthy();

        await searchInput.clear();
        await expect(searchInput).toHaveValue("");
        await expect.poll(async () => await repositoryRows.evaluateAll((rows) => rows.map((row) => ({
            name: row.querySelector("td:nth-child(1) a")?.textContent?.trim() || "",
            url: row.querySelector("td:nth-child(2)")?.textContent?.trim() || "",
        })))).toEqual(initialRepositoryRows);

        await searchInput.fill("LOcal");
        await page.waitForTimeout(2000);
        const mixedCaseSearchRows = await repositoryRows.evaluateAll((rows) => rows.map((row) => ({
            name: row.querySelector("td:nth-child(1) a")?.textContent?.trim() || "",
            url: row.querySelector("td:nth-child(2)")?.textContent?.trim() || "",
        })));
        expect(mixedCaseSearchRows.length).toBeGreaterThan(0);
        expect(mixedCaseSearchRows.every((row) => row.name.toLowerCase() === "local")).toBeTruthy();
    });

    test("【Marketplace-047】应用市场-首页-搜索结合筛选-构造异常-先筛选再搜索", async ({ page }) => {
        await openApplicationListPageByUrl(page);
        await expect(page.getByPlaceholder("搜索应用名称")).toBeVisible({ timeout: 30000 });

        const searchInput = page.getByPlaceholder("搜索应用名称");
        const applicationItems = page.locator(".application_list .application_item");
        const applicationTitles = page.locator(".application_list .application_item .application_item_title");
        const computeFilter = page.locator("label").filter({ hasText: "计算" });
        const initialTitles = await waitForRealApplicationTitles(applicationTitles);

        await expect(searchInput).toBeVisible();
        expect(initialTitles.length).toBeGreaterThan(0);

        await computeFilter.click();
        await page.waitForTimeout(2000);

        const filteredTitles = (await applicationTitles.allTextContents()).map((title) => title.trim());
        expect(filteredTitles.length).toBeGreaterThan(0);

        const existingFilteredTitle = filteredTitles[0];
        expect(existingFilteredTitle).not.toBe("");

        await searchInput.fill(existingFilteredTitle);
        await page.waitForTimeout(2000);

        const successfulSearchTitles = (await applicationTitles.allTextContents()).map((title) => title.trim());
        expect(successfulSearchTitles.length).toBeGreaterThan(0);
        expect(successfulSearchTitles.every((title) => title === existingFilteredTitle)).toBeTruthy();

        await searchInput.clear();
        await expect(searchInput).toHaveValue("");
        await expect.poll(async () => (await applicationTitles.allTextContents()).map((title) => title.trim())).toEqual(filteredTitles);

        const nonFilteredTitle = initialTitles.find((title) => !filteredTitles.includes(title)) || "";
        expect(nonFilteredTitle).not.toBe("");

        await searchInput.fill(nonFilteredTitle);
        await page.waitForTimeout(2000);

        const failedSearchCount = await applicationItems.count();
        expect(failedSearchCount).toBe(0);
        await expect(page.locator(".pod_search").getByText("共0个应用")).toBeVisible();
        await expect(page.getByText("暂无内容")).toBeVisible();
        await expect(page.locator(".container-platform-pagination")).toContainText("共0条");
    });

    test("【Marketplace-048】应用市场-首页-搜索结合筛选-构造异常-先搜索再筛选", async ({ page }) => {
        await openApplicationListPageByUrl(page);
        await expect(page.getByPlaceholder("搜索应用名称")).toBeVisible({ timeout: 30000 });

        const searchInput = page.getByPlaceholder("搜索应用名称");
        const applicationItems = page.locator(".application_list .application_item");
        const applicationTitles = page.locator(".application_list .application_item .application_item_title");
        const computeFilter = page.locator("label").filter({ hasText: "计算" });
            const initialTitles = await waitForRealApplicationTitles(applicationTitles);

        await expect(searchInput).toBeVisible();
        expect(initialTitles.length).toBeGreaterThan(0);

        await computeFilter.click();
        await page.waitForTimeout(2000);
        const computeFilteredTitles = (await applicationTitles.allTextContents()).map((title) => title.trim());
        expect(computeFilteredTitles.length).toBeGreaterThan(0);

        await computeFilter.click();
        await page.waitForTimeout(2000);
        await expect(searchInput).toHaveValue("");
        await expect.poll(async () => (await applicationTitles.allTextContents()).map((title) => title.trim())).toEqual(initialTitles);

        await searchInput.fill("package");
        await page.waitForTimeout(2000);

        const searchedResultCount = await applicationItems.count();
        const searchedTitles = (await applicationTitles.allTextContents()).map((title) => title.trim());
        expect(searchedResultCount).toBeGreaterThan(1);
        expect(searchedTitles.every((title) => title.toLowerCase().includes("package"))).toBeTruthy();

        await computeFilter.click();
        await page.waitForTimeout(2000);

        const filteredResultCount = await applicationItems.count();
        const filteredTitles = (await applicationTitles.allTextContents()).map((title) => title.trim());

        expect(filteredResultCount).toBeGreaterThan(0);
        expect(filteredResultCount).toBeLessThan(searchedResultCount);
        expect(filteredTitles.every((title) => title.toLowerCase().includes("package"))).toBeTruthy();
        expect(filteredTitles.every((title) => computeFilteredTitles.includes(title))).toBeTruthy();
    });

    test("【Marketplace-049】应用市场-首页-筛选-单项筛选-每次重置-基础功能验证", async ({ page }) => {
        await openApplicationListPageByUrl(page);
        await expect(page.getByPlaceholder("搜索应用名称")).toBeVisible({ timeout: 30000 });

        const powerEngineHelpIcon = page.locator("[aria-label='question-circle']").nth(1);
        const extensionHelpIcon = page.locator("[aria-label='question-circle']").nth(2);
        const resetIcon = page.locator("[aria-label='clear']").first();
        const applicationTitles = page.locator(".application_list .application_item .application_item_title");
        const applicationCards = page.locator(".application_list .application_item");
        const getVisibleCards = async () => {
            return await applicationCards.evaluateAll((nodes) => nodes.map((node) => ({
                title: node.querySelector(".application_item_title")?.textContent?.trim() || "",
                source: node.querySelector(".application_item_header .application_item_text")?.textContent?.trim() || "",
            })));
        };
        const filterLabels = (await page.locator("label.container-platform-checkbox-wrapper").allTextContents())
            .map((text) => text.replace(/\s+/g, " ").trim())
            .filter(Boolean);
        const initialTitles = await waitForRealApplicationTitles(applicationTitles);
        const getApplicationTotalCount = async () => {
            const totalText = ((await page.getByText(/共\d+个应用/).first().innerText()) || "").trim();
            return Number(totalText.match(/\d+/)?.[0] || 0);
        };
        const initialTotalCount = await getApplicationTotalCount();

        await powerEngineHelpIcon.hover();
        await expect(page.getByText("通过Helm Chart快速部署算力优化加速应用，释放最大算力潜能。")).toBeVisible();
        await extensionHelpIcon.hover();
        await expect(page.getByText("通过Helm Chart快速部署平台扩展功能，体验扩展组件的无限可能。")).toBeVisible();
        expect(filterLabels.length).toBeGreaterThan(0);

        for (const labelText of filterLabels) {
            const currentFilterLabel = page.locator("label.container-platform-checkbox-wrapper").filter({ hasText: labelText }).first();
            const currentFilterInput = currentFilterLabel.locator("input");

            await currentFilterLabel.click();
            await page.waitForTimeout(1500);
            await expect(currentFilterInput).toBeChecked();

            const filteredTitles = (await applicationTitles.allTextContents()).map((title) => title.trim());

            if (filteredTitles.length > 0) {
                const filteredTotalCount = await getApplicationTotalCount();
                if (labelText === "openFuyao") {
                    const filteredCards = await getVisibleCards();
                    expect(filteredCards.every((card) => card.source.includes("官方仓库"))).toBeTruthy();
                } else {
                    expect(filteredTotalCount).toBeLessThan(initialTotalCount);
                }
            } else {
                await expect(page.locator(".pod_search").getByText("共0个应用")).toBeVisible();
                await expect(page.getByText("暂无内容")).toBeVisible();
            }

            await resetIcon.click();
            await page.waitForTimeout(1500);
            await expect(currentFilterInput).not.toBeChecked();
            await expect.poll(async () => (await applicationTitles.allTextContents()).map((title) => title.trim())).toEqual(initialTitles);
        }
    });

    test("【Marketplace-050】应用市场-首页-筛选-组合条件-同类型-每次重置-基础功能验证", async ({ page }) => {
        await openApplicationListPageByUrl(page);
        await waitForApplicationListReady(page);

        const resetIcon = page.locator("[aria-label='clear']").first();
        const applicationCards = page.locator(".application_list .application_item");
        const getVisibleCards = async () => {
            return await applicationCards.evaluateAll((nodes) => nodes.map((node) => ({
                title: node.querySelector(".application_item_title")?.textContent?.trim() || "",
                source: node.querySelector(".application_item_header .application_item_text")?.textContent?.trim() || "",
            })));
        };
        const toSortedCardKeys = (cards: Array<{ title: string; source: string }>) => cards
            .map((card) => `${card.title}__${card.source}`)
            .sort((left, right) => left.localeCompare(right));
        const initialCards = await getVisibleCards();
        const initialCardKeys = toSortedCardKeys(initialCards);
        const scenarioFilterPairs: Array<[string, string]> = [
            ["人工智能/机器学习", "计算"],
            ["人工智能/机器学习", "日志"],
            ["计算", "监控"],
            ["计算", "日志"],
            ["监控", "日志"],
            ["可观测性", "监控"],
        ];

        const clickFilter = async (labelText: string) => {
            const filterLabel = page.locator("label.container-platform-checkbox-wrapper").filter({ hasText: labelText }).first();
            const filterInput = filterLabel.locator("input");
            await filterLabel.click();
            await page.waitForTimeout(1500);
            await expect(filterInput).toBeChecked();
            return filterInput;
        };

        const resetFilters = async () => {
            await resetIcon.click();
            await page.waitForTimeout(1500);
            await expect.poll(async () => toSortedCardKeys(await getVisibleCards())).toEqual(initialCardKeys);
        };

        const singleFilterResults = new Map<string, Array<{ title: string; source: string }>>();
        for (const labelText of [...new Set(scenarioFilterPairs.flat())]) {
            await clickFilter(labelText);
            const currentCards = await getVisibleCards();
            singleFilterResults.set(labelText, currentCards);
            await resetFilters();
        }

        for (const [firstLabel, secondLabel] of scenarioFilterPairs) {
            const firstFilterInput = await clickFilter(firstLabel);
            const secondFilterInput = await clickFilter(secondLabel);

            const combinedCards = await getVisibleCards();
            const expectedCards = initialCards.filter((card) =>
                (singleFilterResults.get(firstLabel) || []).some((item) => item.title === card.title && item.source === card.source)
                || (singleFilterResults.get(secondLabel) || []).some((item) => item.title === card.title && item.source === card.source),
            );

            expect(toSortedCardKeys(combinedCards)).toEqual(toSortedCardKeys(expectedCards));

            await resetFilters();
            await expect(firstFilterInput).not.toBeChecked();
            await expect(secondFilterInput).not.toBeChecked();
        }
    });

    test("【Marketplace-051】应用市场-首页-筛选-组合条件-不同类型-每次重置-基础功能验证", async ({ page }) => {
        await openApplicationListPageByUrl(page);
        await expect(page.getByPlaceholder("搜索应用名称")).toBeVisible({ timeout: 30000 });

        const resetIcon = page.locator("[aria-label='clear']").first();
        const applicationTitles = page.locator(".application_list .application_item .application_item_title");
        const applicationCards = page.locator(".application_list .application_item");
        const getVisibleCards = async () => {
            return await applicationCards.evaluateAll((nodes) => nodes.map((node) => ({
                title: node.querySelector(".application_item_title")?.textContent?.trim() || "",
                source: node.querySelector(".application_item_header .application_item_text")?.textContent?.trim() || "",
            })));
        };
        const toSortedCardKeys = (cards: Array<{ title: string; source: string }>) => cards
            .map((card) => `${card.title}__${card.source}`)
            .sort((left, right) => left.localeCompare(right));
        const initialTitles = await waitForRealApplicationTitles(applicationTitles);
        const initialCardKeys = toSortedCardKeys(await getVisibleCards());
        const scenarioFilterPairs: Array<[string, string]> = [
            ["扩展组件", "计算"],
            ["扩展组件", "日志"],
            ["扩展组件", "监控"],
            ["扩展组件", "openFuyao"],
            ["扩展组件", "人工智能/机器学习"],
            ["算力引擎插件", "安全"],
        ];

        const clickFilter = async (labelText: string) => {
            const filterLabel = page.locator("label.container-platform-checkbox-wrapper").filter({ hasText: labelText }).first();
            const filterInput = filterLabel.locator("input");
            await filterLabel.click();
            await page.waitForTimeout(1500);
            await expect(filterInput).toBeChecked();
            return filterInput;
        };

        const singleFilterResults = new Map<string, Array<{ title: string; source: string }>>();
        for (const labelText of [...new Set(scenarioFilterPairs.flat())]) {
            await clickFilter(labelText);
            const currentCards = await getVisibleCards();
            singleFilterResults.set(labelText, currentCards);
            await resetIcon.click();
            await page.waitForTimeout(1500);
            await expect.poll(async () => toSortedCardKeys(await getVisibleCards())).toEqual(initialCardKeys);
        }

        for (const [firstLabel, secondLabel] of scenarioFilterPairs) {
            const firstFilterInput = await clickFilter(firstLabel);
            const secondFilterInput = await clickFilter(secondLabel);

            const filteredTitles = (await applicationTitles.allTextContents()).map((title) => title.trim());
            const filteredCards = await getVisibleCards();
            const expectedCards = (singleFilterResults.get(firstLabel) || []).filter((card) =>
                (singleFilterResults.get(secondLabel) || []).some((item) => item.title === card.title && item.source === card.source),
            );

            if (expectedCards.length > 0) {
                expect(toSortedCardKeys(filteredCards)).toEqual(toSortedCardKeys(expectedCards));
            } else {
                await expect(page.locator(".pod_search").getByText("共0个应用")).toBeVisible();
                await expect(page.getByText("暂无内容")).toBeVisible();
                expect(filteredTitles).toEqual([]);
            }

            await resetIcon.click();
            await page.waitForTimeout(1500);
            await expect(firstFilterInput).not.toBeChecked();
            await expect(secondFilterInput).not.toBeChecked();
            await expect.poll(async () => (await applicationTitles.allTextContents()).map((title) => title.trim())).toEqual(initialTitles);
        }
    });

    test("【Marketplace-052】应用市场-首页-筛选-单项筛选-连续筛选不重置", async ({ page }) => {
        await openApplicationListPageByUrl(page);
        await expect(page.getByPlaceholder("搜索应用名称")).toBeVisible({ timeout: 30000 });

        const applicationTitles = page.locator(".application_list .application_item .application_item_title");
        const filterLabels = (await page.locator("label.container-platform-checkbox-wrapper").allTextContents())
            .map((text) => text.replace(/\s+/g, " ").trim())
            .filter(Boolean);
        const selectedLabels = filterLabels.slice(0, 6);
        const initialTitles = await waitForRealApplicationTitles(applicationTitles);

        expect(selectedLabels.length).toBe(6);

        const singleFilterResults = new Map<string, string[]>();
        for (const labelText of selectedLabels) {
            const currentFilterLabel = page.locator("label.container-platform-checkbox-wrapper").filter({ hasText: labelText }).first();
            const currentFilterInput = currentFilterLabel.locator("input");

            await currentFilterLabel.click();
            await page.waitForTimeout(1500);
            await expect(currentFilterInput).toBeChecked();

            const filteredTitles = (await applicationTitles.allTextContents()).map((title) => title.trim());
            singleFilterResults.set(labelText, filteredTitles);

            if (filteredTitles.length === 0) {
                await expect(page.locator(".pod_search").getByText("共0个应用")).toBeVisible();
                await expect(page.getByText("暂无内容")).toBeVisible();
            }

            await currentFilterLabel.click();
            await page.waitForTimeout(1500);
            await expect(currentFilterInput).not.toBeChecked();
            await expect.poll(async () => (await applicationTitles.allTextContents()).map((title) => title.trim())).toEqual(initialTitles);
        }

        const applyAndValidateSingleFilter = async (labelText: string) => {
            const currentFilterLabel = page.locator("label.container-platform-checkbox-wrapper").filter({ hasText: labelText }).first();
            const currentFilterInput = currentFilterLabel.locator("input");

            await currentFilterLabel.click();
            await page.waitForTimeout(1500);
            await expect(currentFilterInput).toBeChecked();

            const filteredTitles = (await applicationTitles.allTextContents()).map((title) => title.trim());
            const expectedTitles = singleFilterResults.get(labelText) || [];

            if (expectedTitles.length > 0) {
                expect(filteredTitles).toEqual(expectedTitles);
            } else {
                await expect(page.locator(".pod_search").getByText("共0个应用")).toBeVisible();
                await expect(page.getByText("暂无内容")).toBeVisible();
                expect(filteredTitles).toEqual([]);
            }

            return { currentFilterLabel, currentFilterInput };
        };

        let previousFilter = await applyAndValidateSingleFilter(selectedLabels[0]);
        for (const labelText of selectedLabels.slice(1)) {
            await previousFilter.currentFilterLabel.click();
            await page.waitForTimeout(1500);
            await expect(previousFilter.currentFilterInput).not.toBeChecked();
            await expect.poll(async () => (await applicationTitles.allTextContents()).map((title) => title.trim())).toEqual(initialTitles);

            previousFilter = await applyAndValidateSingleFilter(labelText);
        }
    });

    test("【Marketplace-053】应用市场-首页-筛选-组合筛选-连续筛选不重置", async ({ page }) => {
        await openApplicationListPageByUrl(page);
        await expect(page.getByPlaceholder("搜索应用名称")).toBeVisible({ timeout: 30000 });

        const applicationTitles = page.locator(".application_list .application_item .application_item_title");
        const applicationCards = page.locator(".application_list .application_item");
        const getVisibleCards = async () => {
            return await applicationCards.evaluateAll((nodes) => nodes.map((node) => ({
                title: node.querySelector(".application_item_title")?.textContent?.trim() || "",
                source: node.querySelector(".application_item_header .application_item_text")?.textContent?.trim() || "",
            })));
        };
        const toSortedCardKeys = (cards: Array<{ title: string; source: string }>) => cards
            .map((card) => `${card.title}__${card.source}`)
            .sort((left, right) => left.localeCompare(right));
        const initialTitles = await waitForRealApplicationTitles(applicationTitles);
        const scenarioFilterPairs: Array<[string, string]> = [
            ["扩展组件", "计算"],
            ["扩展组件", "日志"],
            ["扩展组件", "监控"],
            ["扩展组件", "openFuyao"],
            ["扩展组件", "人工智能/机器学习"],
            ["算力引擎插件", "安全"],
        ];

        const clickFilter = async (labelText: string) => {
            const filterLabel = page.locator("label.container-platform-checkbox-wrapper").filter({ hasText: labelText }).first();
            const filterInput = filterLabel.locator("input");
            await filterLabel.click();
            await page.waitForTimeout(1500);
            await expect(filterInput).toBeChecked();
            return { filterLabel, filterInput };
        };

        const singleFilterResults = new Map<string, Array<{ title: string; source: string }>>();
        for (const labelText of [...new Set(scenarioFilterPairs.flat())]) {
            const { filterLabel, filterInput } = await clickFilter(labelText);
            singleFilterResults.set(labelText, await getVisibleCards());

            await filterLabel.click();
            await page.waitForTimeout(1500);
            await expect(filterInput).not.toBeChecked();
            await expect.poll(async () => (await applicationTitles.allTextContents()).map((title) => title.trim())).toEqual(initialTitles);
        }

        const validatePairResult = async (pair: [string, string]) => {
            const filteredTitles = (await applicationTitles.allTextContents()).map((title) => title.trim());
            const filteredCards = await getVisibleCards();
            const expectedCards = (singleFilterResults.get(pair[0]) || []).filter((card) =>
                (singleFilterResults.get(pair[1]) || []).some((item) => item.title === card.title && item.source === card.source),
            );

            if (expectedCards.length > 0) {
                expect(toSortedCardKeys(filteredCards)).toEqual(toSortedCardKeys(expectedCards));
            } else {
                await expect(page.locator(".pod_search").getByText("共0个应用")).toBeVisible();
                await expect(page.getByText("暂无内容")).toBeVisible();
                expect(filteredTitles).toEqual([]);
            }
        };

        const validateSingleResult = async (labelText: string) => {
            const filteredTitles = (await applicationTitles.allTextContents()).map((title) => title.trim());
            const filteredCards = await getVisibleCards();
            const expectedCards = singleFilterResults.get(labelText) || [];

            if (expectedCards.length > 0) {
                expect(toSortedCardKeys(filteredCards)).toEqual(toSortedCardKeys(expectedCards));
            } else {
                await expect(page.locator(".pod_search").getByText("共0个应用")).toBeVisible();
                await expect(page.getByText("暂无内容")).toBeVisible();
                expect(filteredTitles).toEqual([]);
            }
        };

        let activeFilters = new Map<string, { filterLabel: Locator; filterInput: Locator }>();
        for (const labelText of scenarioFilterPairs[0]) {
            const currentFilter = await clickFilter(labelText);
            activeFilters.set(labelText, currentFilter);
        }
        await validatePairResult(scenarioFilterPairs[0]);

        for (const nextPair of scenarioFilterPairs.slice(1)) {
            const nextFilterSet = new Set(nextPair);

            for (const [activeLabel, activeFilter] of [...activeFilters.entries()]) {
                if (!nextFilterSet.has(activeLabel)) {
                    await activeFilter.filterLabel.click();
                    await page.waitForTimeout(1500);
                    await expect(activeFilter.filterInput).not.toBeChecked();
                    activeFilters.delete(activeLabel);
                    if (activeFilters.size === 0) {
                        await expect.poll(async () => (await applicationTitles.allTextContents()).map((title) => title.trim())).toEqual(initialTitles);
                    } else if (activeFilters.size === 1) {
                        await validateSingleResult([...activeFilters.keys()][0]);
                    }
                }
            }

            for (const nextLabel of nextPair) {
                if (!activeFilters.has(nextLabel)) {
                    const nextFilter = await clickFilter(nextLabel);
                    activeFilters.set(nextLabel, nextFilter);
                }
            }

            await validatePairResult(nextPair);
        }
    });

    test("【Marketplace-054】应用市场-首页-搜索-构造异常-不存在的应用名称-汉字", async ({ page }) => {
        await openApplicationListPageByUrl(page);
        await expect(page.getByPlaceholder("搜索应用名称")).toBeVisible({ timeout: 30000 });

        const searchInput = page.getByPlaceholder("搜索应用名称");
        const applicationTitles = page.locator(".application_list .application_item .application_item_title");
        const initialTitles = await waitForRealApplicationTitles(applicationTitles);
        const nonexistentKeyword = "不存在的应用名称123456";
        const chineseKeyword = "中";

        await expect(searchInput).toBeVisible();

        await searchInput.fill(nonexistentKeyword);
        await page.waitForTimeout(2000);
        await expect(page.locator(".pod_search").getByText("共0个应用")).toBeVisible();
        await expect(page.getByText("暂无内容")).toBeVisible();
        expect((await applicationTitles.allTextContents()).map((title) => title.trim())).toEqual([]);

        await searchInput.clear();
        await expect(searchInput).toHaveValue("");
        await expect.poll(async () => (await applicationTitles.allTextContents()).map((title) => title.trim())).toEqual(initialTitles);

        await searchInput.fill(chineseKeyword);
        await page.waitForTimeout(2000);
        await expect(page.locator(".pod_search").getByText("共0个应用")).toBeVisible();
        await expect(page.getByText("暂无内容")).toBeVisible();
        expect((await applicationTitles.allTextContents()).map((title) => title.trim())).toEqual([]);
    });

    test("【Marketplace-055】应用市场-首页-搜索-构造异常-搜索内容超长-前台", async ({ page }) => {
        await openApplicationListPageByUrl(page);
        await expect(page.getByPlaceholder("搜索应用名称")).toBeVisible({ timeout: 30000 });

        const searchInput = page.getByPlaceholder("搜索应用名称");
        const overlongKeyword = "a".repeat(200);
        const overlongTip = "搜索名称不能超过53个字符";

        await expect(searchInput).toBeVisible();

        await searchInput.fill(overlongKeyword);
        await expect(page.getByText(overlongTip)).toBeVisible();
    });

    test("【Marketplace-058】平台管理-仓库-包管理-概览-搜索-输入长度限制-前台", async ({ page }) => {
        await openRepositoryConfigPageByUrl(page);

        await page.getByRole("tab", { name: "包管理" }).click();
        await page.waitForLoadState("networkidle");

        const searchInput = page.getByPlaceholder("搜索应用名称");
        const overlongKeyword = "a".repeat(300);
        const overlongTip = "搜索名称不能超过53个字符";

        await expect(searchInput).toBeVisible();

        await searchInput.fill(overlongKeyword);
        await expect(page.getByText(overlongTip)).toBeVisible();
    });

    test("【Marketplace-062】平台管理-仓库-仓库配置-仓库详情-搜索-构造异常-前台输入信息超长", async ({ page }) => {
        await openRepositoryConfigPageByUrl(page);

        const remoteRepositoryRow = page.locator("tbody tr")
            .filter({ has: page.locator('a[href*="/stashDetails/openFuyao"]') })
            .first();
        const remoteRepositoryLink = remoteRepositoryRow.locator('a[href*="/stashDetails/openFuyao"]').first();
        await remoteRepositoryLink.click();
        await page.waitForLoadState("networkidle");

        const searchInput = page.getByPlaceholder("搜索包名称");
        const overlongKeyword = "a".repeat(300);
        const overlongTip = "搜索名称不能超过53个字符";

        await expect(searchInput).toBeVisible();

        await searchInput.fill(overlongKeyword);
        await expect(page.getByText(overlongTip)).toBeVisible();
    });

    test("【Marketplace-065】平台管理-仓库-仓库配置(概览)-搜索-构造异常-不存在的仓库名", async ({ page }) => {
        await openRepositoryConfigPageByUrl(page);

        const searchInput = page.getByPlaceholder("请输入仓库");
        const nonexistentKeyword = "不存在的仓库名123456";
        const repositoryRows = page.locator("tbody tr");

        await expect(searchInput).toBeVisible();

        await searchInput.fill(nonexistentKeyword);
        await page.waitForTimeout(2000);

        await expect(page.locator(".container-platform-pagination")).toContainText("共0条");
        await expect(page.getByText("暂无数据")).toBeVisible();
        await expect(page.locator("tbody")).toContainText("暂无数据");
    });

    test("【Marketplace-066】平台管理-仓库-仓库配置(概览)-搜索-构造异常-前台输入信息超长", async ({ page }) => {
        await openRepositoryConfigPageByUrl(page);

        const searchInput = page.getByPlaceholder("请输入仓库");
        const overlongKeyword = "a".repeat(200);
        const overlongTip = "搜索名称不能超过53个字符";

        await expect(searchInput).toBeVisible();

        await searchInput.fill(overlongKeyword);
        await expect(page.getByText(overlongTip)).toBeVisible();
    });

});
