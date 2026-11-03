import { test, type Page } from "@playwright/test";
import {
    deletePackageFixtureByName,
    openPackageManagementPageByUrl,
} from "@/marketplace/utils/marketplace";
import { withAuthorizedPage } from "./utils/application";
import {
    componentNameApp,
    componentNameExt,
    componentNameExt2,
} from "./utils/constants";

function packageNames(): string[] {
    return [
        componentNameApp,
        componentNameExt,
        componentNameExt2,
    ];
}

async function deleteApplicationPackagesIfExist(page: Page): Promise<void> {
    await openPackageManagementPageByUrl(page);
    for (const packageName of packageNames()) {
        await deletePackageFixtureByName(page, packageName);
    }
}

test("清理 application 测试包", async ({ browser }) => {
    test.setTimeout(5 * 60 * 1000);
    await withAuthorizedPage(browser, deleteApplicationPackagesIfExist);
});
