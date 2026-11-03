import { test, expect } from "@playwright/test";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { NodeSSH } from "node-ssh";
import type { Page } from "@playwright/test";
import type { PackageFixture } from "@/marketplace/utils/marketplace";
import {
    ensurePackageFixtureUploaded,
    openPackageManagementPageByUrl,
    syncLocalRepository,
} from "@/marketplace/utils/marketplace";
import { withAuthorizedPage } from "./utils/application";
import {
    componentNameApp,
    componentNameExt,
    componentNameExt2,
    packageVersion,
    packageVersionNext,
    TEST_SSH_ADDRESS,
    TEST_SSH_PASSWORD,
    applicationTestNamespace,
} from "./utils/constants";

const __applicationDir = path.dirname(fileURLToPath(import.meta.url));
const testAssetsDir = path.resolve(__applicationDir, "test-assets");

function packageFixture(packageName: string, version: string): PackageFixture {
    const fileName = `${packageName}-${version}.tgz`;
    return {
        filePath: path.join(testAssetsDir, fileName),
        fileName,
        packageName,
        version,
    };
}

const applicationPackageFixtures: PackageFixture[] = [
    packageFixture(componentNameApp, packageVersion),
    packageFixture(componentNameApp, packageVersionNext),
    packageFixture(componentNameExt, packageVersion),
    packageFixture(componentNameExt, packageVersionNext),
    packageFixture(componentNameExt2, packageVersion),
    packageFixture(componentNameExt2, packageVersionNext),
];

/** Upload local application/extension charts via marketplace APIs. */
async function ensureApplicationPackagesUploaded(page: Page): Promise<void> {
    await syncLocalRepository(page);
    await openPackageManagementPageByUrl(page);
    for (const fixture of applicationPackageFixtures) {
        if (!fs.existsSync(fixture.filePath)) {
            throw new Error(`Application package fixture missing: ${fixture.filePath}`);
        }
        await ensurePackageFixtureUploaded(page, fixture);
    }
}

test(`确保 ${applicationTestNamespace} 命名空间存在`, async () => {
    test.setTimeout(60 * 1000);
    expect(TEST_SSH_ADDRESS, "缺少环境变量 TEST_NODE2_IP").toBeTruthy();
    expect(TEST_SSH_PASSWORD, "缺少环境变量 TEST_NODE2_PASSWORD").toBeTruthy();

    const ssh = new NodeSSH();
    await ssh.connect({
        host: TEST_SSH_ADDRESS,
        username: "root",
        password: TEST_SSH_PASSWORD,
        readyTimeout: 30000,
    });
    try {
        const result = await ssh.execCommand(
            `kubectl create namespace ${applicationTestNamespace} --dry-run=client -o yaml | kubectl apply -f -`,
        );
        if ((result.code ?? 1) !== 0) {
            throw new Error(
                `创建 ${applicationTestNamespace} 命名空间失败: stdout=${result.stdout}, stderr=${result.stderr}`,
            );
        }
    } finally {
        ssh.dispose();
    }
});

test("准备 application 测试包", async ({ browser }) => {
    test.setTimeout(5 * 60 * 1000);
    await withAuthorizedPage(browser, ensureApplicationPackagesUploaded);
});
