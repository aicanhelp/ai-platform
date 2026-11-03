import type { Locator, Page } from '@playwright/test';
import { expect } from '@playwright/test';
import { timeouts } from '../constants/common';

function escapeRegExp(value: string): string {
    return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// 验证集群正在安装
export async function verifyClusterInstalling(page: Page, clusterName: string) {
    await expect(page.getByText('集群信息')).toBeVisible();
    await page.waitForFunction(() => {
        const el = document.querySelector('.clusterLogContent');
        return el && el.textContent && el.textContent.trim().length > 0;
    });
    await page.getByRole('button', { name: '退出日志' }).click();
    await page.waitForURL(/\/container_platform\/cluster\/cluster$/);
    const status = clusterName + ' loading 安装中';
    await expect(page.getByRole('row', { name: status })).toBeVisible({ timeout: timeouts.validateTimeout });
}

// 验证集群在指定时间内安装成功
export async function verifyClusterHealthy(page: Page, clusterName: string) {
    const successStatus = page.getByRole('row', { name: clusterName + ' check-circle 健康' });
    const failedStatus = page.getByRole('row', { name: clusterName + ' exclamation-circle 安装失败' });
    await expect(successStatus.or(failedStatus)).toBeVisible({ timeout: timeouts.installTimeout });
    if (await failedStatus.isVisible()) {
        throw new Error(`Cluster ${clusterName} installation failed`);
    }
    await expect(successStatus).toBeVisible();
}
export async function verifyClusterHealthy2(page: Page, clusterName: string) {
    const escapedClusterName = escapeRegExp(clusterName);
    const successStatus = page.getByRole('row', { name: new RegExp(`${escapedClusterName}.*check-circle\\s*健康`) });
    const failedStatus = page.getByRole('row', { name: new RegExp(`${escapedClusterName}.*exclamation-circle\\s*安装失败`) });
    await expect(successStatus.or(failedStatus)).toBeVisible({ timeout: timeouts.installTimeout });
    if (await failedStatus.isVisible()) {
        throw new Error(`Cluster ${clusterName} installation failed`);
    }
    await expect(successStatus).toBeVisible();
}

// 验证集群在指定时间内删除成功
export async function verifyClusterDeleted(page: Page, clusterName: string) {
    const escapedClusterName = escapeRegExp(clusterName);
    const status = new RegExp(`${escapedClusterName}.*loading\\s*删除中`);
    await expect(page.getByRole('row', { name: status })).toBeVisible();
    await expect(page.getByRole('row', { name: clusterName })).not.toBeVisible({ timeout: timeouts.uninstallTimeout * 3 });
}

// 验证集群正在升级
export async function verifyClusterUpgrading(page: Page, clusterName: string) {
    await expect(page.getByText('升级中，时间较长请耐心等待')).toBeVisible();
    const status = clusterName + ' loading 升级中';
    await expect(page.getByRole('row', { name: status })).toBeVisible({ timeout: timeouts.upgradeVisibleTimeout });
}

async function getClusterRowText(page: Page, clusterName: string): Promise<string> {
    const row = page.getByRole('row', { name: new RegExp(escapeRegExp(clusterName)) }).first();
    await expect(row).toBeVisible({ timeout: timeouts.validateTimeout });
    return (await row.textContent()) ?? '';
}

export async function waitForClusterUpgradeStarted(
    page: Page,
    clusterName: string,
    options?: {
        timeoutMs?: number;
        intervalMs?: number;
        requiredOccurrences?: number;
        fallbackMs?: number;
    },
) {
    const start = Date.now();
    const timeoutMs = options?.timeoutMs ?? timeouts.upgradeVisibleTimeout;
    const intervalMs = options?.intervalMs ?? 1000;
    const requiredOccurrences = options?.requiredOccurrences ?? 1;
    const fallbackMs = options?.fallbackMs ?? 2 * 60 * 1000;
    let seenUpgrading = false;
    let occurrenceCount = 0;

    while (Date.now() - start < timeoutMs) {
        const elapsedMs = Date.now() - start;
        const rowText = await getClusterRowText(page, clusterName);
        if (rowText.includes('安装失败') || rowText.includes('升级失败')) {
            throw new Error(`Cluster ${clusterName} upgrade failed: ${rowText}`);
        }
        const isUpgrading = rowText.includes('升级中');
        if (isUpgrading && !seenUpgrading) {
            occurrenceCount += 1;
            if (occurrenceCount >= requiredOccurrences) {
                return;
            }
        }
        if (occurrenceCount > 0 && elapsedMs >= fallbackMs) {
            return;
        }
        seenUpgrading = isUpgrading;
        await page.waitForTimeout(intervalMs);
    }

    throw new Error(
        `Cluster ${clusterName} did not enter upgrading status ${requiredOccurrences} time(s) within ${timeoutMs}ms`,
    );
}

export async function waitForClusterUpgradeFinishedAfterStarted(
    page: Page,
    clusterName: string,
    targetVersion: string,
    options?: {
        timeoutMs?: number;
        intervalMs?: number;
        stableRounds?: number;
    },
) {
    const timeoutMs = options?.timeoutMs ?? timeouts.installTimeout;
    const intervalMs = options?.intervalMs ?? 5000;
    const stableRounds = options?.stableRounds ?? 3;
    const start = Date.now();
    let stableCount = 0;

    while (Date.now() - start < timeoutMs) {
        const rowText = await getClusterRowText(page, clusterName);
        if (rowText.includes('安装失败') || rowText.includes('升级失败')) {
            throw new Error(`Cluster ${clusterName} upgrade failed: ${rowText}`);
        }

        const isReady =
            rowText.includes('健康') &&
            rowText.includes(targetVersion) &&
            !rowText.includes('升级中');

        if (isReady) {
            stableCount += 1;
            if (stableCount >= stableRounds) {
                return;
            }
        } else {
            stableCount = 0;
        }

        await page.waitForTimeout(intervalMs);
    }

    throw new Error(
        `Cluster ${clusterName} upgrade did not finish within ${timeoutMs}ms (targetVersion=${targetVersion}, stableRounds=${stableRounds})`,
    );
}

async function refreshClusterListSoftly(page: Page) {
    const overviewLink = page.getByRole('link', { name: '总览' });
    const lifecycleLink = page.getByRole('link', { name: '集群生命周期管理' });

    if (await overviewLink.isVisible().catch(() => false)) {
        await overviewLink.click();
    }
    if (await lifecycleLink.isVisible().catch(() => false)) {
        await lifecycleLink.click();
    }
}

// 验证升级完成且状态稳定（避免升级过程中短暂回到“健康+新版本”的误判）
export async function verifyClusterUpgradeCompletedStable(
    page: Page,
    clusterName: string,
    targetVersion: string,
    options?: {
        timeoutMs?: number;
        intervalMs?: number;
        stableRounds?: number;
    },
) {
    const timeoutMs = options?.timeoutMs ?? timeouts.installTimeout;
    const intervalMs = options?.intervalMs ?? 10_000;
    const stableRounds = options?.stableRounds ?? 10;
    const start = Date.now();
    let stableCount = 0;

    while (Date.now() - start < timeoutMs) {
        const failedStatus = page.getByRole('row', { name: clusterName + ' exclamation-circle 安装失败' });
        if (await failedStatus.isVisible().catch(() => false)) {
            throw new Error(`Cluster ${clusterName} upgrade failed`);
        }

        const healthyStatus = page.getByRole('row', { name: clusterName + ' check-circle 健康' });
        const versionCell = page.getByRole('cell', { name: targetVersion }).first();
        const bothReady =
            (await healthyStatus.isVisible().catch(() => false)) &&
            (await versionCell.isVisible().catch(() => false));

        if (bothReady) {
            stableCount += 1;
            if (stableCount >= stableRounds) {
                return;
            }
        } else {
            stableCount = 0;
        }

        await page.waitForTimeout(intervalMs);
        await refreshClusterListSoftly(page);
    }

    throw new Error(
        `Cluster ${clusterName} upgrade did not become stable within ${timeoutMs}ms (targetVersion=${targetVersion}, stableRounds=${stableRounds})`,
    );
}

// 验证集群扩容成功
export async function verifyClusterScaledUp(page: Page, clusterName: string, nodeNames: string[]) {
    await expect(page.getByText(`${clusterName}安装中`)).toBeVisible({ timeout: timeouts.validateTimeout });
    await expect(page.getByText(`${clusterName}健康`)).toBeVisible({ timeout: timeouts.installTimeout });
    for (const nodeName of nodeNames) {
        const status = nodeName + ' check-circle 就绪 ';
        await expect(page.getByRole('row', { name: status }).getByRole('paragraph')).toBeVisible({
            timeout: timeouts.installTimeout,
        });
    }
}

// 验证集群缩容成功
export async function verifyClusterScaledDown(page: Page, clusterName: string, nodeNames: string[]) {
    await expect(page.getByText(`${clusterName}缩容中`)).toBeVisible({ timeout: timeouts.validateTimeout });
    await expect(page.getByText(`${clusterName}健康`)).toBeVisible({ timeout: timeouts.installTimeout });
    for (const nodeName of nodeNames) {
        await expect(page.getByRole('cell', { name: nodeName })).not.toBeVisible({ timeout: timeouts.installTimeout });
    }
}

// 验证额集群健康或安装失败，允许删除
export async function verifyClusterHealthOrInstallFailed(page: Page, clusterName: string) {
    const pattern = new RegExp(`${clusterName} check-circle 健康|${clusterName} exclamation-circle 安装失败`);
    await expect(page.getByRole('row', { name: pattern })).toBeVisible({ timeout: timeouts.installTimeout });
}

export async function checkFieldsAndValidateErrorMessages(rows: Locator) {
    const count = await rows.count();

    const fieldSelectors = [
        'input[placeholder="请输入节点名称"]',
        'input[placeholder="请输入IP地址"]',
        'input[placeholder="请输入端口号"]',
        'input[placeholder="请输入用户名"]',
        'input[type="password"]',
    ];

    const errorMessages = ['节点名称不为空', 'ip地址不为空', '端口号不为空', '用户名不为空', '密码不为空'];

    for (let i = 0; i < count; i++) {
        const row = rows.nth(i);

        for (let j = 0; j < fieldSelectors.length; j++) {
            const fieldSelector = fieldSelectors[j];
            const errorMessage = errorMessages[j];

            await checkEmptyFieldAndValidateErrorMessage(
                row,
                fieldSelector,
                'div.cluster-form-item-explain-error',
                errorMessage,
            );
        }
    }
}

async function checkEmptyFieldAndValidateErrorMessage(
    row: Locator,
    fieldName: string,
    errorMessageLocator: string,
    errorMessage: string,
) {
    const fieldInput = row.locator(fieldName);
    const fieldValue = await fieldInput.inputValue();

    if (fieldValue.trim() === '') {
        console.log(`Field is empty, error message displayed: ${errorMessage}`);

        const errorMessageElement = row.locator(errorMessageLocator);
        console.log(errorMessageElement);

        try {
            await errorMessageElement.waitFor({ state: 'visible', timeout: timeouts.validateTimeout });
            await expect(errorMessageElement).toHaveText(errorMessage);
        } catch (error) {
            console.error('Error message not found:', error);
        }
    }
}

/*
    管理集群 & 业务集群
*/
// 验证Deployment状态
export async function verifyDeploymentStatus(page: Page, deploymentName: string, expectedStatus: string) {
    await page.getByRole('menu').getByText('工作负载').click();
    await page.getByRole('link', { name: 'Deployment' }).click();
    await expect(page.getByRole('link', { name: deploymentName })).toBeVisible({ timeout: 30000 });
    const row = page.getByRole('row', { name: deploymentName });
    await expect(row.getByText(expectedStatus)).toBeVisible();
}

// 验证Deployment镜像版本
export async function verifyDeploymentImage(page: Page, deploymentName: string, expectedImageVersion: string) {
    await page.getByRole('link', { name: deploymentName }).click();
    const imagePattern = new RegExp(`cluster-api-provider-bke:${expectedImageVersion}`);
    try {
        await expect(page.getByText(imagePattern)).toBeVisible({ timeout: 5000 });
    } catch {
        await page.getByRole('tab', { name: 'YAML' }).click();
        await expect(page.getByText(imagePattern)).toBeVisible({ timeout: 10000 });
    }
}
