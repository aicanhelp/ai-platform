import type { Page } from '@playwright/test';
import { expect } from '@playwright/test';
import { addonOptions, createClusterData } from '../constants/common';
import type { NodeInfoEntry } from '../constants/common';

// ==================== 集群创建操作 ====================
export async function goToCreateClusterPage(page: Page) {
    await page.getByRole('button', { name: 'plus 新增集群' }).click();
}

export async function setClusterInfo(page: Page, clusterName: string, version?: string) {
    await page.getByRole('textbox', { name: '集群名称' }).fill(clusterName);
    await page.getByRole('combobox', { name: '要安装的 openFuyao 版本' }).click();
    if (version) {
        await page.getByTitle(version, { exact: true }).click();
    } else {
        await page.locator('.cluster-select-item-option-content').first().click();
    }
}

export type NodeInfoFields = Partial<
    Pick<NodeInfoEntry, 'name' | 'ip' | 'port' | 'userName' | 'password'>
>;

export async function setNodeInfo(
    page: Page,
    type: 'master' | 'node',
    index: number,
    nodeInfo: NodeInfoFields,
) {
    const nodeInfoPrefix = `#${type}_${index}`;
    if (nodeInfo.name !== undefined) {
        const locator = page.locator(nodeInfoPrefix + '_name');
        if (await locator.isVisible()) await locator.fill(nodeInfo.name);
    }
    if (nodeInfo.ip !== undefined) {
        const locator = page.locator(nodeInfoPrefix + '_ip');
        if (await locator.isVisible()) await locator.fill(nodeInfo.ip);
    }
    if (nodeInfo.port !== undefined) {
        const locator = page.locator(nodeInfoPrefix + '_port');
        if (await locator.isVisible()) await locator.fill(nodeInfo.port);
    }
    if (nodeInfo.userName !== undefined) {
        const locator = page.locator(nodeInfoPrefix + '_userName');
        if (await locator.isVisible()) await locator.fill(nodeInfo.userName);
    }
    if (nodeInfo.password !== undefined) {
        const locator = page.locator(nodeInfoPrefix + '_password');
        if (await locator.isVisible()) await locator.fill(nodeInfo.password);
    }
}

export async function addNode(page: Page, type: 'master' | 'node') {
    if (type === 'master') {
        await page.getByRole('button', { name: '添加新节点' }).nth(0).click();
    } else if (type === 'node') {
        await page.getByRole('button', { name: '添加新节点' }).nth(1).click();
    }
}

export async function setBalanceIp(page: Page, ip: string) {
    await page.getByRole('textbox', { name: '负载均衡器IP地址' }).fill(ip);
}

export async function setAdvancedConfig(page: Page, addonOption: string) {
    await page.getByRole('combobox', { name: '镜像仓库地址' }).click();
    await page.locator('.cluster-select-dropdown').nth(1).locator('.cluster-select-item-option-content').first().click();
    await page.getByRole('combobox', { name: '镜像仓库IP地址' }).click();
    await page.locator('.cluster-select-dropdown').nth(2).locator('.cluster-select-item-option-content').first().click();
    await page.getByText('openFuyao Cores').click();
    if (addonOption === 'openFuyao Cores') {
        await page.getByTitle('openFuyao Cores').nth(1).click();
        await page.getByRole('combobox', { name: 'Chart仓库地址' }).click();
        await page.getByTitle(createClusterData.chartDomain).click();
    } else {
        await page.getByTitle(addonOption, { exact: true }).click();
    }
}

export async function setAdvancedOfflineConfig(page: Page, addonOption: string) {
    // 镜像仓库地址（离线）
    await page.getByRole('combobox', { name: '镜像仓库地址' }).click();
    await page.getByTitle(createClusterData.offlineImageDomain).click();
    // 镜像仓库IP地址（取下拉首项，避免依赖外部 bootstrapIp）
    await page.getByRole('combobox', { name: '镜像仓库IP地址' }).click();
    await page.locator('.cluster-select-dropdown').last().locator('.cluster-select-item-option-content').first().click();
    // 组件选择
    await page.getByText('openFuyao Cores').click();
    if (addonOption === 'openFuyao Cores') {
        // 选择 openFuyao Cores，并设置 Chart 仓库地址为下拉首项（离线）
        await page.getByTitle('openFuyao Cores', { exact: true }).click();
        await page.getByRole('combobox', { name: 'Chart仓库地址' }).click();
        await page.locator('.cluster-select-dropdown').last().locator('.cluster-select-item-option-content').first().click();
    } else {
        await page.getByTitle(addonOption, { exact: true }).click();
    }
}

export async function validateCluster(page: Page) {
    await page.getByRole('button', { name: '一键校验' }).click();
}

export async function createCluster(page: Page) {
    await page.getByRole('button', { name: '创 建' }).click();
}

export interface FillClusterAndValidateOptions {
    clusterName: string;
    version?: string;
    masterInfo: NodeInfoFields;
    addonOption?: string;
    useBootstrap?: boolean;
    offlineImageDomain?: string;
    bootstrapIp?: string;
    validateTimeout?: number;
}

export async function fillClusterAndValidate(
    page: Page,
    {
        clusterName,
        version,
        masterInfo,
        addonOption,
        useBootstrap = false,
        offlineImageDomain,
        bootstrapIp,
        validateTimeout,
    }: FillClusterAndValidateOptions,
) {
    await goToCreateClusterPage(page);
    await setClusterInfo(page, clusterName, version);
    await setNodeInfo(page, 'master', 0, masterInfo);

    if (useBootstrap) {
        await page.getByRole('combobox', { name: '镜像仓库地址' }).click();
        await page.getByTitle(offlineImageDomain!).click();
        await page.getByRole('combobox', { name: '镜像仓库IP地址' }).click();
        await page.getByTitle(bootstrapIp!).click();
        await page.getByRole('combobox', { name: 'Chart仓库地址' }).click();
        const chartAddress = `http://${bootstrapIp}:${createClusterData.chartPort}`;
        await page.getByTitle(chartAddress).click();
    } else {
        await setAdvancedConfig(page, addonOption ?? addonOptions.noAddon);
    }

    await page.getByRole('button', { name: '一键校验' }).click();
    await expect(page.getByText('一键校验成功，请创建')).toBeVisible({ timeout: validateTimeout });
}

// ==================== 集群扩缩容操作 ====================
export async function scaleUpSetNodeInfo(page: Page, index: number, nodeInfo: NodeInfoFields) {
    const nodeInfoPrefix = '#node_' + index;
    if (nodeInfo.name !== undefined) {
        const locator = page.locator(nodeInfoPrefix + '_hostname');
        if (await locator.isVisible()) await locator.fill(nodeInfo.name);
    }
    if (nodeInfo.ip !== undefined) {
        const locator = page.locator(nodeInfoPrefix + '_ip');
        if (await locator.isVisible()) await locator.fill(nodeInfo.ip);
    }
    if (nodeInfo.port !== undefined) {
        const locator = page.locator(nodeInfoPrefix + '_port');
        if (await locator.isVisible()) await locator.fill(nodeInfo.port);
    }
    if (nodeInfo.userName !== undefined) {
        const locator = page.locator(nodeInfoPrefix + '_userName');
        if (await locator.isVisible()) await locator.fill(nodeInfo.userName);
    }
    if (nodeInfo.password !== undefined) {
        const locator = page.locator(nodeInfoPrefix + '_password');
        if (await locator.isVisible()) await locator.fill(nodeInfo.password);
    }
}

export async function scaleUpAddNode(page: Page) {
    await page.getByRole("button", { name: "plus-circle 添加新节点" }).click();
}

export async function scaleUpCheckNodeInfo(page: Page) {
    await page.getByRole('button', { name: '一键校验' }).click();
}

export async function scaleUpConfirm(page: Page) {
    await page.getByRole('button', { name: /确\s*定/ }).click();
}

export async function scaleDownConfirm(page: Page) {
    await page.getByRole('checkbox', { name: '已了解此次删除不可撤销' }).check();
    // 如果当前节点为引导节点的提示，则点击checkbox，没有就跳过
    if (
        await page
            .getByRole('checkbox', { name: '当前节点包含引导节点，谨慎删除，删除后引导节点功能不可用。' })
            .isVisible()
    ) {
        await page
            .getByRole('checkbox', { name: '当前节点包含引导节点，谨慎删除，删除后引导节点功能不可用。' })
            .check();
    }
    await page.getByLabel('删除节点').getByRole('button', { name: /删\s*除/ }).click();
}

export async function closeLogModal(page: Page) {
    await page.waitForFunction(() => {
        const el = document.querySelector('.clusterLogContent');
        return el && el.textContent && el.textContent.trim().length > 0;
    });
    await page.getByRole('button', { name: '退出日志' }).click();
}

// ==================== 集群删除操作 ====================
async function deleteClusterOnce(page: Page, clusterName: string) {
    await page.getByRole('row', { name: clusterName }).locator('svg').nth(1).click();
    await page.getByRole('button', { name: '删除' }).click();
    await page.getByRole('checkbox', { name: '已了解此次删除为深度删除，不可撤销。' }).click();
    // 如果有集群包含引导节点的提示，则点击checkbox，没有就跳过
    if (
        await page
            .getByRole('checkbox', { name: '当前集群节点包含引导节点，谨慎删除，删除后引导节点功能不可用。' })
            .isVisible()
    ) {
        await page
            .getByRole('checkbox', { name: '当前集群节点包含引导节点，谨慎删除，删除后引导节点功能不可用。' })
            .check();
    }
    await page.getByRole('button', { name: '删 除' }).click();
}

export interface DeleteClusterOptions {
    maxRetries?: number;
    dispatchVerifyTimeout?: number;
    retryIntervalMs?: number;
}

export async function deleteCluster(
    page: Page,
    clusterName: string,
    options?: DeleteClusterOptions,
) {
    const maxRetries = options?.maxRetries ?? 3;
    const dispatchVerifyTimeout = options?.dispatchVerifyTimeout ?? 15_000;
    const retryIntervalMs = options?.retryIntervalMs ?? 2_000;
    const deletingPattern = new RegExp(`${clusterName}.*loading\\s*删除中`);
    const clusterRowPattern = new RegExp(clusterName);

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            await deleteClusterOnce(page, clusterName);
            await expect(page.getByRole('row', { name: deletingPattern })).toBeVisible({
                timeout: dispatchVerifyTimeout,
            });
            return;
        } catch {
            const clusterRow = page.getByRole('row', { name: clusterRowPattern }).first();
            if (!(await clusterRow.isVisible().catch(() => false))) {
                return;
            }
            await page.keyboard.press('Escape').catch(() => {});
            await page.waitForTimeout(retryIntervalMs);
        }
    }
    throw new Error(`集群 ${clusterName} 删除请求在 ${maxRetries} 次尝试后仍未成功下发`);
}

// ==================== 集群升级操作 ====================
export async function startClusterUpgrade(page: Page, clusterName: string, targetVersion: string) {
    await page.getByRole('row', { name: clusterName }).locator('.anticon.anticon-more > svg').click();
    await page.getByRole('button', { name: '升级', exact: true }).click();

    const versionCombo = page.getByRole('combobox', { name: '* 选择升级版本' });
    await versionCombo.waitFor({ state: 'visible' });
    await versionCombo.click();

    const versionOption = page.getByTitle(targetVersion, { exact: true }).locator('div').first();
    await expect(versionOption).toBeVisible({ timeout: 10000 });
    await versionOption.click();

    await page.getByRole('button', { name: /确\s*定/ }).click();
}

// ==================== 其他 ====================
// 重置Deployment镜像版本（用于测试收尾）
export async function resetDeploymentImage(page: Page, deploymentName: string, targetImageTag: string) {
    await page.getByRole('menu').getByText('工作负载').click();
    await page.getByRole('link', { name: 'Deployment' }).click();
    await page.getByRole('link', { name: deploymentName }).click();

    await page.getByRole('button', { name: '操作 down' }).click();
    await page.getByRole('button', { name: '修改', exact: true }).click();

    // 使用编辑器内置查找功能定位镜像行
    await page.getByText('namespace: cluster-system').click();
    // 调出查找框 (Ctrl+F)
    await page.keyboard.press('Control+f');
    // 输入查找关键字
    const searchKeyword = 'cluster-api-provider-bke:';
    await page.getByRole('textbox', { name: 'Find' }).fill(searchKeyword);
    await page.getByRole('button', { name: 'next' }).click();

    // 查找后，目标行会滚动到可视区域，使用精确的文本匹配来定位高亮的行
    const imagePrefix = 'image: cr.openfuyao.cn/openfuyao/cluster-api-provider-bke:';
    const imageLocator = page.getByText(new RegExp(imagePrefix));

    // 等待目标行可见
    await expect(imageLocator.first()).toBeVisible({ timeout: 5000 });

    const count = await imageLocator.count();
    for (let i = 0; i < count; i++) {
        const element = imageLocator.nth(i);
        // 获取当前行的完整文本，以便保留缩进
        // 注意：textContent 可能包含非换行空格 (\u00a0)，需要将其转换为普通空格
        const currentLineText = (await element.textContent()) || '';
        const indentMatch = currentLineText.match(/^[\s\u00a0]*/);
        const indentation = (indentMatch ? indentMatch[0] : '').replace(/\u00a0/g, ' ');

        await element.click();

        // 选中整行并删除
        await page.keyboard.press('End');
        await page.keyboard.press('Shift+Home');
        await page.keyboard.press('Shift+Home');
        await page.keyboard.press('Backspace');

        // 输入新内容 (使用提取的缩进)
        const newImage = `cr.openfuyao.cn/openfuyao/cluster-api-provider-bke:${targetImageTag}`;
        await page.keyboard.type(`${indentation}image: ${newImage}`);

        // 如果有多个匹配项，再次按 Enter 查找下一个
        if (i < count - 1) {
            await page.getByRole('textbox', { name: 'Find' }).press('Enter');
            await page.waitForTimeout(500);
        }
    }

    await page.getByRole('button', { name: /确\s*定/ }).click();

    await expect(page.getByRole('tab', { name: '详情' })).toBeVisible({ timeout: 10000 });
}
