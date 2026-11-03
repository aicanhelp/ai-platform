import type { Page } from '@playwright/test';
import { expect } from '@playwright/test';

// 对应在应用市场安装组件
export async function installComponent(page: Page, componentName: string, appName: string) {
    await page.getByRole('link', { name: '应用列表' }).click();
    await page.getByRole('textbox', { name: '搜索应用名称' }).fill(componentName);
    await page.getByRole('button', { name: 'search' }).click();
    await page.locator('#root_container').getByText(componentName).click();
    await page.getByRole('button', { name: '部 署' }).click();
    await page.getByRole('textbox', { name: '* 应用名称' }).click();
    await page.getByRole('textbox', { name: '* 应用名称' }).fill(appName);
    await page.getByRole('button', { name: /确\s*定/ }).click();
}

// 对应检查deployment (检查指定namespace下有一定数量状态为active的resource)
export async function checkResourceNum(
    page: Page,
    namespace: string,
    resource: string,
    deploymentNum: number,
) {
    await page.getByText('工作负载').click();
    await page.getByRole('link', { name: resource }).click();
    await page.getByText('all').click();
    await page.getByText(namespace, { exact: true }).click();
    await expect(page.getByRole('row', { name: 'Active' }).getByRole('paragraph')).toHaveCount(deploymentNum);
}
