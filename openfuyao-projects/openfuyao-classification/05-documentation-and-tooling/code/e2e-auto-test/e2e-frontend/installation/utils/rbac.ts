import type { Page } from '@playwright/test';
import { expect } from '@playwright/test';

type PlaywrightRole = Parameters<Page['getByRole']>[0];

// 登录操作
export async function login(page: Page, username: string, password: string) {
    await page.getByRole('textbox', { name: '*用户名' }).fill(username);
    await page.getByRole('textbox', { name: '*密码' }).fill(password);
    await page.getByRole('button', { name: '登录' }).click();
    await expect(page.getByText(username)).toBeVisible();
}

// 对应首次登录修改密码界面操作
export async function firstLoginChangePassword(page: Page) {
    await page.getByRole('textbox', { name: '*新密码' }).fill(process.env.TEST_PASSWORD!);
    await page.locator('#confirm-password').fill(process.env.TEST_PASSWORD!);
    await page.getByRole('button', { name: /确\s*认/ }).click();
}

// 注销登录
export async function logout(page: Page) {
    await page.getByRole('img', { name: 'user' }).locator('svg').click();
    await page.getByRole('button', { name: '退出登录' }).click();
}

export async function navigateToPasswordSettings(page: Page) {
    await page.getByRole('img', { name: 'user' }).locator('svg').click();
    await page.getByRole('button', { name: '用户设置' }).click();
    await page.getByRole('link', { name: '密码设置' }).click();
}

export async function changePassword(page: Page, password: string, newPassword: string) {
    await page.getByRole('textbox', { name: '* 当前密码' }).fill(password);
    await page.locator('#new_password').fill(newPassword);
    await page.locator('#confirm_password').fill(newPassword);
    await page.getByRole('button', { name: /确\s*认/ }).click();
}

export interface SubpageNav {
    role: PlaywrightRole;
    name: string;
}

// 导航到 subPage 页面（角色、角色绑定、服务账号）
export async function navigateToSubpage(page: Page, subPage: SubpageNav) {
    await page.getByText('RBAC管理').click();
    await page.getByRole(subPage.role, {
        name: subPage.name,
        exact: true,
    }).click();
    await expect(page.getByRole('list').getByText(subPage.name)).toBeVisible();
}

export interface CreatePageNav {
    role: PlaywrightRole;
    name: string;
}

// 进入新增页面
export async function navigateToCreatePage(page: Page, createPage: CreatePageNav, createPageMessage: string | RegExp) {
    await page.getByRole('button', { name: '创建' }).click();
    await page.getByRole(createPage.role, {
        name: createPage.name,
        exact: true,
    }).click();
    await expect(page.getByText(createPageMessage)).toBeVisible();
}

export interface SearchResultConfig {
    SEARCH_INPUT: { role: PlaywrightRole; name: string };
    RESULT_NAME_CELL: { role: PlaywrightRole; name?: string };
    RESULT_TYPE_CELL: { role: PlaywrightRole; name: string };
}

// 创建并通过搜索验证成功
export async function createAndVerify(page: Page, searchResult: SearchResultConfig, createName: string) {
    await page.getByRole('button', { name: /确\s*定/ }).click();
    await expect(page.getByText('创建成功')).toBeVisible();
    await page.getByRole(searchResult.SEARCH_INPUT.role, {
        name: searchResult.SEARCH_INPUT.name,
    }).fill(createName);
    await page.getByRole('button', { name: '搜 索' }).click();
    await expect(
        page.getByRole(searchResult.RESULT_NAME_CELL.role, {
            name: createName,
        }),
    ).toBeVisible();
    await expect(
        page.getByRole(searchResult.RESULT_TYPE_CELL.role, {
            name: searchResult.RESULT_TYPE_CELL.name,
        }),
    ).toBeVisible();
}
