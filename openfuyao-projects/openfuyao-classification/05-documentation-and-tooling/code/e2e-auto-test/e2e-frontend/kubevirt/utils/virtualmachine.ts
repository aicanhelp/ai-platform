import { Page } from '@playwright/test';

export const clickInstanceTab = async (page: Page) => {
  await page.getByRole('tab', { name: '实例' }).click();
};

export const searchInstance = async (page: Page, name: string) => {
  await page.getByPlaceholder('搜索实例名称').fill(name);
  await page.keyboard.press('Enter');
};

export const clickRowAction = async (page: Page, name: string) => {
  const row = page.getByRole('row', { name: new RegExp(name) }).first();
  await row.locator('[aria-label="more"]').first().click();
  await page.locator('.pop_modal').waitFor({ state: 'visible', timeout: 5000 });
};

export const clickStartVm = async (page: Page) => {
  await page.locator('.pop_modal').getByRole('button', { name: '开机' }).click();
};

export const clickStopVm = async (page: Page) => {
  await page.locator('.pop_modal').getByRole('button', { name: '关机' }).click();
};

export const clickRestartVm = async (page: Page) => {
  await page.locator('.pop_modal').getByRole('button', { name: '重启' }).click();
};

export const clickEditInstance = async (page: Page) => {
  await page.locator('.pop_modal').getByText('修改').first().click();
  await page.waitForURL(/\/cr\//, { timeout: 10000 });
};

export const clickDeleteInstance = async (page: Page) => {
  await page.locator('.pop_modal').getByRole('button', { name: '删除' }).click();
};

export const getEditorContent = async (page: Page): Promise<string> => {
  const editor = page.locator('.cm-editor .cm-content').first();
  await editor.waitFor({ state: 'visible', timeout: 5000 });
  return (await editor.textContent()) || '';
};
