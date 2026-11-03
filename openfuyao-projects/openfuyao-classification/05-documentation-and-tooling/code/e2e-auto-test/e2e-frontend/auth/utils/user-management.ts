import { expect, Page } from "@playwright/test";
import { TEST_FUYAOURL } from "../consts";

// 在用户管理界面删除指定用户
async function deleteUser(page: Page, username: string, platformRole: string) {
  await page.getByRole('row').filter({ hasText: username }).getByLabel('more').click();
  await page.getByRole('button', { name: '删除' }).click();
  await expect(page.getByRole('dialog', { name: '删除用户' })).toBeVisible();
  await page.getByRole('checkbox', { name: '已了解删除操作无法恢复' }).check();
  await page.getByRole('button', { name: '删 除' }).click();
  await expect(page.getByText(`删除用户${username}成功！`).first()).toBeVisible();
}

// 在集群成员界面邀请用户为集群成员并指定集群角色
async function inviteUserWithRole(page: Page, username: string, clusterRole: string) {
  await page.getByRole('button', { name: '邀 请' }).click();
  await expect(page.getByRole('dialog', { name: '邀请成员' })).toBeVisible();
  await page.getByRole('row').filter({ hasText: username }).getByLabel('plus-circle').click();
  await page.locator('.container-platform-select-in-form-item').click();
  await page.getByTitle(clusterRole).locator('div').click();
  await page.getByRole('button', { name: /确\s*定|确\s*认/ }).click();
  // 等待对话框关闭，避免后续检查时出现多个匹配元素
  await expect(page.getByRole('dialog', { name: '邀请成员' })).not.toBeVisible();
}

// 在集群成员界面移除集群成员
async function removeClusterUser(page: Page, username: string, clusterRole: string) {
  // 点击该行操作列中的"more"按钮
  await page.getByRole('row', { name: `${username} ${clusterRole}` }).getByLabel('more').click();
  await page.getByRole('button', { name: '移除' }).click();
  await expect(page.getByRole('dialog', { name: '移除成员' })).toBeVisible();
  await page.getByRole('checkbox', { name: '已了解移除操作无法恢复' }).check();
  await page.getByRole('button', { name: '移 除' }).click();
  await expect(page.getByText(`移除${username}成员成功！`).first()).toBeVisible();
}

// 在集群成员界面修改集群成员角色
async function editClusterUser(page: Page, username: string, oldRole: string, newRole: string) {
  // 点击该行操作列中的"more"按钮
  await page.getByRole('row', { name: `${username} ${oldRole}` }).getByLabel('more').click();
  await page.getByRole('button', { name: '修改角色' }).click();
  await expect(page.getByRole('dialog', { name: '修改角色' })).toBeVisible();
  await page.getByLabel('修改角色').getByTitle(oldRole).click();
  // 在对话框内选择新角色（可能存在多个下拉菜单，使用first）
  await page.locator('.container-platform-select-dropdown').getByTitle(newRole).first().click();
  await page.getByRole('button', { name: /确\s*定|确\s*认/ }).click();
}

// 验证集群成员角色是否已更新
async function verifyClusterUserRole(page: Page, username: string, expectedRole: string) {
  // 等待修改成功提示消息出现（可能存在多个通知，使用first）
  await expect(page.getByText(`编辑${username}成员成功！`).first()).toBeVisible();

  // 等待表格重新加载完成
  await page.waitForLoadState('networkidle');

  // 验证表格中的角色是否已更新
  const row = page.getByRole('row', { name: username + " " + expectedRole + " more" });
  await expect(row).toBeVisible();
}

// 生成带时间戳和随机数的用户名后缀
function generateUserSuffix(): string {
  const timestamp = Date.now().toString(36);
  const randomNum = Math.floor(Math.random() * 10000).toString().padStart(4, '0');
  return `${timestamp}-${randomNum}`;
}

// 创建用户并返回带后缀的用户名
async function createUser(
  page: Page,
  username: string,
  initPassword: string,
  platformRole: string,
  describe: string = ""
): Promise<string> {
  const suffixedUsername = `${username}-${generateUserSuffix()}`;
  await page.goto(`${TEST_FUYAOURL}/user_manage/user`);
  await page.getByRole('button', { name: /创\s*建/ }).click();
  await expect(page.getByRole('dialog', { name: '创建用户' })).toBeVisible();
  await page.getByRole('textbox', { name: '* 用户名称' }).fill(suffixedUsername);
  await page.getByRole('textbox', { name: '* 初始密码' }).fill(initPassword);
  await page.locator('.container-platform-select-in-form-item').click();
  await page.locator('.container-platform-select-dropdown').getByTitle(platformRole).locator('div').click();
  await page.getByRole('textbox', { name: '用户描述' }).fill(describe);
  await page.getByRole('button', { name: /确\s*定|确\s*认/ }).click();
  await expect(page.getByText('创建用户成功！').first()).toBeVisible();
  await page.waitForTimeout(300);
  return suffixedUsername;
}

export {
  deleteUser,
  inviteUserWithRole,
  removeClusterUser,
  editClusterUser,
  verifyClusterUserRole,
  createUser
}