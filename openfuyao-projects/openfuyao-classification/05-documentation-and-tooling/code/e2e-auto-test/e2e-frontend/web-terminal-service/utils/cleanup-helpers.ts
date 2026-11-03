import { expect, Locator, Page } from '@playwright/test';
import { TEST_FUYAOURL } from '../consts';

async function getFirstVisibleLocator(locators: Locator[], errorMessage: string): Promise<Locator> {
  for (const locator of locators) {
    const candidate = locator.first();
    if (await candidate.isVisible().catch(() => false)) {
      return candidate;
    }
  }

  throw new Error(errorMessage);
}

async function waitForFirstVisibleLocator(
  locators: Locator[],
  errorMessage: string,
  timeout = 10000,
): Promise<Locator> {
  await expect.poll(async () => {
    for (let i = 0; i < locators.length; i++) {
      const candidate = locators[i].first();
      if (await candidate.isVisible().catch(() => false)) {
        return i;
      }
    }

    return -1;
  }, {
    timeout,
  }).not.toBe(-1);

  return getFirstVisibleLocator(locators, errorMessage);
}

async function searchTableByKeyword(page: Page, keyword: string) {
  const searchInput = await getFirstVisibleLocator([
    page.getByPlaceholder(/搜索|查找|用户/i),
    page.getByRole('textbox', { name: /搜索|查找|用户/i }),
    page.locator('input[type="text"]'),
  ], 'User search input not found');

  await searchInput.fill(keyword);

  const searchButton = await getFirstVisibleLocator([
    page.getByRole('button', { name: /search|搜索/i }),
    page.locator('[aria-label="search"]'),
  ], 'User search button not found').catch(() => null);

  if (searchButton) {
    await searchButton.click();
  } else {
    await searchInput.press('Enter');
  }

  await page.waitForLoadState('networkidle');
}

async function getRowActionButton(row: Locator) {
  return waitForFirstVisibleLocator([
    row.getByLabel('more'),
    row.getByRole('img', { name: 'more' }),
    row.locator('[aria-label="more"]'),
  ], 'Row action button not found');
}

async function findActionButtonWithRetry(
  page: Page,
  row: Locator,
  candidateLocators: Locator[],
  errorMessage: string,
  timeout = 15000,
): Promise<Locator> {
  for (const locator of candidateLocators) {
    const candidate = locator.first();
    if (await candidate.isVisible().catch(() => false)) {
      return candidate;
    }
  }

  const moreButton = await getRowActionButton(row);
  for (let attempt = 0; attempt < 3; attempt++) {
    await moreButton.scrollIntoViewIfNeeded().catch(() => {});
    await moreButton.click();

    const actionButton = await waitForFirstVisibleLocator(
      candidateLocators,
      errorMessage,
      attempt === 2 ? timeout : 3000,
    ).catch(() => null);

    if (actionButton) {
      return actionButton;
    }

    await page.keyboard.press('Escape').catch(() => {});
    await page.waitForTimeout(300);
  }

  return waitForFirstVisibleLocator(candidateLocators, errorMessage, timeout);
}

export async function deleteUserFromUserManagement(page: Page, username: string) {
  await page.goto(`${TEST_FUYAOURL}/user_manage/user`);
  await page.waitForLoadState('networkidle');

  await searchTableByKeyword(page, username);

  const row = page.locator('tbody tr').filter({ hasText: username }).first();
  await expect(row).toBeVisible({ timeout: 30000 });

  const deleteButton = await findActionButtonWithRetry(page, row, [
    page.getByRole('tooltip').getByRole('button').filter({ hasText: /删\s*除/ }),
    page.getByRole('tooltip').getByRole('button').last(),
    page.getByRole('button', { name: /删\s*除/ }),
    page.getByRole('button').filter({ hasText: /删\s*除/ }),
  ], 'Delete button not found');
  await deleteButton.click();

  const confirmDialog = page.getByRole('dialog');
  await expect(confirmDialog).toBeVisible({ timeout: 30000 });

  const confirmCheckbox = await getFirstVisibleLocator([
    confirmDialog.getByRole('checkbox'),
    page.getByRole('checkbox'),
  ], 'Delete confirmation checkbox not found');
  await confirmCheckbox.check();

  const confirmDeleteButton = await waitForFirstVisibleLocator([
    confirmDialog.getByRole('button', { name: /删\s*除/ }),
    confirmDialog.getByRole('button').filter({ hasText: /删\s*除/ }),
    page.getByRole('button', { name: /删\s*除/ }).last(),
  ], 'Confirm delete button not found');
  await confirmDeleteButton.click();
}

export async function removeClusterUserFromMembers(page: Page, username: string) {
  await page.goto(`${TEST_FUYAOURL}/container_platform/clusterMember`);
  await page.waitForLoadState('networkidle');

  await searchTableByKeyword(page, username);

  const row = page.locator('tbody tr').filter({ hasText: username }).first();
  await expect(row).toBeVisible({ timeout: 30000 });

  const removeButton = await findActionButtonWithRetry(page, row, [
    page.getByRole('button', { name: /移\s*除/ }),
    page.getByRole('button').filter({ hasText: /移\s*除/ }),
  ], 'Remove button not found');
  await removeButton.click();

  const confirmDialog = page.getByRole('dialog');
  await expect(confirmDialog).toBeVisible({ timeout: 30000 });

  const confirmCheckbox = await getFirstVisibleLocator([
    confirmDialog.getByRole('checkbox'),
    page.getByRole('checkbox'),
  ], 'Remove confirmation checkbox not found');
  await confirmCheckbox.check();

  const confirmRemoveButton = await waitForFirstVisibleLocator([
    confirmDialog.getByRole('button', { name: /移\s*除/ }),
    confirmDialog.getByRole('button').filter({ hasText: /移\s*除/ }),
    page.getByRole('button', { name: /移\s*除/ }).last(),
  ], 'Confirm remove button not found');
  await confirmRemoveButton.click();
}
