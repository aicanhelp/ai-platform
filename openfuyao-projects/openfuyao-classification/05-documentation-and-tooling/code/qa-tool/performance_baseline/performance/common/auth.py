import logging
import json
from pathlib import Path

from playwright.sync_api import BrowserContext, expect, Page


def create_if_not_exists(file_path):
    path = Path(file_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=4)


def is_logged_in(page: Page) -> bool:
    """
    检查是否已登录（已登录的的话会重定向到控制面的主页）
    :param page: Playwright page 对象
    :return: 是否已登录
    """
    page.goto('/oauth2/auth/login/fuyaoPasswordProvider')
    page.wait_for_load_state('networkidle')
    current_url = page.url
    return '/oauth2/auth/login/fuyaoPasswordProvider' not in current_url


def login(
    context: BrowserContext,
    credentials: dict[str, str],
    session_file: str,
    logger: logging.Logger | None = None,
) -> None:
    """
    执行登录操作。
    """
    log = logger or logging.getLogger(__name__)
    page = context.new_page()

    if is_logged_in(page):
        log.info('Already logged in.')
    else:
        log.info('Not logged in, attempting login...')
        page.get_by_label('用户名').fill(credentials['username'])
        page.get_by_label('密码').fill(credentials['password'])
        page.get_by_role('button', name='登录').click()

        expect(page.locator(".nav").get_by_text(credentials['username'])).to_be_visible()
        log.info('Login successful.')
        context.storage_state(path=session_file)

    page.close()
