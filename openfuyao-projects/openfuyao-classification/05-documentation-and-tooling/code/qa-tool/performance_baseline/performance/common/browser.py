from playwright.sync_api import sync_playwright


def start_browser(base_url, storage_state, headless: bool = True):
    """
    启动浏览器实例。

    Args:
        base_url: 基础URL
        storage_state: 存储状态
        headless: 是否无头模式，默认 True

    Returns:
        tuple: (playwright, browser, context, page) - 浏览器实例
    """
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=headless)
    context, page = create_browser_context(browser, base_url, storage_state)
    return playwright, browser, context, page


def create_browser_context(browser, base_url, storage_state):
    """
    创建浏览器上下文。

    Args:
        browser: 要创建上下文的浏览器实例
        base_url: 基础URL
        storage_state: 存储状态

    Returns:
        tuple: (context, page) - 浏览器上下文和页面
    """
    context_options = {
        'base_url': base_url,
        'storage_state': storage_state,
        'ignore_https_errors': True
    }
    context = browser.new_context(**context_options)
    page = context.new_page()
    return context, page


def close_browser(browser, playwright):
    """
    关闭浏览器实例。

    Args:
        browser: 要关闭的浏览器实例
        playwright: 要停止的playwright实例
    """
    if browser:
        browser.close()

    if playwright:
        playwright.stop()
