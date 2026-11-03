from playwright.sync_api import Locator, Page

COLOCATION_OVERVIEW_PAGE_URI = "/container_platform/colocation/ColocationOverview"


def visit_colocation_overview_page(page: Page):
    page.goto(COLOCATION_OVERVIEW_PAGE_URI)


def get_sidemenu_link_overview(page: Page) -> Locator:
    return page.get_by_role('link', name='概览', exact=True)


def get_sidemenu_link_configuration(page: Page) -> Locator:
    return page.get_by_role('link', name='混部策略配置', exact=True)


def get_sidemenu_link_monitor(page: Page) -> Locator:
    return page.get_by_role('link', name='混部监控', exact=True)


def do_click_info_1(page: Page) -> None:
    page.get_by_text('1环境准备').get_by_label('question-circle').click()


def get_drawer_title_1(page: Page) -> Locator:
    return page.get_by_role('heading', name='环境准备')


def do_click_info_3(page: Page) -> None:
    page.get_by_text('3工作负载部署').get_by_label('question-circle').click()


def get_drawer_title_3(page: Page) -> Locator:
    return page.get_by_role('heading', name='工作负载部署')


def do_close_drawer(page: Page) -> None:
    page.get_by_role('img', name='close').locator('svg').click()


def do_click_link_2(page: Page) -> None:
    page.get_by_role('link', name='配置混部策略').click()


def do_click_link_3(page: Page) -> None:
    page.get_by_role('link', name='部署工作负载').click()


def do_click_link_4(page: Page) -> None:
    page.get_by_role('link', name='查看在离线混部监控').click()