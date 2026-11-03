import re

from playwright.sync_api import Locator, Page

COLOCATION_CONFIG_PAGE_URI = "/container_platform/colocation/ColocationConfiguration"
COLOCATION_MONITOR_PAGE_URI = "/container_platform/colocation/ColocationMonitor"


def visit_colocation_config_page(page: Page):
    page.goto(COLOCATION_CONFIG_PAGE_URI)
    page.wait_for_selector('.colocation-website-table-row')  # 等待表格加载完成 


def get_searchbox(page: Page) -> Locator:
    return page.get_by_role('searchbox', name='搜索节点')


def do_node_seach(page: Page, node_name: str) -> None:
    get_searchbox(page).fill(node_name)
    page.get_by_role('button', name='search').click()


def get_node_name(page: Page) -> list[Locator]:
    return page.locator(f'table tbody tr td:first-child').all()


def get_node_status_column(page: Page) -> list[Locator]:
    return page.locator(f'table tbody tr td:nth-child(2)').all()


def get_colocation_switches(page: Page) -> list[Locator]:
    return page.get_by_role('table').get_by_role('switch').all()


def get_columnheaders(page: Page) -> dict[str, Locator]:
    return {
        'name': page.get_by_role('columnheader', name='节点名称'),
        'status': page.get_by_role('columnheader', name='状态'),
        'ip_addr': page.get_by_role('columnheader', name='IP地址'),
        'os': page.get_by_role('columnheader', name='OS版本'),
        'colocation_enabled': page.get_by_role('columnheader', name='开启混部节点')
    }


def do_filter_status(page: Page, status: str) -> None:
    page.get_by_role('button', name='filter').click()
    page.get_by_role('menuitem', name=status, exact=True).click()
    page.get_by_role('button', name=re.compile(r'确\s*定')).click()


def get_row_by_node_name(page: Page, node_name: str) -> Locator:
    # playwright对表格行的匹配，会匹配各个单元格也以空格为风格福拼接后的结果
    # 通过正则匹配，^匹配开头，节点名称后面加个空格，确保根据节点名匹配行
    # 比如：避免匹配fuyao-worker但是还匹配到fuyao-workerxxxxx的行
    return page.get_by_role('row', name=re.compile(rf'^{node_name} '))


def do_toggle_node_colocation(page: Page, node_name: str) -> None:
    row = get_row_by_node_name(page, node_name)
    row.get_by_role('switch').click()


def get_node_switch_ischecked(page: Page, node_name: str) -> bool:
    row = get_row_by_node_name(page, node_name)
    return row.get_by_role('switch').is_checked()


def get_sync_button(page: Page) -> Locator:
    return page.get_by_role('button', name='sync')


def do_open_modal(page: Page) -> None:
    page.get_by_role('button', name='混部策略参数配置').click()


def get_modal(page: Page) -> Locator:
    return page.get_by_role('dialog', name='混部策略参数配置')


def do_close_modal(page: Page) -> None:
    get_modal(page).get_by_role('button', name='Close').click()


def do_click_modal_confirm(page: Page) -> None:
    get_modal(page).get_by_role('button', name=re.compile(r'确\s*定')).click()


def do_click_modal_cancel(page: Page) -> None:
    get_modal(page).get_by_role('button', name=re.compile(r'取\s*消')).click()


def get_switched(page: Page) -> list[Locator]:
    return get_modal(page).get_by_role('switch').all()
    

def get_usage_plugin_switch(page: Page) -> Locator:
    return page.locator('div').filter(has_text=re.compile(r'^负载感知调度$')).get_by_role('switch')


def do_toggle_usage_plugin_switch(page: Page) -> None:
    get_usage_plugin_switch(page).click()


def get_usage_threshold_switch(page: Page) -> Locator:
    return page.get_by_role('switch', name='真实负载调度阈值开关')


def do_toggle_usage_threshold_switch(page: Page) -> None:
    get_usage_threshold_switch(page).click()


def get_usage_threshold_textbox(page: Page) -> tuple[Locator, Locator]:
    return page.get_by_role('textbox', name='节点CPU真实负载调度阈值'), page.get_by_role('textbox', name='节点内存真实负载调度阈值')


def get_eviction_switch(page: Page) -> Locator:
    return page.locator('div').filter(has_text=re.compile(r'^离线负载水位线驱逐$')).get_by_role('switch')


def do_toggle_eviction_switch(page: Page) -> None:
    get_eviction_switch(page).click()


def get_eviction_threshold_textbox(page: Page) -> tuple[Locator, Locator]:
    return page.get_by_role('textbox', name='CPU驱逐水位线'), page.get_by_role('textbox', name='内存驱逐水位线')


def visit_colocation_monitor_page(page: Page):
    page.goto(COLOCATION_MONITOR_PAGE_URI)
    page.wait_for_load_state('networkidle')