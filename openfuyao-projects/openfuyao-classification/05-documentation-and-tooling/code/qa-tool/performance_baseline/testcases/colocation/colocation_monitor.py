from playwright.sync_api import Locator, Page


COLOCATION_MONITOR_PAGE_URI = "/container_platform/colocation/ColocationMonitor"


def visit_colocation_monitor_page(page: Page):
    page.goto(COLOCATION_MONITOR_PAGE_URI)


def get_monitor_paragraphs(page: Page) -> list[Locator]:
    return page.locator('.monitor_namespace').get_by_role('paragraph').all()


def do_switch_to_cluster_monitor(page: Page):
    page.get_by_role('tab', name='集群级混部监控').click()
    page.wait_for_timeout(1000)


def do_switch_to_node_monitor(page: Page):
    page.get_by_role('tab', name='节点级混部监控').click()
    page.wait_for_timeout(1000)