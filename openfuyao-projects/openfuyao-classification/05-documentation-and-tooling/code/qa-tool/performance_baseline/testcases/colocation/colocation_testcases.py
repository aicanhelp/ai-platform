import re
import time
from playwright.sync_api import Page
from . import colocation_configuration as cc


def get_colocation_config(page: Page):
    # 点击混部配置页面的刷新按钮若干次
    cc.visit_colocation_config_page(page)
    for _ in range(5):
        cc.get_sync_button(page).click()
        time.sleep(1)


def update_node_colocation(page: Page):
    # 对每一个节点都开启、关闭若干次
    cc.visit_colocation_config_page(page)
    for _ in range(3):
        for sw in cc.get_colocation_switches(page):
            sw.click()
            time.sleep(3)
        for sw in cc.get_colocation_switches(page):
            sw.click()
            time.sleep(3)


def update_colocation_config(page: Page):
    # 修改混部配置若干次
    cc.visit_colocation_config_page(page)
    for _ in range(6):
        cc.do_open_modal(page)
        page.locator("div").filter(has_text=re.compile(r"^负载感知调度$")).get_by_role("switch").click()
        page.locator("form").filter(has_text="CPU弹性限流").locator("#enabledThreshold").click()
        page.locator("form").filter(has_text="内存异步分级回收").locator("#enabledThreshold").click()
        page.locator("form").filter(has_text="访存带宽和L3 Cache限制").locator("#enabledThreshold").click()
        page.locator("form").filter(has_text="基于PSI指标的驱逐").locator("#enabledThreshold").click()
        page.locator("form").filter(has_text="离线负载水位线驱逐").locator("#enabledThreshold").click()

        cc.do_click_modal_confirm(page)
        time.sleep(3)


def get_colocation_monitor(page: Page):
    cc.visit_colocation_monitor_page(page)
    for _ in range(5):
        with page.expect_response("**/colocation-metrics/nodes/**") as r:
            page.get_by_role("tab", name="节点级混部监控").click()
        with page.expect_response("**/colocation-metrics/cluster") as r:
            page.get_by_role("tab", name="集群级混部监控").click()
