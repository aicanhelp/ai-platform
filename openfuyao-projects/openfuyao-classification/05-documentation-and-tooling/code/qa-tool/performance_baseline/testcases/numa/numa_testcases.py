import time
from playwright.sync_api import Page


def get_and_update_numa_config(page: Page):
    page.goto("/container_platform/scheduling/NumaOverview")
    for _ in range(5):
        page.get_by_role('link', name='亲和策略配置').click()
        for sw in page.get_by_role("switch").all():
            sw.click()
            try:
                modal = page.get_by_role("dialog")
                modal.wait_for(state="visible", timeout=3000)
            except Exception:
                # some of interactions would show no dialog, just skip
                page.wait_for_timeout(1)
            else:
                modal.get_by_role("button", name="确 定").click()
            time.sleep(20)  # wait for longer so that kubelet restarts
        page.get_by_role("link", name="概览").click()


def view_numa_monitor(page: Page):
    page.goto("/container_platform/scheduling/NumaOverview")
    for _ in range(5):
        page.get_by_role('link', name='集群NUMA监控').click()
        page.get_by_role("table").get_by_role("link").first.click()
        page.get_by_role("link", name="概览").click()
