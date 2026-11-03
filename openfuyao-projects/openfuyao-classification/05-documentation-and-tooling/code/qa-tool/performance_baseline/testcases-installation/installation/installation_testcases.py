import os

from playwright.sync_api import Page, expect, TimeoutError as PlaywrightTimeoutError


TEST_CLUSTER_NAME = "test-cluster"
TEST_VERSION = "v25.09"

TIMEOUT = 10 * 60 * 1000    # 10 minutes


def e2e(page: Page):
    for _ in range(5):
        page.goto("/")
        # 删除
        page.get_by_role("img", name="more").locator("svg").click(timeout=TIMEOUT)
        page.get_by_role("button", name="删除").click()
        page.get_by_text("已了解此次删除为深度删除，不可撤销。").click()
        page.get_by_role("button", name="删 除").click()
        expect(page.get_by_role("row", name=TEST_CLUSTER_NAME)).not_to_be_visible(timeout=TIMEOUT)
        # 安装集群
        page.get_by_role("button", name="plus 新增集群").click()
        page.get_by_role("textbox", name="集群名称").fill(TEST_CLUSTER_NAME)
        page.get_by_role("combobox", name="要安装的 openFuyao 版本").click()
        page.get_by_title(TEST_VERSION).click()
        page.get_by_role("textbox", name="请输入节点名称").fill(os.getenv('TEST_NODE1_NAME', ''))
        page.get_by_role("textbox", name="请输入IP地址").fill(os.getenv('TEST_NODE1_IP', ''))
        page.get_by_role('spinbutton', name='请输入端口号').fill(os.getenv('TEST_NODE1_PORT', ''))
        page.get_by_role("textbox", name="请输入用户名").fill(os.getenv('TEST_NODE1_USER', ''))
        page.get_by_role("textbox", name="请输入密码").fill(os.getenv('TEST_NODE1_PASSWORD', ''))
        page.get_by_role("combobox", name="镜像仓库地址").click()
        page.get_by_title(os.getenv('IMAGE_ADDR', '')).click()
        page.get_by_role("combobox", name="镜像仓库IP地址").click()
        page.get_by_title(os.getenv('IMAGE_IP', '')).click()
        page.get_by_role("combobox", name="Chart仓库地址").click()
        page.get_by_title(os.getenv('CHART_ADDR', '')).click()
        page.get_by_role("button", name="一键校验").click()
        page.get_by_role("button", name="创 建").click()
        try:
            page.get_by_role("button", name="退出日志").wait_for(timeout=5000)
        except PlaywrightTimeoutError:
            page.get_by_role("button", name="创 建").click()
            page.get_by_role("button", name="退出日志").wait_for(timeout=1000)

        # 升级
        expect(page.get_by_role("row", name=TEST_CLUSTER_NAME).get_by_text("健康")).to_be_visible(timeout=TIMEOUT)
        page.get_by_role("row", name=TEST_CLUSTER_NAME).get_by_role("img", name="more").click()
        page.get_by_role("button", name="升级", exact=True).click()
        page.get_by_role("combobox", name="* 选择升级版本").click()
        page.locator(".cluster-select-item").first.click()  # 选择第一个可用的版本
        page.get_by_role("button", name="确 定").click()
        try:
            page.wait_for_timeout(5000)
            page.locator('dialog').get_by_text('升级集群').wait_for(state='hidden', timeout=1000)
        except PlaywrightTimeoutError:
            page.get_by_role("button", name="确 定").click()
        expect(page.get_by_role("row", name=TEST_CLUSTER_NAME).get_by_text("健康")).to_be_visible(timeout=TIMEOUT)


def scale_node(page: Page):
    for _ in range(5):
        page.goto("/")
        expect(page.get_by_role("row", name=TEST_CLUSTER_NAME).get_by_text("健康")).to_be_visible(timeout=TIMEOUT)
        page.get_by_role("link", name=TEST_CLUSTER_NAME).click()
        # 扩容
        page.get_by_role("button", name="plus 新增节点").click()
        page.get_by_role("textbox", name="请输入节点名称").fill(os.getenv('TEST_NODE2_NAME', ''))
        page.get_by_role("textbox", name="请输入IP地址").fill(os.getenv('TEST_NODE2_IP', ''))
        page.get_by_role("spinbutton", name="请输入端口号").fill(os.getenv('TEST_NODE2_PORT', ''))
        page.get_by_role("textbox", name="请输入用户名").fill(os.getenv('TEST_NODE2_USER', ''))
        page.locator("#node_0_password").fill(os.getenv('TEST_NODE2_PASSWORD', ''))
        page.get_by_role("button", name="一键校验").click()
        page.wait_for_timeout(1000)
        page.get_by_role("button", name="下一步").click()
        page.wait_for_timeout(1000)
        page.get_by_role("button", name="退出日志").click()
        expect(page.get_by_role("row", name=os.getenv('TEST_NODE2_NAME', ''))
            .get_by_text("就绪", exact=True)).to_be_visible(timeout=TIMEOUT)
        # 缩容
        page.get_by_role("row", name="node").get_by_role("cell", name="delete").locator("svg").click()
        page.get_by_text("已了解此次删除不可撤销").click()
        page.get_by_role("dialog").get_by_role("button", name="删 除").click()
        page.get_by_role("button", name="退出日志").click()
        expect(page.get_by_role("row", name=os.getenv('TEST_NODE2_NAME', ''))
            .get_by_text("就绪")).not_to_be_visible(timeout=TIMEOUT)
