from datetime import datetime, timezone, timedelta
from pathlib import Path
import random
from typing import Literal, Optional
import threading
import time

from kubernetes import config

from performance.runner.constants import (
    ENV_FILE,
    KUBECONFIG_PATH,
    RESOURCE_MONITOR_DELAY_PERIOD,
)
from performance.runner.kubeconfig import fetch_and_save_kubeconfig, validate_kubeconfig
from performance.runner.network_monitor.entrypoint import exec_network_monitor, generate_network_csv
from performance.runner.resource_monitor.entrypoint import exec_resource_monitor, generate_resource_csv
from performance.runner.session import SessionContext

MonitorMode = Literal["featured", "resource"]


class PerformanceTestRunner:
    """
    性能测试运行器类，负责管理和执行所有特性的性能测试。
    """

    def __init__(
        self,
        testcase_path: Path = None,
        features: Optional[list[str]] = None,
        headless: bool = True,
        monitor_mode: MonitorMode = "featured",
        duration_seconds: Optional[float] = None,
        session: SessionContext = None,
    ):
        """
        初始化性能测试运行器。

        Args:
            testcase_path: 测试用例目录路径，默认为 ./testcases
            features: 要跑的特性名列表，须与用例配置中的 category 一致
            headless: Playwright 是否以无头模式启动 Chromium，默认 True
            monitor_mode: featured=网络+资源双监控；resource=仅资源监控
            duration_seconds: 资源监控运行时长（秒）；0 表示无限。仅 resource 模式使用。
            session: 会话上下文，提供日志路径与数据写入器
        """
        if session is None:
            raise ValueError("session is required")

        self.session = session
        self.features = features or []
        self.headless = headless
        self.monitor_mode = monitor_mode
        self.duration_seconds = duration_seconds
        self.general_logger = session.general_logger()

        self.general_logger.info('=' * 80)
        self.general_logger.info(f'Performance Test new round start at {session.paths.run_id}!')
        self.general_logger.info('=' * 80)

        self.testcase_path = testcase_path or Path('./testcases')
        self.testcase_list_path = self.testcase_path / 'network_testcase_list.yaml'
        self.resource_category_path = self.testcase_path / 'resource_pod_category.yaml'

        self.general_logger.info(f'测试用例路径: {self.testcase_path}')
        self.general_logger.info(f'运行模式: {self.monitor_mode}')
        if self.monitor_mode == "resource":
            duration_desc = (
                f"{self.duration_seconds}s"
                if self.duration_seconds and self.duration_seconds > 0
                else "无限（直至手动中断）"
            )
            self.general_logger.info(f'资源监控时长: {duration_desc}')
        self.general_logger.info(f'特性列表: {", ".join(self.features)}')
        self.general_logger.info(f'网络测试配置: {self.testcase_list_path}')
        self.general_logger.info(f'资源测试配置: {self.resource_category_path}')

    def _ensure_kubeconfig(self):
        if not KUBECONFIG_PATH.exists():
            self.general_logger.info(f"kubeconfig 文件路径 {KUBECONFIG_PATH} 不存在，尝试从远程获取...")
            fetch_and_save_kubeconfig()
        try:
            config.load_kube_config(config_file=str(KUBECONFIG_PATH))
            validate_kubeconfig()
        except Exception:
            self.general_logger.info(f"{KUBECONFIG_PATH} 不是有效的 kubeconfig文件，尝试从远程获取...")
            fetch_and_save_kubeconfig()

        if not validate_kubeconfig():
            self.general_logger.error("当前 kubeconfig 文件不可用，无法访问集群。")
            fetch_and_save_kubeconfig()

    def run_feature_test(self, feature_category: str):
        """
        执行单个特性的性能测试，按 monitor_mode 启动 network/resource 监控。

        :param feature_category: 特性类别字符串。
        """
        run_network = self.monitor_mode == "featured"
        run_resource = True

        if run_resource:
            self._ensure_kubeconfig()

        threads = []
        network_thread = None

        if run_network:
            def network_monitor_wrapper():
                self.general_logger.info(f"[{feature_category}] network_monitor 启动。")
                try:
                    exec_network_monitor(
                        self.session,
                        feature_category,
                        self.testcase_list_path,
                        headless=self.headless,
                    )
                    self.general_logger.info(f"[{feature_category}] network_monitor 完成, 请查看采集信息。")
                except Exception:
                    self.general_logger.error(
                        f"[{feature_category}] network_monitor 异常, 请查看步骤日志。",
                        exc_info=True,
                    )

            network_thread = threading.Thread(
                target=network_monitor_wrapper,
                name=f"network-{feature_category}",
            )
            threads.append(network_thread)

        resource_thread = None
        if run_resource:
            if self.monitor_mode == "featured":
                stop_thread = network_thread
                duration_seconds = None
            else:
                stop_thread = None
                duration_seconds = self.duration_seconds

            def resource_monitor_wrapper():
                self.general_logger.info(f"[{feature_category}] resource_monitor 启动。")
                try:
                    exec_resource_monitor(
                        self.session,
                        feature_category,
                        self.resource_category_path,
                        stop_thread=stop_thread,
                        duration_seconds=duration_seconds,
                    )
                    self.general_logger.info(f"[{feature_category}] resource_monitor 完成, 请查看采集信息。")
                except Exception:
                    self.general_logger.error(
                        f"[{feature_category}] resource_monitor 异常, 请查看步骤日志。",
                        exc_info=True,
                    )

            resource_thread = threading.Thread(
                target=resource_monitor_wrapper,
                name=f"resource-{feature_category}",
            )
            threads.append(resource_thread)

        if run_network:
            network_thread.start()
        if run_resource:
            resource_thread.start()

        for t in threads:
            t.join()

        self.general_logger.info(f"[{feature_category}] 性能测试结束")

    def schedule_and_run_all_features(self, delayed: bool = False):
        """
        为每个特性生成当前时间后 DELAY 小时内的随机启动时间，
        并在该时间点以线程方式启动特性测试。
        所有特性测试完成后，主程序自动结束。
        """
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE)
        now = datetime.now(timezone.utc).astimezone()
        threads = []
        for feature in self.features:
            if delayed:
                delta_seconds = random.randint(0, int(RESOURCE_MONITOR_DELAY_PERIOD * 60 * 60))
                start_time = now + timedelta(seconds=delta_seconds)
                self.general_logger.info(f"[{feature}]将在  {start_time.strftime('%H:%M:%S')} 启动测试")

                def delayed_run(feature=feature, start_time=start_time):
                    wait = (start_time - datetime.now(timezone.utc).astimezone()).total_seconds()
                    if wait > 0:
                        time.sleep(wait)
                    self.run_feature_test(feature)

                t = threading.Thread(target=delayed_run, name=f"feature-delayed-{feature}")
            else:
                t = threading.Thread(
                    target=self.run_feature_test,
                    args=(feature,),
                    name=f"feature-{feature}",
                )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        self.general_logger.info("所有特性测试已完成")

        result_csv_path = self.session.paths.resource_monitor / 'result.csv'
        self.general_logger.info(f'Resource Monitor 输出csv结果到{result_csv_path}...')
        generate_resource_csv(self.session, result_csv_path)
        if self.monitor_mode == "featured":
            result_csv_path = self.session.paths.network_monitor / 'result.csv'
            self.general_logger.info(f'Network Monitor 输出csv结果到{result_csv_path}...')
            generate_network_csv(
                self.session, result_csv_path, self.testcase_list_path, self.features
            )
