from collections import defaultdict
import csv
import importlib
import json
from logging import Logger
import os
from pathlib import Path
from time import sleep
import traceback

import numpy as np
from playwright.sync_api import Page
import yaml

from performance.common.auth import login, create_if_not_exists
from performance.common.env import load_pw_config
from performance.common.browser import start_browser, close_browser
from performance.runner.network_monitor.network_monitor import NetworkMonitor
from performance.runner.network_monitor.util import match_url_to_entrypoint
from performance.runner.constants import ENV_FILE
from performance.runner.session import SessionContext, API_TIMING_FILE


def exec_network_monitor(
    ctx: SessionContext,
    feature_category: str,
    testcase_list_path: Path,
    headless: bool = True,
):
    step_logger = ctx.step_logger("network", feature_category)
    step_logger.info('=' * 80)
    step_logger.info('Network monitor new round start!')
    step_logger.info('=' * 80)

    testcases = _load_testcases(feature_category, testcase_list_path)
    if not testcases:
        step_logger.info('未找到目标API或用例，退出任务...')
        return

    monitor = NetworkMonitor(ctx, feature_category, step_logger)
    playwright = browser = context = page = None
    try:
        pw_config = load_pw_config(ENV_FILE)
        context_config = pw_config['context']
        create_if_not_exists(context_config['storage_state'])
        step_logger.info(f"Playwright上下文环境变量: {context_config}")

        playwright, browser, context, page = start_browser(
            base_url=context_config['base_url'],
            storage_state=context_config['storage_state'],
            headless=headless,
        )
        client = context.new_cdp_session(page)
        step_logger.info('启动Playwright浏览器' + (' (headless)' if headless else ' (有界面)'))

        login(context, pw_config['credentials'], context_config['storage_state'], logger=step_logger)
        step_logger.info('验证成功')

        for module in testcases['modules']:
            for func in module['functions']:
                func_full = f"{testcases['dir']}{module['name']}::{func['name']}"
                step_logger.info(f"即将执行用例: {func_full}")
                monitor.start_monitoring(client, func['name'], func['targetAPIs'])
                _exec_task(
                    testcase_name=func['name'],
                    testcase_file=module['name'],
                    testcase_dir=testcases['dir'],
                    page=page,
                    step_logger=step_logger,
                )
                sleep(1)
                monitor.stop_monitoring(client)

    except Exception as e:
        step_logger.error(f'监控过程中发生错误: {e} \n {traceback.format_exc()}')
    finally:
        if browser or playwright:
            close_browser(browser, playwright)


def generate_network_csv(
    ctx: SessionContext,
    result_csv_path: Path,
    testcase_list_path: Path,
    features: list[str],
):
    def calc_stats(values: list[float], res_type: str, round_digit: int = 3) -> dict[str, float]:
        arr = np.array(values, dtype=float)
        return {
            f"{res_type}_avg": round(float(np.mean(arr)), round_digit),
            f"{res_type}_p25": round(float(np.percentile(arr, 25)), round_digit),
            f"{res_type}_p75": round(float(np.percentile(arr, 75)), round_digit),
            f"{res_type}_p90": round(float(np.percentile(arr, 90)), round_digit),
            f"{res_type}_p99": round(float(np.percentile(arr, 99)), round_digit),
        }

    with open(testcase_list_path, 'r', encoding='utf-8') as file:
        testcase_list = yaml.safe_load(file)
    target_apis = set()
    for feature_teastcases in testcase_list:
        feature = feature_teastcases['category']
        if feature not in features:
            continue
        testcases = feature_teastcases['testcases']
        for module in testcases['modules']:
            for func in module['functions']:
                for api in func['targetAPIs']:
                    api_tuple = (feature, api['path'], api['method'])
                    if api_tuple not in target_apis:
                        target_apis.add(api_tuple)
    target_apis = list(target_apis)

    fieldnames = ['feature', 'api', 'method', 'sample'] + \
        ['timing_avg', 'timing_p25', 'timing_p75', 'timing_p90', 'timing_p99']
    csv_data = []

    network_monitor_dir = ctx.paths.network_monitor
    if not network_monitor_dir.exists():
        return

    for feature in os.listdir(network_monitor_dir):
        feature_dir = network_monitor_dir / feature
        if not feature_dir.is_dir():
            continue
        timing_file = feature_dir / API_TIMING_FILE
        if not timing_file.exists():
            continue

        with timing_file.open('r', encoding='utf-8') as f:
            feature_results = [json.loads(line) for line in f if line.strip()]

        api_timing = defaultdict(list)
        for res in feature_results:
            status = res.get('status')
            if status != 200:
                continue
            match_api = None
            for api in target_apis:
                if (
                    match_url_to_entrypoint(res['url'], api[1])
                    and res['method'].lower() == api[2]
                ):
                    match_api = api
                    break
            if not match_api:
                continue
            total = res['timing']['total']
            api_timing[match_api].append(float(total))

        for match_api, api_res in api_timing.items():
            csv_data.append(
                list(match_api) + [len(api_timing[match_api])] +
                list(calc_stats(api_res, 'timing').values())
            )

    csv_data.sort(key=lambda row: (row[0], row[1], row[2]))

    result_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(csv_data)


def _load_testcases(feature_category, testcase_list_path: Path) -> dict:
    with open(testcase_list_path, 'r', encoding='utf-8') as file:
        testcase_list = yaml.safe_load(file)
    if not testcase_list:
        return {}

    feature_testcases = next(
        (t for t in testcase_list if isinstance(t, dict) and t.get('category') == feature_category),
        None,
    )
    if not feature_testcases:
        return {}

    return feature_testcases.get('testcases', {})


def _exec_task(testcase_name, testcase_file, testcase_dir, page: Page, step_logger: Logger):
    import_path = f"{testcase_dir.replace('/', '.')}{testcase_file.replace('.py', '')}"

    try:
        module = importlib.import_module(import_path)
        function = getattr(module, testcase_name)
        step_logger.info(f"执行测试用例: {testcase_name} 来自 {testcase_file}")
        function(page)

    except ImportError as e:
        step_logger.error(f"无法导入模块 {import_path}: {e}, \n {traceback.format_exc()}")
    except AttributeError as e:
        step_logger.error(f"无法在模块 {import_path} 找到函数 {testcase_name} : {e}, \n {traceback.format_exc()}")
    except Exception as e:
        step_logger.error(
            f"错误: 执行测试用例 {testcase_name} 时发生错误: {e}, \n {traceback.format_exc()}"
        )
