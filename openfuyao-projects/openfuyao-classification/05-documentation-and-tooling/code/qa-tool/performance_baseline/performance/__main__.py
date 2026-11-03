import argparse
import sys
from pathlib import Path
import yaml

from performance.runner.session import create_session
from performance.runner.runner import PerformanceTestRunner


def _load_resource_categories(resource_category_path: Path) -> list[str]:
    with open(resource_category_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or []
    return [item['category'] for item in data if isinstance(item, dict) and item.get('category')]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Performance Test Runner')
    parser.add_argument('--testcase', type=str, default='./testcases',
                        help='测试用例目录路径 (默认: ./testcases)')
    parser.add_argument(
        '--mode',
        choices=['featured', 'resource'],
        required=True,
        help='运行模式：featured=网络+资源双监控（需 --features）；resource=仅资源监控（需 --duration）',
    )
    parser.add_argument(
        '--features',
        type=str,
        default=None,
        help='特性名，多个用英文逗号分隔，须与用例配置中的 category 一致。featured 模式下必填',
    )
    parser.add_argument(
        '--headless',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='是否以无头模式启动浏览器（默认：开启无头，使用 --no-headless 显示浏览器窗口）',
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        metavar='SECONDS',
        help='资源监控运行时长（秒）；0 表示无限运行。resource 模式下必填',
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='写入 performance_test.log、step.log 的级别（默认：INFO）',
    )
    parser.add_argument(
        '--console-level',
        type=str,
        default='WARNING',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='stdout 输出级别（默认：WARNING）',
    )
    parser.add_argument(
        '--log-dir',
        type=Path,
        default=Path('./performance_log'),
        help='会话根目录（默认：./performance_log）',
    )
    parser.add_argument(
        '--log-cdp',
        action='store_true',
        default=False,
        help='仅 network：写入 cdp_events/{case}.jsonl',
    )
    args = parser.parse_args()

    session = create_session(
        work_dir=args.log_dir,
        log_level=args.log_level,
        console_level=args.console_level,
        enable_cdp_log=args.log_cdp,
    )
    general_logger = session.general_logger()

    testcase_path = Path(args.testcase)
    if not testcase_path.exists():
        general_logger.error(f"错误: 测试用例路径不存在: {testcase_path}")
        sys.exit(1)

    resource_category_path = testcase_path / 'resource_pod_category.yaml'

    if args.mode == 'featured':
        if not args.features or not args.features.strip():
            general_logger.error("错误: featured 模式必须通过 --features 指定特性名")
            sys.exit(1)
        if args.duration is not None:
            general_logger.warning('--duration 仅在 resource 模式下生效，当前将被忽略')
        features = [f.strip() for f in args.features.split(',') if f.strip()]
        duration_seconds = None
    else:
        if args.duration is None:
            general_logger.error("错误: resource 模式必须通过 --duration 指定运行时长（0 表示无限）")
            sys.exit(1)
        if args.features and args.features.strip():
            features = [f.strip() for f in args.features.split(',') if f.strip()]
        else:
            features = _load_resource_categories(resource_category_path)
            if not features:
                general_logger.error(f"错误: 未指定 --features，且 {resource_category_path} 中无可用 category")
                sys.exit(1)
            general_logger.info(f"resource 模式未指定 --features，将监控全部 category: {', '.join(features)}")
        duration_seconds = args.duration

    if not features:
        general_logger.error("错误: 至少需要一个有效的特性名")
        sys.exit(1)

    runner = PerformanceTestRunner(
        testcase_path=testcase_path,
        features=features,
        headless=args.headless,
        monitor_mode=args.mode,
        duration_seconds=duration_seconds,
        session=session,
    )
    runner.schedule_and_run_all_features()
