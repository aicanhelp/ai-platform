import os
from pathlib import Path

from dotenv import load_dotenv


ENV_FILE = './.env'

load_dotenv(ENV_FILE)

CWD_PATH = Path.cwd()
CACHE_PATH = CWD_PATH / '.cache'

# kubeconfig文件本地路径
KUBECONFIG_PATH = CACHE_PATH / os.environ.get('KUBECONFIG_SFTP_LOCAL_PATH', 'config')
# 资源监控延迟启动窗口（hour）；schedule_and_run_all_features(delayed=True) 时使用
RESOURCE_MONITOR_DELAY_PERIOD = 0
# 资源监控采集集群数据的间隔时间
RESOURCE_MONITOR_INTERVAL = 5
# 资源监控采集失败最大次数，设置为0则无上限
RESOURCE_MONITOR_FAILED_THRESHOLD = 3
