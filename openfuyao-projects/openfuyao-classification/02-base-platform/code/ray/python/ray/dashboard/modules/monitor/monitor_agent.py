import asyncio
import ray.dashboard.utils as dashboard_utils
import logging
import psutil
import ray._private.ray_constants as ray_constants

logger = logging.getLogger(__name__)


class monitorAgent(dashboard_utils.DashboardAgentModule):
    def init(self, dashboard_agent):
        super().init(dashboard_agent)

    def get_all_children_recursive(self, pid):
        try:
            process = psutil.Process(pid)
            children = process.children(recursive=True)
            return {child.pid: pid for child in children}
        except psutil.NoSuchProcess:
            return {}

    async def run(self, server):
        target_pid = psutil.Process().ppid()
        monitored = self.get_all_children_recursive(target_pid)
        RAY_FORK_MONITOR_AGENT = ray_constants.env_bool("RAY_FORK_MONITOR_AGENT", False)
        while RAY_FORK_MONITOR_AGENT:
            new_children = self.get_all_children_recursive(target_pid)

            # 合并新发现的子进程到监控列表
            for child_pid, parent_pid in new_children.items():
                if child_pid not in monitored:
                    monitored[child_pid] = parent_pid
                    logger.info(f"发现新子进程: PID={child_pid}, 父进程={parent_pid}")

            # 检查所有被监控进程的父进程状态
            for pid in list(monitored.keys()):
                try:
                    proc = psutil.Process(pid)
                    current_ppid = proc.ppid()

                    # 如果父进程变为 1，且原始父进程不是 1（避免误杀系统进程）
                    if current_ppid == 1 and monitored[pid] != 1:
                        logger.info(
                            f"检测到孤儿进程 PID={pid} (原父进程={monitored[pid]}), 正在终止..."
                        )
                        proc.kill()
                        del monitored[pid]
                except psutil.NoSuchProcess:
                    del monitored[pid]
                    logger.info(f"进程已退出: PID={pid}")

            # 清理已退出的进程记录
            alive_pids = psutil.pids()
            dead_pids = [pid for pid in monitored if pid not in alive_pids]
            for pid in dead_pids:
                del monitored[pid]

            await asyncio.sleep(5)

    @staticmethod
    def is_minimal_module():
        return False
