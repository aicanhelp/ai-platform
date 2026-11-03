import pytest
import ray
from ray import serve
from starlette.requests import Request


metrics_data = {}

# 可以创建一个 fixture 来帮助记录指标，但这并非强制，测试函数也可以直接更新 metrics_data
@pytest.fixture(scope="function")  # function scope: 每个测试函数获取一个新的 recorder 实例（如果需要）
def serve_metrics_recorder():
    """提供一个简单的函数来记录指标"""

    def record(total_requests=0, elapsed_time=0, throughput=0):
        metrics_data["total_requests"] = total_requests
        metrics_data["elapsed_time"] = elapsed_time
        metrics_data["throughput"] = throughput

    return record


# 实现 pytest_terminal_summary 钩子
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """在测试运行结束时向终端摘要添加自定义指标。"""
    terminalreporter.ensure_newline()
    terminalreporter.section("Ray Serve Benchmark", sep="=", bold=True)

    if metrics_data:
        total_requests = metrics_data.get("total_requests", 0)
        elapsed_time = metrics_data.get("elapsed_time", 0)
        throughput = metrics_data.get("throughput", 0)

        terminalreporter.write_line(f"Total processed reqeusts: {total_requests}")
        terminalreporter.write_line(f"Total elapsed time: {elapsed_time:.2f} sec")
        terminalreporter.write_line(f"Throughput: {throughput:.2f} req/sec")
    else:
        terminalreporter.write_line("No custom metrics recorded.")

    terminalreporter.section("", sep="=", bold=True)

    terminalreporter.ensure_newline()


@pytest.fixture()
def start_ray_serve(num_replicas=2):
    ray.init()

    @serve.deployment(num_replicas=num_replicas)
    class DeploymentExample:
        async def __call__(self, request: Request):
            return "hi"

    app = DeploymentExample.bind()
    serve.run(app)

    yield

    serve.shutdown()
    ray.shutdown()
