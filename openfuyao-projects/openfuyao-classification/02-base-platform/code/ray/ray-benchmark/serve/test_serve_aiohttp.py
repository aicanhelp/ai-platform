import asyncio
import aiohttp
import pytest
import time


@pytest.mark.asyncio
async def test_throughput(start_ray_serve, serve_metrics_recorder):
    # TODO: get url from start_ray_serve fixture
    url = "http://localhost:8000/"
    concurrency = 100
    test_duration = 20

    async def worker(session, counter):
        while True:
            try:
                async with session.get(url) as response:
                    await response.read()
                    counter[0] += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"请求失败: {e}")

    async with aiohttp.ClientSession() as session:
        counters = [[0] for _ in range(concurrency)]
        tasks = []

        for i in range(concurrency):
            task = asyncio.create_task(worker(session, counters[i]))
            tasks.append(task)

        start_time = time.time()

        await asyncio.sleep(test_duration)

        for task in tasks:
            task.cancel()

        # 等待所有任务完成
        await asyncio.gather(*tasks, return_exceptions=True)

        # 记录结束时间
        end_time = time.time()

        # 计算总请求数和吞吐量
        total_requests = sum(counter[0] for counter in counters)
        elapsed_time = end_time - start_time
        throughput = total_requests / elapsed_time

        serve_metrics_recorder(
            total_requests=total_requests,
            elapsed_time=elapsed_time,
            throughput=throughput,
        )
