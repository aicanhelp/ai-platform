from typing import Any
from urllib.parse import urlparse


def first_positive(*args: Any) -> int:
    """
    返回第一个正数，如果没有正数则返回 -1
    
    :param args: 任意数量的数值参数
    :return: 第一个正数，如果没有正数则返回 -1
    """
    for value in args:
        if value > 0:
            return value
    return -1


def match_url_to_entrypoint(url: str, endpoint: str) -> bool:
    parsed_url = urlparse(url)
    url_path = parsed_url.path
    url_parts = url_path.strip('/').split('/')
    endpoint_parts = endpoint.strip('/').split('/')
    # url path的段数不会少于接口段数，直接判定为False
    if len(url_parts) < len(endpoint_parts):
        return False
    # 只比较最后的部分，便于zip遍历
    url_parts = url_parts[-len(endpoint_parts):]

    for url_part, endpoint_part in zip(url_parts, endpoint_parts):
        # 路径参数以{}包裹，可以匹配任意值
        if endpoint_part.startswith('{') and endpoint_part.endswith('}'):
            continue
        if url_part != endpoint_part:
            return False
    return True