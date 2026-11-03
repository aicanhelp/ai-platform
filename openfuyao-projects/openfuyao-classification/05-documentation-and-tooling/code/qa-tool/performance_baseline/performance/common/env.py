from typing import Any
import os
from dotenv import load_dotenv

DEFAULT_SSH_PORT = 22
STORAGE_PATH = '.cache/storage_state.json'


def load_pw_config(env=None) -> dict[str, dict]:
    if env:
        load_dotenv(env)

    base_url = os.getenv('BASE_URL')
    test_username = os.getenv('TEST_USERNAME')
    test_password = os.getenv('TEST_PASSWORD')
    if not base_url:
        raise Exception('Test target is not set, specify the BASE_URL in .env.test')
    if not test_username or not test_password:
        raise Exception('Please set TEST_USERNAME and TEST_PASSWORD in .env.test')
    return {
        'context': {
           'base_url': base_url,
           'storage_state': STORAGE_PATH,
        },
        'credentials': {
            'username': test_username,
            'password': test_password,
        }
    }


def load_ssh_config(env=None) -> dict[str, Any]:
    if env:
        load_dotenv(env)

    hostname = os.getenv('KUBECONFIG_SFTP_HOST')
    port = int(os.getenv('KUBECONFIG_SFTP_PORT', DEFAULT_SSH_PORT))
    username = os.getenv('KUBECONFIG_SFTP_USER')
    password = os.getenv('KUBECONFIG_SFTP_PASS')

    if not all([hostname, port, username, password]):
        raise ValueError('SFTP参数不完整，请在 .env 文件提供 host, port, username, password')

    return {
        "hostname": hostname,
        "port": port,
        "username": username,
        "password": password
    }


def load_copy_task(env=None) -> tuple[str, str]:
    if env:
        load_dotenv(env)

    remote_path = os.getenv('KUBECONFIG_SFTP_REMOTE_PATH', "/root/.kube/config")
    kube_config_filename = remote_path.split("/")[-1]
    local_path = os.getenv('KUBECONFIG_LOCAL_PATH', f"./{kube_config_filename}")

    if not all([remote_path, local_path]):
        raise ValueError('SFTP参数不完整，请在 .env 文件提供 remote_path, local_path')

    return remote_path, local_path
