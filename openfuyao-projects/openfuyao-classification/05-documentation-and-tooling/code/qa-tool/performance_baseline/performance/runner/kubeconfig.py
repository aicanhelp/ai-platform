import os

from kubernetes import client

from performance.runner.constants import KUBECONFIG_PATH


def fetch_and_save_kubeconfig(host=None, port=22, username=None, password=None, remote_path=None):
    """
    使用 SFTP 从远程服务器下载 kubeconfig 文件并保存到本地。
    参数可通过环境变量或函数参数传递。
    """
    # 支持从环境变量读取参数
    host = host or os.environ.get('KUBECONFIG_SFTP_HOST')
    port = int(os.environ.get('KUBECONFIG_SFTP_PORT', port))
    username = username or os.environ.get('KUBECONFIG_SFTP_USER')
    password = password or os.environ.get('KUBECONFIG_SFTP_PASS')
    remote_path = remote_path or os.environ.get('KUBECONFIG_SFTP_REMOTE_PATH')

    if not all([host, username, password, remote_path]):
        raise ValueError('SFTP参数不完整，请提供 host, username, password, remote_path')
    
    remote_server_info = {
        "host": host,
        "port": port,
        "username": username,
        "password": password
    }

    sftp_get(remote_server_info, remote_path, str(KUBECONFIG_PATH))


def sftp_get(remote_server_info, remote_path, local_path):
    """
    使用 SFTP 从远程服务器下载文件并保存到本地。
    如果本地路径不存在，会创建所需的目录结构和空白文件。
    """
    import paramiko
    from pathlib import Path
    
    local_config_path = Path(local_path)
    
    if not local_config_path.exists():
        local_config_path.parent.mkdir(parents=True, exist_ok=True)
        local_config_path.touch()
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=remote_server_info["host"],
        port=remote_server_info["port"],
        username=remote_server_info["username"],
        password=remote_server_info["password"])
    sftp = ssh.open_sftp()
    sftp.get(remote_path, local_path)
    sftp.close()
    ssh.close()


def validate_kubeconfig():
    """
    验证当前 kubeconfig 文件内容是否能正常访问集群。
    :raises: Exception 如果 kubeconfig 无效则抛出异常。
    """
    try:
        core_v1 = client.CoreV1Api()
        core_v1.list_namespace(_request_timeout=5)
        return True
    except Exception as e:
        return False
