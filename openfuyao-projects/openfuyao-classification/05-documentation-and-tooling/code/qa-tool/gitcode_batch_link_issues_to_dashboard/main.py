import json
from datetime import datetime
import time
import sys
import os
import logging
import yaml
import requests

# 程序功能：扫描全量仓库，批量将issue关联到对应版本的问题单看板，并配置自定义字段。也可以检查已有的自定义字段并刷新。

# 手动配置部分
# 目标看板的版本。可以填写的值包含'v25.09', 'all', 'v25.06', 'v25.03', 'v24.12', 'v24.09', 'v24.06'
CURRENT_DASHBOARD = 'v25.09'
# 是否扫描全量仓库列表
SCAN_REPOS_LIST = True    # True表示扫描全量仓库列表，并检测未关联到看板的issue，同时耗时也较久。False表示跳过扫描仓库，仅检测已有条目的自定义字段信息是否正确
# 用户账号 (用于登录GitCode) 注意用户权限较低时可能访问不到全量仓库，会影响到后续获取全量仓库列表的完整性
USER_ACCOUNT = ''
# 用户密码 (用于登录GitCode) 获取该密码哈希的方法：开启浏览器F12并手动登录GitCode，找到POST方法的login请求，将其请求body体中的password值拷贝到此处
USER_PASSWORD_WITH_HASH = ''
# 日志保留天数
LOG_FILE_KEEP_DAYS = 30
# 日志文件存储目录
# LOG_DIR = "/var/log/qa-tool/gitcode_batch_link_issues_to_dashboard"
LOG_DIR = 'C:\\Users\\Administrator\\Desktop\\openFuyao\\qa-logs'
# 里程碑列表
MILESTONE_LIST = ('v25.09', 'v25.06', 'v25.03', 'v24.12', 'v24.09', 'v24.06')


# 静态变量初始化
with open('dashboard_field_ids.yaml', 'r', encoding='utf-8') as file:
    YAML_FIELD_ID = yaml.safe_load(file)

# 变量初始化
AUTHORIZATION = ''    # 后续由gitcode_login()方法赋值
COOKIE = ''           # 后续由gitcode_login()方法赋值
dashboard_id = ''     # 对应看板id。当前id对应“openFuyao - 问题单”
dashboard_field_id_creator = ''              # “创建人”字段
dashboard_field_id_issue_create_date = ''    # “创建时间”字段
dashboard_field_id_issue_update_date = ''    # “更新时间”字段
dashboard_field_id_regress_test_failure_info = ''    # “回归不通过次数”字段
repos = []            # 需要扫描的仓库名list
issues_link_totaly_couont = 0    # 本次扫描所关联的issue总数
dashboard_items_modify_totaly_count = 0
all_items_of_specified_dashboard = []    # 指定看板中已关联的所有条目
# 公用请求头
header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/'
                  '135.0.0.0 Safari/537.36',
    'Referer': 'https://gitcode.com/',
    'Content-Type': 'application/json',
    'Accept': 'application/json, text/plain, */*',
    'Authorization': AUTHORIZATION,
    'Cookie': COOKIE
}

# 日志模块配置
# os.makedirs(LOG_DIR, exist_ok=True)  # 自动创建目录（若不存在）
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
logging.basicConfig(
    level=logging.INFO,  # 显示 INFO 及以上级别
    format="[%(asctime)s][%(levelname)s] %(message)s",  # 时间 - 级别 - 消息
    datefmt="%Y-%m-%d %H:%M:%S",  # 时间格式
    encoding='utf-8',                  # 文件编码（支持中文）
    filename=f"{LOG_DIR}/log-{current_time}.log",
    filemode="w"
)


def gitcode_login(_user_account, _user_password):
    # 通过GitCode的用户账号/密码，获取登录后的认证凭据 AUTHORIZATION, COOKIE
    global AUTHORIZATION, COOKIE
    url_gitcode_login = f'https://web-api.gitcode.com/uc/api/v1/user/oauth/login'
    header_login = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/'
                      '135.0.0.0 Safari/537.36',
        'Referer': 'https://gitcode.com/',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Encoding': 'gzip, deflate, br, zstd',
    }
    data_login = f"username={_user_account}&password={_user_password}"
    r_gitcode_login = requests.post(url_gitcode_login, headers=header_login, data=data_login)
    if r_gitcode_login.status_code == 200:
        logging.info(f'登录GitCode成功 [用户账号: {_user_account}] [响应码: {r_gitcode_login.status_code}]')
        response_header_login = r_gitcode_login.headers    # 变量类型: <class 'requests.structures.CaseInsensitiveDict'>
        set_cookie = response_header_login["Set-Cookie"]    # 变量类型: str
        logging.debug(f'响应头Set-Cookie: {set_cookie}')
        # 将set_cookie中各个键值对转化为list
        set_cookies_value = [cookie.strip() for cookie in set_cookie.split(";")]
        # 从list中寻找包含'GITCODE_ACCESS_TOKEN='的元素
        for cookie in set_cookies_value:
            if cookie.count('GITCODE_ACCESS_TOKEN='):
                authorization = cookie
        # 获取'GITCODE_ACCESS_TOKEN='等号后的字符串，并拼接为authorization的最终格式
        authorization = 'Bearer ' + authorization[authorization.find('GITCODE_ACCESS_TOKEN=') + 21:]
        # 向全局变量赋值
        AUTHORIZATION = authorization
        COOKIE = set_cookie
        # 刷新全局变量header
        header['Authorization'] = AUTHORIZATION
        header['Cookie'] = COOKIE
        logging.debug(f'AUTHORIZATION: {AUTHORIZATION}')
        logging.debug(f'COOKIE: {COOKIE}')
    elif r_gitcode_login.status_code == 401:
        logging.error(f'用户: {_user_account} 登录GitCode失败！[响应码:{r_gitcode_login.status_code}] 可能是账号/密码不正确')
    elif r_gitcode_login.status_code // 100 == 5:
        logging.error(f'用户: {_user_account} 登录GitCode失败！[响应码:{r_gitcode_login.status_code}] 可能是服务器故障')
    else:
        logging.error(f'用户: {_user_account} 登录GitCode失败！[响应码:{r_gitcode_login.status_code}] 原因未知')


def refresh_authorization(current_cookie):
    # 刷新认证凭据。当前未使用，未调试成功
    # 根据一个存活的Cookie生成新的authorization token
    url_refresh_authorization = f'https://web-api.gitcode.com/uc/api/v1/user/oauth/token'
    header_refresh_authorization = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/'
                      '135.0.0.0 Safari/537.36',
        'Referer': 'https://gitcode.com/',
        'Content-Type': 'application/json',
        'Accept': 'application/json, text/plain, */*',
        'Cookie': current_cookie
    }
    r_refresh_authorization = requests.get(url_refresh_authorization, headers=header_refresh_authorization)
    t_refresh_authorization = json.loads(r_refresh_authorization.text)
    logging.info(f't_refresh_authorization: {t_refresh_authorization}')
    new_authorization = t_refresh_authorization['access_token']
    logging.info(f'access_token刷新成功')
    return new_authorization


def check_auth():
    # 检查当前认证凭据是否有效
    url_check_auth = 'https://web-api.gitcode.com/api/v2/ac/auth/enterprises/3'
    r_check_auth = requests.get(url_check_auth, headers=header)
    logging.debug(f'r_check_auth.headers: {r_check_auth.headers}')
    if r_check_auth.status_code == 401:
        logging.error(f'认证失败，响应码{r_check_auth.status_code}。请检查AUTHORIZATION和COOKIE 认证凭据错误或已经过期')
        sys.exit(1)
    elif r_check_auth.status_code == 200:
        logging.info(f'GitCode认证凭据Check: Passed')
    else:
        logging.error(f'认证失败，响应码{r_check_auth.status_code}。请检查AUTHORIZATION和COOKIE 服务器错误或未知原因')
        sys.exit(1)


def parse_dashboard_ids(_milestone):
    # 通过用户配置的current_dashboard值，匹配dashboard和各项自定义字段的id。如果后续新建其他版本的看板，需要刷新该方法。
    global dashboard_id, dashboard_field_id_creator, dashboard_field_id_issue_create_date, \
        dashboard_field_id_issue_update_date, dashboard_field_id_regress_test_failure_info
    for tmp_milestone in MILESTONE_LIST:
        if tmp_milestone == _milestone:
            dashboard_id = YAML_FIELD_ID[_milestone]['dashboard_id'] \
                if YAML_FIELD_ID[_milestone]['dashboard_id'] else dashboard_id
            dashboard_field_id_regress_test_failure_info = YAML_FIELD_ID[_milestone]['dashboard_field_id_regress_test_failure_info'] \
                if YAML_FIELD_ID[_milestone]['dashboard_field_id_regress_test_failure_info'] else dashboard_field_id_regress_test_failure_info
        else:
            continue


def format_date(date_str):
    # 从 ISO 8601 时间格式中，提取日期和时间。用于修改创建时间、更新时间等自定义字段
    dt = datetime.fromisoformat(date_str)
    date = dt.date()  # 日期部分
    __time = dt.strftime("%H:%M")  # 时间部分（小时:分钟）
    output = str(date) + ' ' + str(__time)
    return output


def get_target_repos():
    # 获取全量仓库列表
    global repos
    try:
        # 每页最多传回50个仓库。目前仓库150+
        for page in range(1, 7):
            url_get_repos_per50 = (f'https://web-api.gitcode.com/api/v2/groups/openFuyao/projects?orgId=openFuyao&page'
                                   f'={page}&per_page=50&simple=false&include_subgroups=true&search=&with_programming_'
                                   f'language=&order_by=last_activity_at&sort=desc&tag=&repo_type=0,1,2')
            r_get_repos_per50 = requests.get(url_get_repos_per50, headers=header)
            t_get_repos_per50 = json.loads(r_get_repos_per50.text)
            for repo in t_get_repos_per50['content']:
                repos.append(repo['name'])
    except Exception as e:
        logging.error(f'尝试获取所有仓库失败。请重新运行程序 {e}')
    logging.info(f'所有仓库列表: {repos}')
    repos_count = len(repos)
    logging.info(f'仓库总数: {repos_count}')
    logging.info(f'')


def get_issues_wait_to_link_from_specified_repo(_current_repo):
    # 获取指定仓库的所有未关联任何看板的issue
    url_get_issues_from_specified_repo = \
        f'https://web-api.gitcode.com/api/v1/kanban/{dashboard_id}/project/openFuyao%2F{current_repo}/issues_list'
    r_get_issues_from_specified_repo = requests.get(url_get_issues_from_specified_repo, headers=header)
    return r_get_issues_from_specified_repo


def get_issue_extend_info(_project_id, _issue_iid):
    # 获取issue扩展字段信息（包含issue类型：缺陷/需求/测试用例）
    url_get_issue_extend_info = \
        f'https://web-api.gitcode.com/issuepr/api/v1/issue/{issue_project_id}/issue-extend/info/{issue_iid}'
    r_get_issue_extend_info = requests.get(url_get_issue_extend_info, headers=header)
    return r_get_issue_extend_info


def get_issue_labels(_repo_name, _issue_iid):
    # 获取issue的标签
    url_get_issue_labels = \
        f'https://web-api.gitcode.com/issuepr/api/v1/issue/openFuyao%2F{_repo_name}/issues/{_issue_iid}'
    r_get_issue_labels = requests.get(url_get_issue_labels, headers=header)
    return r_get_issue_labels


def link_issues_to_dashboard(_dashboard_id, _repo_name, _issue_iid):
    # 将issue关联到看板
    url_link_issues_to_dashboard = \
        f'https://web-api.gitcode.com/api/v1/kanban/{_dashboard_id}/kanban_item/add_item_of_issues_pr'
    data_link_issues_to_dashboard = {
        "project_id": f"openFuyao/{_repo_name}",
        "source_reqs": [{"source_type": 1, "source_id": _issue_iid}]    # 此处也支持多个issue在同一个请求中关联
    }
    r_link_issues_to_dashboard = requests.post(url_link_issues_to_dashboard,
                                                headers=header, data=json.dumps(data_link_issues_to_dashboard))
    return r_link_issues_to_dashboard


def get_all_items_of_specified_dashboard():
    # 获取指定看板的所有条目（即所有数据）
    items_of_specified_dashboard = []
    logging.info(f'开始获取{CURRENT_DASHBOARD}看板的所有数据item...')
    try:
        for i in range(1, 7):
            # 获取指定看板的所有items。每页200个，最多6页，即总数上限1200个
            url_get_all_items_of_specified_dashboard = \
                f'https://web-api.gitcode.com/api/v1/kanban/{dashboard_id}/kanban_item/list?page={i}&per_page=200'
            r_get_all_items_of_specified_dashboard = requests.get(
                url_get_all_items_of_specified_dashboard, headers=header)
            t_get_all_items_of_specified_dashboard = json.loads(r_get_all_items_of_specified_dashboard.text)
            if t_get_all_items_of_specified_dashboard['content']:
                items_of_specified_dashboard += t_get_all_items_of_specified_dashboard['content']
    except Exception as e:
        logging.error(f'尝试获取{CURRENT_DASHBOARD}看板的所有数据item失败 {e}')
    len_items = len(items_of_specified_dashboard)
    logging.info(f'{CURRENT_DASHBOARD}看板的所有数据item获取完毕【共计{len_items}条】 (单个看板item上限为200条)')
    return items_of_specified_dashboard


def get_specified_issue_info(repo, issue_iid):
    # 获取指定issue的信息。包含创建时间、更新时间、创建人等
    project = 'openFuyao'
    url_get_specified_issue_info = (f'https://web-api.gitcode.com/issuepr/api/v1/issue/{project}%2F{repo}'
                                    f'/issues/{issue_iid}')
    r_get_specified_issue_info = requests.get(url_get_specified_issue_info, headers=header)
    t_specified_issue_info = json.loads(r_get_specified_issue_info.text)
    return t_specified_issue_info


def get_issue_discussions(repo, issue_iid):
    # 获取指定issue的所有评论区信息（包含用户评论、issue状态/字段修改等）
    """整体逻辑：首次访问discussions接口，最多会返回20条数据（包含end_id和end_system_id字段）。此时如果产生一个类似于
    “滚动到页面底部时继续向下滚动”的操作，也就是请求更多的discussions数据，则会触发url_get_more_issue_discussions，
    以end_id和end_system_id作为入参，如此循环来获取更多discussions数据。当url_get_more_issue_discussions返回的data字段
    为空时，break循环"""
    project = 'openFuyao'
    page_size = '20'
    discussions_totally_count = 0    # 首次请求可以获取到discussions总数
    end_id = 0
    end_system_id = 0
    issue_discussions_info = []
    url_get_issue_discussions = \
        (f'https://web-api.gitcode.com/issuepr/api/v1/issue/{project}%2F{repo}/issues/{issue_iid}/discussions?issue_iid'
         f'={issue_iid}&project_id={project}%2F{repo}&type=user&page=1&per_page={page_size}&sort=asc')
    r_get_issue_discussions = requests.get(url_get_issue_discussions, headers=header)
    t_get_issue_discussions = json.loads(r_get_issue_discussions.text)
    discussions_totally_count = t_get_issue_discussions['total']    # 首次请求可以获取到discussions总数
    issue_discussions_info += t_get_issue_discussions['content']['data']
    end_id = t_get_issue_discussions['content']['end_id']
    end_system_id = t_get_issue_discussions['content']['end_system_id']
    while (True):
        # 循环获取更多discussions。当issue_discussions_info长度达到discussions_totally_count时跳出循环
        if len(issue_discussions_info) < discussions_totally_count:
            url_get_more_issue_discussions = \
                (f'https://web-api.gitcode.com/issuepr/api/v1/issue/{project}%2F{repo}/issues/{issue_iid}/discussions?'
                 f'issue_iid={issue_iid}&project_id={project}%2F{repo}&type=user&page=1&per_page={page_size}&sort=asc&'
                 f'end_id={end_id}&end_system_id={end_system_id}')
            r_get_more_issue_discussions = requests.get(url_get_more_issue_discussions, headers=header)
            t_get_more_issue_discussions = json.loads(r_get_more_issue_discussions.text)
            issue_discussions_info += t_get_more_issue_discussions['content']['data']
            end_id = t_get_more_issue_discussions['content']['end_id']
            end_system_id = t_get_more_issue_discussions['content']['end_system_id']
            continue
        elif len(issue_discussions_info) == discussions_totally_count:
            # 当issue_discussions_info长度达到discussions_totally_count时跳出循环
            break
        else:
            logging.error(f'出现预期之外的结果：获取到的discussions数量大于理论数量 请排查get_issue_discussions()方法')
            break
    return issue_discussions_info


def parse_regress_test_failure_number(issue_discussions_info):
    # 解析回归不通过次数
    key_info = []    # 从issue_discussions_info中提取指定issue中所有的用户评论类信息
    regress_test_failure_count = 0
    try:
        content = issue_discussions_info
    except Exception as e:
        logging.error(f'当前issue的discussions_info中没有content字段，请手动排查。 {e} {issue_discussions_info}')
        return '未知'
    # 先从discussions中提取所有涉及issue状态变化的信息，放入key_info
    for i in content:
        try:
            # content中部分元素不包含action字段，所以需要使用try
            if i['notes']:
                key_info += i['notes']
        except Exception as e:
            logging.debug(f'未找到key名"notes" {e}')
            continue
    logging.debug(f'问题单状态变化信息: {key_info}')
    # 从key_info中解析回归不通过次数
    for i in key_info:
        try:
            # 防止key_info中部分元素不包含body字段，所以使用try
            if '【回归不通过】' in i['body']:
                regress_test_failure_count += 1
        except Exception as e:
            logging.error(f'key_info参数中部分元素不包含body字段 {e}')
    # 由于自定义字段的值为str格式，所以这里直接转换为str返回
    return str(regress_test_failure_count)


def modify_dashboard_item_field_value(_dashboard_id, _item_id, field_id, value_to_modify):
    # 标准的自定义字段（文本类）修改方法。 可以代替dashboard_set_regress_test_failure_info()
    url_modify_dashboard_item_field_value = f'https://web-api.gitcode.com/api/v1/kanban/{_dashboard_id}/kanban_item'
    data_modify_dashboard_item_field_value = [{
        "kanban_item_id": _item_id,
        "kanban_field_id": field_id,
        "kanban_field_value": {"value": value_to_modify}
    }]
    time.sleep(0.2)
    requests.put(
        url_modify_dashboard_item_field_value, headers=header, data=json.dumps(data_modify_dashboard_item_field_value))


def modify_custom_field_null_info(items_of_specified_dashboard):
    # 修改自定义字段中的空白信息，目前仅包括'回归不通过次数'
    global dashboard_items_modify_totaly_count
    logging.info(f'')
    logging.info(f'开始获取issue详情信息并与看板item信息进行对比，填写空值并刷新数据..')
    for item in items_of_specified_dashboard:
        issue_url = item['values'][0]['value']['url']
        tmp_repo = issue_url.split('/')[-3]
        tmp_issue_iid = issue_url.split('/')[-1]
        # 获取issue的操作信息(包含状态变化信息)。用于统计问题单回归次数
        t_get_issue_discussions = get_issue_discussions(repo=tmp_repo, issue_iid=tmp_issue_iid)
        # 解析回归不通过次数
        regress_test_failure_info = parse_regress_test_failure_number(issue_discussions_info=t_get_issue_discussions)
        # 获取看板中自定义字段的当前值
        for value in item['values']:
            match value['field_name']:
                case '回归不通过次数':
                    current_item_regress_test_failure_info = value['value']
                    current_item_regress_test_failure_info_field_id = value['field_id']
        # 判断看板中字段值与从issue详情中信息是否一致，如果不一致则修改
        item_id = item['id']
        if current_item_regress_test_failure_info != regress_test_failure_info:
            logging.info(f'')
            logging.info(f'[检测到待修改条目][{item["id"]}] ({item["values"][0]["value"]["url"]}) '
                         f'{item["values"][0]["value"]["title"]}')
            dashboard_items_modify_totaly_count += 1
        if current_item_regress_test_failure_info != regress_test_failure_info:
            # 如果看板回归不通过次数和issue详情的回归不通过次数不一致，则修改看板
            modify_dashboard_item_field_value(
                _dashboard_id=dashboard_id,
                _item_id=item_id,
                field_id=current_item_regress_test_failure_info_field_id,
                value_to_modify=regress_test_failure_info
            )
            logging.info(f'修改看板字段: 回归不通过次数 - 由\'{current_item_regress_test_failure_info}\'修改为\''
                  f'{regress_test_failure_info}\'')
    logging.info(f'')
    logging.info(f'{CURRENT_DASHBOARD}看板中所有信息已填写+刷新完毕！')
    logging.info(f'')


def dashboard_set_regress_test_failure_info(_item_id, _regress_test_failure_info):
    # 修改“回归不通过次数”自定义字段
    _url_dashboard_edit = f'https://web-api.gitcode.com/api/v1/kanban/{dashboard_id}/kanban_item'
    data_dashboard_set_regress_test_failure_info = [{
        "kanban_item_id": _item_id,
        "kanban_field_id": dashboard_field_id_regress_test_failure_info,
        "kanban_field_value": {"value": _regress_test_failure_info}
    }]
    requests.put(_url_dashboard_edit, headers=header, data=json.dumps(data_dashboard_set_regress_test_failure_info))


def clean_logs_by_time(_log_dir, _keep_days=LOG_FILE_KEEP_DAYS):
    """按时间保留最近 N 天的日志（删除更早文件）"""
    cutoff_time = datetime.now().timestamp() - _keep_days * 86400  # 86400为1天
    # 收集所有日志文件（排除当前写入的 app.log）
    try:
        for filename in os.listdir(_log_dir):
            file_path = os.path.join(_log_dir, filename)
            if os.path.isfile(file_path):
                mtime = os.path.getmtime(file_path)
                if mtime < cutoff_time:
                    os.remove(file_path)
                    logging.info(f"[删除过期日志文件][超过{_keep_days}天]: {file_path}")
    except Exception as e:
        logging.error(f'日志清理失败 {e}')


def print_time_spent(_time_start):
    # 耗时统计
    _time_end = time.time()
    _time_spent = _time_end - _time_start
    _time_spent = round(_time_spent, 3)
    if _time_spent > 60:
        # 大于60秒时，转化为x分x秒
        _time_spent = round(_time_spent)  # 四舍五入去掉小数位
        _time_min = _time_spent // 60
        _time_sec = _time_spent % 60
        logging.info(f'程序运行结束。总耗时： {_time_min} 分 {_time_sec} 秒')
    else:
        logging.info(f'程序运行结束。总耗时： {_time_spent} 秒')


if __name__ == "__main__":
    logging.info(f'### Gitcode Batch Link Issues to Dashboard ###')
    time_start = time.time()
    gitcode_login(USER_ACCOUNT, USER_PASSWORD_WITH_HASH)
    check_auth()
    logging.info(f'')
    # time_start_pretty = time.strftime("%Y/%m/%d %H:%M", time.localtime())
    # 根据用户配置的current_dashboard匹配各项自定义字段的id
    parse_dashboard_ids(CURRENT_DASHBOARD)
    if SCAN_REPOS_LIST == True:
        # 获取需要扫描的repo名称列表。默认为全部repo
        logging.info(f'根据配置 [SCAN_REPOS_LIST == True] 开始扫描全量仓库，并扫描所有未关联到看板的问题单issue')
        get_target_repos()
    elif SCAN_REPOS_LIST == False:
        logging.info(f'根据配置 [SCAN_REPOS_LIST == False] 跳过扫描全量仓库，并跳过扫描未关联到看板的问题单issue')
    else:
        logging.error(f'"SCAN_REPOS_LIST"赋值错误！（仅允许赋值为True或False）')
    # 遍历仓库，解析待关联的issue，并关联到看板，然后修改自定义字段
    for current_repo in repos:
        logging.info(f'当前仓库: {current_repo}')
        # 从指定仓库获取所有待添加的issue列表
        # 获取指定仓库的所有未关联任何看板的issue
        r_get_issues_wait_to_link_from_specified_repo = get_issues_wait_to_link_from_specified_repo(current_repo)
        # issues表示当前仓库中，待关联到问题单看板的issue编号列表
        issues = []
        try:
            logging.debug(f'该仓库还未关联到看板的issues: {r_get_issues_wait_to_link_from_specified_repo.text}')
            # 获取指定仓库中，待关联的issues列表
            # issues = filter_issues_list_wait_to_link(r_get_issues_wait_to_link_from_specified_repo)
            issues_wait_to_link_from_specified_repo = json.loads(
                r_get_issues_wait_to_link_from_specified_repo.text)['result']
            for issue in issues_wait_to_link_from_specified_repo:
                issue_iid = issue['iid']
                issue_project_id = issue['project_id']
                # 检查issue的type是否为“缺陷”。防止把其他类型的issue(如“任务”、“需求”)关联到问题单看板。
                r_get_issue_extend_info = get_issue_extend_info(issue['project_id'], issue['iid'])
                # 获取指定issue的所有labels
                r_get_issue_labels = get_issue_labels(current_repo, issue_iid)
                t_get_issue_labels = json.loads(r_get_issue_labels.text)
                if 'enterprise_labels' in t_get_issue_labels:
                    for current_label in t_get_issue_labels['enterprise_labels']:
                        # 先判断labels中包含版本标签
                        if 'name' in current_label:
                            if CURRENT_DASHBOARD in current_label['name']:
                                tmp_name = current_label['name']
                                # 判断issue是否存在“issue类型”扩展字段。如果存在，则继续判断类型是否为“缺陷”
                                if r_get_issue_extend_info.text:
                                    t_check_issue_type = json.loads(r_get_issue_extend_info.text)
                                    if 'type' in t_check_issue_type and t_check_issue_type['type'] == '缺陷':
                                        logging.info(
                                            f'Checking... {issue_iid}#: Label【{tmp_name}】符合预期，且issue类型是'
                                            f'“缺陷” 【{issue_iid}#可以关联到看板】')
                                        issues.append(issue_iid)
                                        issues_link_totaly_couont += 1
                                    else:
                                        logging.info(
                                            f'Checking.. 【{issue_iid}#】: 当前issue有扩展字段，但issue类型不是“缺陷”'
                                            f'。无需关联到问题单看板')
                                else:
                                    logging.info(
                                        f'Checking.. 【{issue_iid}#】: Label【{tmp_name}】符合预期。但是当前issue没有'
                                        f'扩展字段，推测属于Gitee迁移issue 【{issue_iid}#可以关联到看板】')
                                    issues.append(issue_iid)
                                    issues_link_totaly_couont += 1
                            else:
                                pass
                        else:
                            logging.error(f'程序异常：enterprise_labels字段中未检测到name字段')
                else:
                    logging.error(f'程序异常：未检测到enterprise_labels字段key名')
        except Exception as e:
            logging.info(f'当前仓库没有待关联到看板的问题单issue {e}')
        logging.info(f'待关联问题单issue编号: {issues}')
        # 遍历待关联的issue列表，依次关联到看板，并配置看板的自定义字段
        for current_issue in issues:
            # 将指定issue关联到看板
            r_link_issues_to_dashboard = link_issues_to_dashboard(dashboard_id, current_repo, current_issue)
            logging.debug(f'link_issue_to_dashboard: {r_link_issues_to_dashboard.text}')
            # 获取issue的操作信息(包含状态变化信息)。用于统计问题单回归次数
            t_get_issue_discussions = get_issue_discussions(repo=current_repo, issue_iid=current_issue)
            # 解析回归不通过次数
            regress_test_failure_info = (
                parse_regress_test_failure_number(issue_discussions_info=t_get_issue_discussions))
            # 获取看板的当前行id，用于后续修改
            item_id = json.loads(r_link_issues_to_dashboard.text)[0]["id"]
            # 修改看板 - 回归不通过次数
            modify_dashboard_item_field_value(
                _dashboard_id=dashboard_id,
                _item_id=item_id,
                field_id=dashboard_field_id_regress_test_failure_info,
                value_to_modify=regress_test_failure_info
            )
            logging.info(f'[issue_iid:{issue_iid}] 关联issue + 修改看板自定义字段【OK】')
        logging.info('')

    logging.info(f'开始check并刷新看板中已有的自定义字段信息（回归不通过次数）')
    all_items_of_specified_dashboard = get_all_items_of_specified_dashboard()
    modify_custom_field_null_info(all_items_of_specified_dashboard)

    logging.info(f'************************************************************')
    logging.info(f'{CURRENT_DASHBOARD}问题单issue关联看板 + 看板信息填充/刷新【已完成】')
    logging.info(f'本次关联问题单issue共计：{issues_link_totaly_couont}个')
    logging.info(f'本次修改看板item共计：{dashboard_items_modify_totaly_count}条')
    # 时间模块，统计程序执行耗时
    print_time_spent(time_start)
    check_auth()
    logging.info(f'************************************************************')
    clean_logs_by_time(LOG_DIR)

