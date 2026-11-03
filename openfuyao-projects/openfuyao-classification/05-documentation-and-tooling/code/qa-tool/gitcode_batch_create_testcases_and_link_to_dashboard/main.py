import json
import time
from datetime import datetime
import logging
import yaml
import sys
import os
import requests
from openpyxl import load_workbook

# 程序功能：读取 EXCEL_FILE_PATH 中的测试用例，并在 REPO_NAME 仓库创建测试用例issue，并将测试用例issue关联到对应里程碑的测试用例看板。

# 必须配置部分
# Excel中必须存在这几列："里程碑", "功能模块", "用例编号", "用例名称", "维护人", "用例类型", "优先级", "前置条件", "备注", "步骤描述", "预期结果"
# Excel的标题行必须是第1行
EXCEL_FILE_PATH = 'C:\\Users\\Administrator\\Desktop\\openFuyao\\测试用例批量上传-测试.xlsx'
# 创建issue的目标repo（测试用例默认放在test-case仓库）
REPO_NAME = 'test-case'
# 用户账号 (用于登录GitCode) 注意用户权限较低时可能访问不到全量仓库，会影响到后续获取全量仓库列表的完整性
USER_ACCOUNT = ''
# 用户密码 (用于登录GitCode) 获取该密码哈希的方法：开启浏览器F12并手动登录GitCode，找到POST方法的login请求，将其请求body体中的password值拷贝到此处
USER_PASSWORD_WITH_HASH = ''
# 日志保留天数
LOG_FILE_KEEP_DAYS = 30
# 日志文件存储目录
# LOG_DIR = "/var/log/qa-tool/gitcode_batch_create_testcases_and_link_to_dashboard"
LOG_DIR = 'C:\\Users\\Administrator\\Desktop\\openFuyao\\qa-logs'
# 里程碑列表
MILESTONE_LIST = ('v25.09', 'v25.06', 'v25.03', 'v24.12', 'v24.09', 'v24.06')


# 静态变量初始化
with open('dashboard_field_ids.yaml', 'r', encoding='utf-8') as file:
    YAML_FIELD_ID = yaml.safe_load(file)
with open('user_info.yaml', 'r', encoding='utf-8') as file2:
    YAML_USER_INFO = yaml.safe_load(file2)

# 看板相关变量初始化
AUTHORIZATION = ''  # 后续由gitcode_login()方法赋值
COOKIE = ''  # 后续由gitcode_login()方法赋值
dashboard_id = ''  # 对应看板id。可以从看板详情的http请求/api/v1/kanban/iid_kanban_detail返回值中，id字段获取
dashboard_field_id_testcase_id = ''  # 用例编号
dashboard_field_id_testcase_priority = ''  # 用例优先级
testcase_priority_p0 = ''  # 用例优先级 - P0
testcase_priority_p1 = ''  # 用例优先级 - P1
testcase_priority_p2 = ''  # 用例优先级 - P2
testcase_priority_p3 = ''  # 用例优先级 - P3
dashboard_field_id_module = ''  # 功能模块
dashboard_field_id_milestone_v = ''  # 里程碑版本（注意这是自定义字段，不是默认的“里程碑”字段。因为默认字段维护侧还没建立完毕）
milestone_v2406 = ''  # 里程碑版本 - v24.06
milestone_v2409 = ''  # 里程碑版本 - v24.09
milestone_v2412 = ''  # 里程碑版本 - v24.12
milestone_v2503 = ''  # 里程碑版本 - v25.03
milestone_v2506 = ''  # 里程碑版本 - v25.06
dashboard_field_id_testcase_type = ''  # 用例类型
dashboard_field_id_testcase_maintainer = ''  # 维护人。对应sheet页内，各个列名
dashboard_field_id_testcase_review_state = ''  # 用例评审状态
testcase_review_state_draft = ''  # 用例评审状态 - 草稿
testcase_review_state_reviewing = ''  # 用例评审状态 - 评审中
testcase_review_state_complete = ''  # 用例评审状态 - 评审通过
testcase_review_state_discarded = ''  # 用例评审状态 - 已废弃
dashboard_field_id_testcase_create_date = ''  # 测试用例创建时间
dashboard_field_id_pre_request = ''  # 前置条件
dashboard_field_id_test_steps = ''  # 步骤描述
dashboard_field_id_expected_result = ''  # 预期结果
dashboard_field_id_remark = ''  # 备注
# 创建issue需要的参数初始化
username = ''
object_id = ''
nickname = ''
email = ''
assignee_ids = ''
mobile = ''
testcase_priority = ''
milestone_v = ''
milestone_id = ''
issue_priority = ''
issue_title = ''
issue_description = ''
issue_type = ''
issue_lable = ''
confidential = ''
# 其他参数
all_items_of_dashboard = []
# 公用请求头
header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                  ' Chrome/135.0.0.0 Safari/537.36',
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
    encoding='utf-8',  # 文件编码（支持中文）
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
        response_header_login = r_gitcode_login.headers  # 变量类型: <class 'requests.structures.CaseInsensitiveDict'>
        set_cookie = response_header_login["Set-Cookie"]  # 变量类型: str
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
        logging.error(
            f'用户: {_user_account} 登录GitCode失败！[响应码:{r_gitcode_login.status_code}] 可能是账号/密码不正确')
    elif r_gitcode_login.status_code // 100 == 5:
        logging.error(f'用户: {_user_account} 登录GitCode失败！[响应码:{r_gitcode_login.status_code}] 可能是服务器故障')
    else:
        logging.error(f'用户: {_user_account} 登录GitCode失败！[响应码:{r_gitcode_login.status_code}] 原因未知')


def check_auth():
    url_check_auth = 'https://web-api.gitcode.com/api/v2/ac/auth/enterprises/3'
    r_check_auth = requests.get(url_check_auth, headers=header)
    logging.debug(f'r_check_auth.headers: {r_check_auth.headers}')
    if r_check_auth.status_code == 401:
        logging.error(f'认证失败，响应码{r_check_auth.status_code}。请检查AUTHORIZATION和COOKIE，可能已经过期')
        sys.exit(1)
        # 如果token过期，则重新请求并刷新
        # logging.info(f'尝试获取新的access_token..')
        # new_authorization = refresh_authorization(COOKIE)
        # AUTHORIZATION = 'Bearer ' + new_authorization
        # logging.info(f'new AUTHORIZATION: \'{AUTHORIZATION}\'')
    elif r_check_auth.status_code == 200:
        logging.info(f'GitCode认证凭据有效性check: Passed')
    else:
        logging.error(f'认证失败，响应码{r_check_auth.status_code}。请检查AUTHORIZATION和COOKIE格式填写是否正确')
        sys.exit(1)


def parse_target_dashboard_ids(milestone):
    global dashboard_id, dashboard_field_id_testcase_id, dashboard_field_id_testcase_priority, \
        testcase_priority_p0, testcase_priority_p1, testcase_priority_p2, testcase_priority_p3, \
        dashboard_field_id_module, dashboard_field_id_milestone_v, \
        milestone_v2406, milestone_v2409, milestone_v2412, milestone_v2503, milestone_v2506, \
        dashboard_field_id_testcase_type, dashboard_field_id_testcase_maintainer, \
        dashboard_field_id_testcase_review_state, testcase_review_state_draft, testcase_review_state_reviewing, \
        testcase_review_state_complete, testcase_review_state_discarded, \
        dashboard_field_id_testcase_create_date, dashboard_field_id_pre_request, dashboard_field_id_test_steps, \
        dashboard_field_id_expected_result, dashboard_field_id_remark
    for tmp_milestone in MILESTONE_LIST:
        if tmp_milestone == milestone:
            dashboard_id = YAML_FIELD_ID[milestone]['dashboard_id'] \
                if YAML_FIELD_ID[milestone]['dashboard_id'] else dashboard_id
            dashboard_field_id_testcase_id = YAML_FIELD_ID[milestone]['dashboard_field_id_testcase_id'] \
                if YAML_FIELD_ID[milestone]['dashboard_field_id_testcase_id'] else dashboard_field_id_testcase_id
            dashboard_field_id_testcase_priority = YAML_FIELD_ID[milestone]['dashboard_field_id_testcase_priority'] \
                if YAML_FIELD_ID[milestone][
                'dashboard_field_id_testcase_priority'] else dashboard_field_id_testcase_priority
            testcase_priority_p0 = YAML_FIELD_ID[milestone]['testcase_priority_p0'] \
                if YAML_FIELD_ID[milestone]['testcase_priority_p0'] else testcase_priority_p0
            testcase_priority_p1 = YAML_FIELD_ID[milestone]['testcase_priority_p1'] \
                if YAML_FIELD_ID[milestone]['testcase_priority_p1'] else testcase_priority_p1
            testcase_priority_p2 = YAML_FIELD_ID[milestone]['testcase_priority_p2'] \
                if YAML_FIELD_ID[milestone]['testcase_priority_p2'] else testcase_priority_p2
            testcase_priority_p3 = YAML_FIELD_ID[milestone]['testcase_priority_p3'] \
                if YAML_FIELD_ID[milestone]['testcase_priority_p3'] else testcase_priority_p3
            dashboard_field_id_module = YAML_FIELD_ID[milestone]['dashboard_field_id_module'] \
                if YAML_FIELD_ID[milestone]['dashboard_field_id_module'] else dashboard_field_id_module
            dashboard_field_id_testcase_type = YAML_FIELD_ID[milestone]['dashboard_field_id_testcase_type'] \
                if YAML_FIELD_ID[milestone]['dashboard_field_id_testcase_type'] else dashboard_field_id_testcase_type
            dashboard_field_id_testcase_maintainer = YAML_FIELD_ID[milestone]['dashboard_field_id_testcase_maintainer'] \
                if YAML_FIELD_ID[milestone][
                'dashboard_field_id_testcase_maintainer'] else dashboard_field_id_testcase_maintainer
            dashboard_field_id_testcase_review_state = YAML_FIELD_ID[milestone][
                'dashboard_field_id_testcase_review_state'] \
                if YAML_FIELD_ID[milestone][
                'dashboard_field_id_testcase_review_state'] else dashboard_field_id_testcase_review_state
            testcase_review_state_draft = YAML_FIELD_ID[milestone]['testcase_review_state_draft'] \
                if YAML_FIELD_ID[milestone]['testcase_review_state_draft'] else testcase_review_state_draft
            testcase_review_state_reviewing = YAML_FIELD_ID[milestone]['testcase_review_state_reviewing'] \
                if YAML_FIELD_ID[milestone]['testcase_review_state_reviewing'] else testcase_review_state_reviewing
            testcase_review_state_complete = YAML_FIELD_ID[milestone]['testcase_review_state_complete'] \
                if YAML_FIELD_ID[milestone]['testcase_review_state_complete'] else testcase_review_state_complete
            testcase_review_state_discarded = YAML_FIELD_ID[milestone]['testcase_review_state_discarded'] \
                if YAML_FIELD_ID[milestone]['testcase_review_state_discarded'] else testcase_review_state_discarded
            dashboard_field_id_testcase_create_date = YAML_FIELD_ID[milestone][
                'dashboard_field_id_testcase_create_date'] \
                if YAML_FIELD_ID[milestone][
                'dashboard_field_id_testcase_create_date'] else dashboard_field_id_testcase_create_date
            dashboard_field_id_pre_request = YAML_FIELD_ID[milestone]['dashboard_field_id_pre_request'] \
                if YAML_FIELD_ID[milestone]['dashboard_field_id_pre_request'] else dashboard_field_id_pre_request
            dashboard_field_id_test_steps = YAML_FIELD_ID[milestone]['dashboard_field_id_test_steps'] \
                if YAML_FIELD_ID[milestone]['dashboard_field_id_test_steps'] else dashboard_field_id_test_steps
            dashboard_field_id_expected_result = YAML_FIELD_ID[milestone]['dashboard_field_id_expected_result'] \
                if YAML_FIELD_ID[milestone][
                'dashboard_field_id_expected_result'] else dashboard_field_id_expected_result
            dashboard_field_id_remark = YAML_FIELD_ID[milestone]['dashboard_field_id_remark'] \
                if YAML_FIELD_ID[milestone]['dashboard_field_id_remark'] else dashboard_field_id_remark
        else:
            continue


def parse_excel(file_path):
    # 处理Excel。最终将Excel的值装入extracted_data[]
    workbook = load_workbook(file_path)
    sheet = workbook.active
    # 获取标题行并确定索引
    # excel标题行必须在第1行
    excel_header = [cell.value for cell in sheet[1]]
    columns_to_extract = ["里程碑", "功能模块", "用例编号", "用例名称", "维护人", "用例类型", "优先级", "前置条件",
                          "备注", "步骤描述", "预期结果"]
    # 创建列标题到索引的映射
    col_index_map = {title: _idx + 1 for _idx, title in enumerate(excel_header)}
    # 获取实际列索引
    actual_columns = [col_index_map[col] for col in columns_to_extract]
    # 遍历数据行，将所有数据放入extracted_data
    _extracted_data = []
    for row in sheet.iter_rows(min_row=2, values_only=True):
        row_data = [row[i - 1] for i in actual_columns]
        _extracted_data.append(row_data)
    if _extracted_data:
        logging.info(f'读取Excel文件完毕')
        return _extracted_data
    else:
        logging.error(f'读取Excel文件失败。Excel文件内容可能为空')
        sys.exit(1)


def format_date(date_str):
    # 从 ISO 8601 时间格式中，提取日期和时间并return
    dt = datetime.fromisoformat(date_str)
    tmp_date = dt.date()  # 日期部分
    tmp_time = dt.strftime("%H:%M")  # 时间部分（小时:分钟）
    output = str(tmp_date) + ' ' + str(tmp_time)
    return output


def get_all_issues_info():
    _all_issues_info = []
    for _page in range(1, 21):
        _url_get_all_issues_info = \
            (f'https://web-api.gitcode.com/issuepr/api/v1/issue/openFuyao%2F{REPO_NAME}/issues?scope=all&state=opened'
             f'&page={_page}&project_id=openFuyao%2F{REPO_NAME}&per_page=50')
        _r_get_all_issues_info = requests.get(_url_get_all_issues_info, headers=header)
        _t_get_all_issues_info = json.loads(_r_get_all_issues_info.text)
        # _all_issues_info += _t_get_all_issues_info['issues']
        # _all_issues_info.append(_t_get_all_issues_info['issues'])
        if _t_get_all_issues_info['issues']:
            _all_issues_info += _t_get_all_issues_info['issues']
        else:
            pass
    return _all_issues_info


def parse_detailed_user_info(__username_chinese):
    global username, nickname, email, mobile, object_id, assignee_ids
    for user in YAML_USER_INFO['users']:
        if __username_chinese == user['username_chinese']:
            username = user['username_gitcode']
            nickname = user['nickname']
            email = user['email']
            mobile = user['mobile']
            object_id = user['object_id']
            assignee_ids = user['assignee_ids']
            break
        else:
            continue


def check_whether_issue_alredy_exist(_issue_title, _all_issues_info):
    _flag_issue_alredy_exist = 0  # 0表示未找到相同标题的issue  1表示找到相同标题的issue
    for _issue in _all_issues_info:
        if _issue_title == _issue['title']:
            _flag_issue_alredy_exist = 1
            break
        else:
            continue
    if _flag_issue_alredy_exist == 1:
        return 1
    else:
        return 0


def get_issue_iid_from_title_in_all_issues_info(_issue_title, _all_issues_info):
    for _issue in _all_issues_info:
        if _issue_title == _issue['title']:
            return _issue['iid']
        else:
            continue


def create_testcase_issue():
    data_create_issue = {
        "title": issue_title,
        "description": issue_description,
        "confidential": confidential,
        "assignee": {
            "access_token": AUTHORIZATION[7:],
            "refresh_token": "",
            "xauth_token": "",
            "id": 5273547,
            "object_id": object_id,
            "username": username,
            "nickname": nickname,
            "avatar": "",
            "email": email,
            "domain_id": "cec6c668d41444b4af3b8159a72a7a40",
            "arts_id": assignee_ids,
            "iam_id": "69f8ae9878fd48e2bce93301aeb1aa39",
            "login_id": "681db29be4b985165b497dc3",
            "status": 0,
            "audit_status": 10,
            "has_password": "true",
            "create_time": "2025-04-23T06:36:10.000+00:00",
            "user_status_enum": "SUCCESS",
            "is_new": "false",
            "mobile": mobile,
            "score": 0,
            "growth": 0,
            "level": 1,
            "next_level": 15,
            "need_tips": "true",
            "style_tips": "prompt_red_dot"},
        "assignee_ids": [assignee_ids],
        "project_id": f"openFuyao%2F{REPO_NAME}",
        "labels": [],
        "enterprise_labels": issue_lable,
        "type": issue_type,
        "priority": issue_priority,
        "milestone_id": milestone_id,
        "issue_category": "-",
        "issue_stage": "-",
        "issue_severity": "-"
    }
    url_create_issue = f'https://web-api.gitcode.com/issuepr/api/v1/issue/openFuyao%2F{REPO_NAME}/issues'
    _r_create_issue = requests.post(url_create_issue, headers=header, data=json.dumps(data_create_issue))
    return _r_create_issue


def parse_milestone_id(_data_excel):
    global milestone_id
    _url_get_milestone_id = (f'https://web-api.gitcode.com/issuepr/api/v1/projects/openFuyao%2F{REPO_NAME}/milestones?'
                             f'repoId=openFuyao%2F{REPO_NAME}&state=active&page=1&per_page=100')
    _r_get_milestone_id = requests.get(_url_get_milestone_id, headers=header)
    _t_get_milestone_id = json.loads(_r_get_milestone_id.text)
    for _milestone in _t_get_milestone_id['content']['milestones']:
        if _milestone['title'] == _data_excel[0]:
            milestone_id = _milestone['id']
            break
        else:
            continue


def parse_data_from_excel_to_create_issue(__data_excel):
    global issue_priority, issue_title, issue_description, issue_type, issue_lable, confidential
    # 首先明确创建issue时需要填写的各个字段
    # issue优先级。页面上的“Issue优先级”（严重，主要，次要，不重要）对应用例级别P0-P3
    match __data_excel[6]:
        case 'P0' | 'p0':
            issue_priority = '严重'
        case 'P1' | 'p1':
            issue_priority = '主要'
        case 'P2' | 'p2':
            issue_priority = '次要'
        case 'P3' | 'p3':
            issue_priority = '不重要'
    # issue标题
    issue_title = __data_excel[3]
    # issue正文
    issue_description = \
        (f"**用例名称**：{__data_excel[3]}\n**用例编号**：{__data_excel[2]}\n**功能模块**：{__data_excel[1]}\n**优先级**"
         f"：{__data_excel[6]}\n**维护人**：{__data_excel[4]}\n**里程碑**：{__data_excel[0]}\n**用例类型**："
         f"{__data_excel[5]}\n**备注**：{__data_excel[8]}\n**前置条件**：{__data_excel[7]}\n\n**步骤描述**：\n"
         f"{__data_excel[9]}\n\n**预期结果**：\n{__data_excel[10]}")
    # issue类型
    issue_type = '测试用例'
    # issue标签
    issue_lable = ['Test']
    match __data_excel[0]:
        case version if '25.12' in version:
            issue_lable.append('v25.12')
        case version if '25.09' in version:
            issue_lable.append('v25.09')
        case version if '25.06' in version:
            issue_lable.append('v25.06')
        case version if '25.03' in version:
            issue_lable.append('v25.03')
        case version if '24.12' in version:
            issue_lable.append('v24.12-beta1')
        case version if '24.09' in version:
            issue_lable.append('v24.09-beta1')
        case version if '24.06' in version:
            issue_lable.append('v24.06-beta1')
    # issue可见。false表示公开，true表示私密
    confidential = 'false'


def get_all_items_from_dashboard(_dashboard_id):
    _items_of_specified_dashboard = []
    # logging.info(f'开始获取{CURRENT_DASHBOARD}看板的所有数据item...')
    try:
        for i in range(1, 7):
            # 获取指定看板的所有items。每页200个，最多6页，即总数上限1200个
            _url_get_all_items_of_specified_dashboard = \
                f'https://web-api.gitcode.com/api/v1/kanban/{_dashboard_id}/kanban_item/list?page={i}&per_page=200'
            r_get_all_items_of_specified_dashboard = requests.get(
                _url_get_all_items_of_specified_dashboard, headers=header)
            t_get_all_items_of_specified_dashboard = json.loads(r_get_all_items_of_specified_dashboard.text)
            if t_get_all_items_of_specified_dashboard['content']:
                _items_of_specified_dashboard += t_get_all_items_of_specified_dashboard['content']
    except Exception as e:
        logging.error(f'尝试获取看板的所有数据item失败 {e}')
    logging.debug(f'_items_of_specified_dashboard: {_items_of_specified_dashboard}')
    _items_count = len(_items_of_specified_dashboard)
    logging.info(f'看板的所有数据item获取完毕【共计{_items_count}条】 (单个看板item上限为200条)')
    return _items_of_specified_dashboard


def get_item_id_from_title_in_all_items(_title, _all_items_of_dashboard):
    _item_id_to_cancel_link = []
    for _item in _all_items_of_dashboard:
        if _item['values'][0]['value']['title'] == _title:
            _item_id_to_cancel_link.append(_item['id'])
    return _item_id_to_cancel_link


def cancel_link_items_from_bashboard(_items_id_to_cancel_link):
    for _item_id in _items_id_to_cancel_link:
        _url_cancel_link_items_from_bashboard = f'https://web-api.gitcode.com/api/v1/kanban/{dashboard_id}/kanban_item'
        data_cancel_link_items_from_bashboard = {
            "kanban_item_ids": [_item_id]
        }
        requests.delete(_url_cancel_link_items_from_bashboard,
                        headers=header, data=json.dumps(data_cancel_link_items_from_bashboard))


def get_specified_issue_info(_issue_id):
    _url_get_specified_issue_info = \
        f'https://web-api.gitcode.com/issuepr/api/v1/issue/openFuyao%2F{REPO_NAME}/issues/{issue_id}'
    _r_get_specified_issue_info = requests.get(_url_get_specified_issue_info, headers=header)
    return _r_get_specified_issue_info


def link_issue_to_dashboard(_issue_id, _dashboard_id):
    _url_link_to_dashboard = \
        f'https://web-api.gitcode.com/api/v1/kanban/{_dashboard_id}/kanban_item/add_item_of_issues_pr'
    data_link_to_dashboard = {
        "project_id": f"openFuyao/{REPO_NAME}",
        "source_reqs": [{"source_type": 1, "source_id": str(_issue_id)}]
    }
    _r = requests.post(_url_link_to_dashboard, headers=header, data=json.dumps(data_link_to_dashboard))
    logging.debug(f'将用例issue关联到看板 {_r.text}')
    return _r


def dashboard_set_testcase_id(_item_id, _data_excel):
    _url_dashboard_edit = f'https://web-api.gitcode.com/api/v1/kanban/{dashboard_id}/kanban_item'
    data_dashboard_set_testcase_id = [{
        "kanban_item_id": _item_id,
        "kanban_field_id": dashboard_field_id_testcase_id,
        "kanban_field_value": {"value": _data_excel[2]}
    }]
    requests.put(_url_dashboard_edit, headers=header, data=json.dumps(data_dashboard_set_testcase_id))


def dashboard_set_testcase_priority(_item_id, _data_excel):
    _url_dashboard_edit = f'https://web-api.gitcode.com/api/v1/kanban/{dashboard_id}/kanban_item'
    _testcase_priority = ''
    match _data_excel[6]:
        case 'P0' | 'p0':
            _testcase_priority = testcase_priority_p0
        case 'P1' | 'p1':
            _testcase_priority = testcase_priority_p1
        case 'P2' | 'p2':
            _testcase_priority = testcase_priority_p2
        case 'P3' | 'p3':
            _testcase_priority = testcase_priority_p3
    data_dashboard_set_testcase_priority = [{
        "kanban_item_id": _item_id,
        "kanban_field_id": dashboard_field_id_testcase_priority,
        "kanban_field_value": {"id": _testcase_priority}
    }]
    requests.put(_url_dashboard_edit, headers=header, data=json.dumps(data_dashboard_set_testcase_priority))


def dashboard_set_module(_item_id, _data_excel):
    _url_dashboard_edit = f'https://web-api.gitcode.com/api/v1/kanban/{dashboard_id}/kanban_item'
    data_dashboard_set_module = [{
        "kanban_item_id": _item_id,
        "kanban_field_id": dashboard_field_id_module,
        "kanban_field_value": {"value": _data_excel[1]}
    }]
    requests.put(_url_dashboard_edit, headers=header, data=json.dumps(data_dashboard_set_module))


def dashboard_set_milestone(_item_id, _data_excel):
    _url_dashboard_edit = f'https://web-api.gitcode.com/api/v1/kanban/{dashboard_id}/kanban_item'
    data_dashboard_set_milestone = [{
        "kanban_item_id": _item_id,
        "kanban_field_id": dashboard_field_id_milestone_v,
        "kanban_field_value": {"value": _data_excel[5]}
    }]


def dashboard_set_testcase_type(_item_id, _data_excel):
    _url_dashboard_edit = f'https://web-api.gitcode.com/api/v1/kanban/{dashboard_id}/kanban_item'
    data_dashboard_set_testcase_type = [{
        "kanban_item_id": _item_id,
        "kanban_field_id": dashboard_field_id_testcase_type,
        "kanban_field_value": {"value": _data_excel[5]}
    }]
    requests.put(_url_dashboard_edit, headers=header, data=json.dumps(data_dashboard_set_testcase_type))


def dashboard_set_maintainer(_item_id, _data_excel):
    _url_dashboard_edit = f'https://web-api.gitcode.com/api/v1/kanban/{dashboard_id}/kanban_item'
    if _data_excel[4] == '李强':
        _testcase_maintainer = '李强1'
    else:
        _testcase_maintainer = _data_excel[4]
    data_dashboard_set_maintainer = [{
        "kanban_item_id": _item_id,
        "kanban_field_id": dashboard_field_id_testcase_maintainer,
        "kanban_field_value": {"value": _testcase_maintainer}
    }]
    requests.put(_url_dashboard_edit, headers=header, data=json.dumps(data_dashboard_set_maintainer))


def dashboard_set_testcase_review_state(_item_id, _data_excel):
    _url_dashboard_edit = f'https://web-api.gitcode.com/api/v1/kanban/{dashboard_id}/kanban_item'
    data_dashboard_set_testcase_review_state = [{
        "kanban_item_id": _item_id,
        "kanban_field_id": dashboard_field_id_testcase_review_state,
        "kanban_field_value": {"id": testcase_review_state_complete}
    }]
    requests.put(_url_dashboard_edit, headers=header, data=json.dumps(data_dashboard_set_testcase_review_state))


def dashboard_set_testcase_create_date(_item_id, _issue_create_date):
    _url_dashboard_edit = f'https://web-api.gitcode.com/api/v1/kanban/{dashboard_id}/kanban_item'
    data_dashboard_set_testcase_create_date = [{
        "kanban_item_id": _item_id,
        "kanban_field_id": dashboard_field_id_testcase_create_date,
        "kanban_field_value": {"value": _issue_create_date}
    }]
    requests.put(_url_dashboard_edit, headers=header, data=json.dumps(data_dashboard_set_testcase_create_date))


def dashboard_set_pre_request(_item_id, _data_excel):
    _url_dashboard_edit = f'https://web-api.gitcode.com/api/v1/kanban/{dashboard_id}/kanban_item'
    data_dashboard_set_pre_request = [{
        "kanban_item_id": _item_id,
        "kanban_field_id": dashboard_field_id_pre_request,
        "kanban_field_value": {"value": _data_excel[7]}
    }]
    requests.put(_url_dashboard_edit, headers=header, data=json.dumps(data_dashboard_set_pre_request))


def dashboard_set_test_steps(_item_id, _data_excel):
    _url_dashboard_edit = f'https://web-api.gitcode.com/api/v1/kanban/{dashboard_id}/kanban_item'
    data_dashboard_set_test_steps = [{
        "kanban_item_id": _item_id,
        "kanban_field_id": dashboard_field_id_test_steps,
        "kanban_field_value": {"value": _data_excel[9]}
    }]
    requests.put(_url_dashboard_edit, headers=header, data=json.dumps(data_dashboard_set_test_steps))


def dashboard_set_expected_result(_item_id, _data_excel):
    _url_dashboard_edit = f'https://web-api.gitcode.com/api/v1/kanban/{dashboard_id}/kanban_item'
    data_dashboard_set_expected_result = [{
        "kanban_item_id": _item_id,
        "kanban_field_id": dashboard_field_id_expected_result,
        "kanban_field_value": {"value": _data_excel[10]}
    }]
    requests.put(_url_dashboard_edit, headers=header, data=json.dumps(data_dashboard_set_expected_result))


def dashboard_set_remark(_item_id, _data_excel):
    _url_dashboard_edit = f'https://web-api.gitcode.com/api/v1/kanban/{dashboard_id}/kanban_item'
    data_dashboard_set_remark = [{
        "kanban_item_id": _item_id,
        "kanban_field_id": dashboard_field_id_remark,
        "kanban_field_value": {"value": _data_excel[8]}
    }]
    requests.put(_url_dashboard_edit, headers=header, data=json.dumps(data_dashboard_set_remark))


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
    logging.info(f'### Gitcode Batch Create TestCases and Link to Dashboard ###')
    time_start = time.time()
    gitcode_login(USER_ACCOUNT, USER_PASSWORD_WITH_HASH)
    check_auth()
    extracted_data = parse_excel(EXCEL_FILE_PATH)
    all_issues_info = get_all_issues_info()
    logging.debug(f'all_issues_info: {all_issues_info}')
    all_issues_count = len(all_issues_info)
    logging.info(f'【{REPO_NAME}】仓库所有[已开启]状态issue数量: {all_issues_count}')
    # logging.info(f'{REPO_NAME}仓库所有[已开启]状态issue数量:', len(all_issues_info))
    for idx, data_excel in enumerate(extracted_data, start=2):
        items_id_to_cancel_link = []
        logging.info(f'开始解析Excel第{idx}行  data_excel: {data_excel}')
        parse_target_dashboard_ids(data_excel[0])
        parse_data_from_excel_to_create_issue(data_excel)
        # 将Excel中的“维护人”关联到issue的“负责人”。 username取自assignee.name
        # 获取用户信息（后续创建issue时需要填写）
        parse_detailed_user_info(data_excel[4])
        # 获取里程碑版本对应的milestone_id（后续创建issue时需要填写）
        parse_milestone_id(data_excel)
        # 在已有issue中检查当前的用例标题是否已存在，如果已存在则跳过创建用例issue  0表示未找到相同标题的issue  1表示找到相同标题的issue
        if check_whether_issue_alredy_exist(data_excel[3], all_issues_info) == 1:
            # 获取已经存在的issue信息
            issue_id = get_issue_iid_from_title_in_all_issues_info(data_excel[3], all_issues_info)
            logging.warning(f'已经找到与用例标题相同的issue 编号:{issue_id}，不会重复创建issue。如果已有的issue内容'
                            f'非预期，建议手动修改issue内容，或手动关闭该issue，然后重新执行本工具')
        else:
            # 创建测试用例issue
            r_create_issue = create_testcase_issue()
            logging.info(f'创建用例issue {r_create_issue.text}')
            # 增加创建issue间隔，尝试消除平台的攻击告警，防止因告警而中断程序
            time.sleep(3)
            issue_id = json.loads(r_create_issue.text)["iid"]
        # 获取看板所有的item 找到当前excel对应的相同标题的item，获取其item_id， 解除与看板的绑定。
        all_items_of_dashboard = get_all_items_from_dashboard(dashboard_id)
        logging.debug(f'all_items_of_dashboard: {all_items_of_dashboard}')
        # 尝试获取与用例title相同的看板item
        items_id_to_cancel_link = get_item_id_from_title_in_all_items(data_excel[3], all_items_of_dashboard)
        # 接上一步，如果获取到，则解除与看板的绑定
        if items_id_to_cancel_link:
            cancel_link_items_from_bashboard(items_id_to_cancel_link)
            logging.info(f'检测到已存在的看板item，已取消issue关联看板(后续重新关联)  issue标题: "{data_excel[3]}"'
                         f'  item_id: {items_id_to_cancel_link}')

        # 将issue关联到指定看板
        r_link_to_dashboard = link_issue_to_dashboard(issue_id, dashboard_id)

        # 获取issue信息（当前主要是为了获取issue创建时间，后续修改看板）
        r_get_specified_issue_info = get_specified_issue_info(issue_id)
        specified_issue_info = json.loads(r_get_specified_issue_info.text)
        logging.debug(f'specified_issue_info: {specified_issue_info}')
        issue_create_date = specified_issue_info['created_at']  # 示例："2025-05-14T11:01:22.836+08:00"
        issue_create_date = format_date(issue_create_date)
        # 获取看板的当前行id，用于后续修改
        item_id = json.loads(r_link_to_dashboard.text)[0]["id"]
        logging.info(f'正在修改看板自定义字段...')
        # 修改看板 - 用例编号
        dashboard_set_testcase_id(item_id, data_excel)
        # 修改看板 - 用例优先级
        dashboard_set_testcase_priority(item_id, data_excel)
        # 修改看板 - 功能模块
        dashboard_set_module(item_id, data_excel)
        # 修改看板 - 里程碑
        # dashboard_set_milestone(item_id, data_excel)
        # 修改看板 - 用例类型
        dashboard_set_testcase_type(item_id, data_excel)
        # 修改看板-维护人
        dashboard_set_maintainer(item_id, data_excel)
        # 修改看板 - 用例评审状态
        dashboard_set_testcase_review_state(item_id, data_excel)
        # 修改看板 - 用例创建时间
        dashboard_set_testcase_create_date(item_id, issue_create_date)
        # 修改看板 - 前置条件
        dashboard_set_pre_request(item_id, data_excel)
        # 修改看板 - 步骤描述
        dashboard_set_test_steps(item_id, data_excel)
        # 修改看板 - 预期结果
        dashboard_set_expected_result(item_id, data_excel)
        # 修改看板 - 备注
        dashboard_set_remark(item_id, data_excel)
    # 再检查一次认证凭据有效性，防止程序运行过程中失效
    check_auth()
    # 时间模块，统计程序执行耗时
    print_time_spent(time_start)
    clean_logs_by_time(LOG_DIR)

