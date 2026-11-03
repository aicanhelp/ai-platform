# qa-tool

## 仓库介绍

该仓库用于存放sig-qa相关测试人员可能用到的自研工具/脚本。目前包含2个工具：

- 批量创建测试用例并关联到看板 `gitcode_batch_create_testcases_and_link_to_dashboard`

- 批量关联问题单到看板 `gitcode_batch_link_issues_to_dashboard`

仓库所属sig组： [sig-qa](https://gitcode.com/openFuyao/sig-qa)

## 工具介绍

### gitcode_batch_create_testcases_and_link_to_dashboard

- 工具介绍：在GitCode平台批量创建手工测试用例issue，并关联到对应版本的测试用例看板

```
注1：目前手工测试用例本质上是test-case仓库的issue（issue类别为“测试用例”）。工具会根据测试用例的“里程碑”字段，创建对应里程碑的测试用例issue，再关联到对应里程碑的测试用例看板。
注2：关于重复创建或关联的判断逻辑：工具的基础判断依据是“用例名称”字段，即创建测试用例issue时，工具会检测相同名称的用例issue是否存在，如果已存在则不会重复创建issue，仅会将已存在的issue内容刷新到最新。 用例关联看板的逻辑相同，即相同名称的测试用例不会在看板中关联两次。
```

- 工具链接: [gitcode_batch_create_testcases_and_link_to_dashboard](https://gitcode.com/openFuyao/qa-tool/tree/main/gitcode_batch_create_testcases_and_link_to_dashboard)

### gitcode_batch_link_issues_to_dashboard

- 工具介绍：在GitCode平台批量将指定里程碑的问题单issue关联到对应版本的问题单看板；并检查看板中的自定义字段值，如果字段值有变动则刷新。并支持日志记录和自动清理，方便周期自动执行。

```
注1：问题单issue关联看板的基础判断依据是issue的Label中的版本号，是否与里程碑相同。
注2：工具仅会关联“缺陷”类型的issue，不会关联其他类型的issue。
```

- 工具链接: [gitcode_batch_link_issues_to_dashboard](https://gitcode.com/openFuyao/qa-tool/tree/main/gitcode_batch_link_issues_to_dashboard)

## 项目层级结构

```
├───.gitcode
│   └───ISSUE_TEMPLATE
├───.idea
│   └───inspectionProfiles
├───gitcode_batch_create_testcases_and_link_to_dashboard    # 批量创建测试用例并关联到看板
│   └───Template_excel_of_test_cases.xlsx    # 测试用例Excel文件模板
│   └───dashboard_field_ids.yaml    # 配置看板字段的id
│   └───main.py    # 工具脚本main文件，具体用法参照"使用指南"章节
│   └───userinfo.yaml    # 存放用户信息，用于创建issue
└───gitcode_batch_link_issues_to_dashboard    # 批量关联问题单到看板
│   └───dashboard_field_ids.yaml    # 配置看板字段的id
│   └───main.py    # 工具脚本main文件，具体用法参照"使用方法"章节
```

## 使用方法

### gitcode_batch_create_testcases_and_link_to_dashboard

##### 步骤1：用例编写

根据测试用例模板Excel文件（Template_excel_of_test_cases.xlsx）编写测试用例

##### 步骤2：工具配置

```
# mian.py
"EXCEL_FILE_PATH": 测试用例模板Excel文件的实际存放路径
"REPO_NAME": 创建测试用例issue的仓库。保持默认"test-case"即可
"USER_ACCOUNT": GitCode用户账号，用于获取认证凭据
"USER_PASSWORD_WITH_HASH": GitCode用户密码(哈希)，用于获取认证凭据。获取方法：开启浏览器F12并手动登录GitCode，找到POST方法的login请求，拷贝请求body体中的password值。
"LOG_FILE_KEEP_DAYS": 日志文件保存的天数。超过该天数会被清理。
"LOG_DIR": 日志文件的存储路径。（需提前创建）
"MILESTONE_LIST": 该元组中需要包含当前里程碑版本号

# dashboard_field_ids.yaml
如果进入新的里程碑，需要在该文件创建新的版本号，并通过浏览器F12从看板中将id配置到该文件。

# userinfo.yaml
确保测试用例的“维护人”信息存在于该配置文件中。
```

##### 步骤3：工具运行

执行 `python3 main.py` 即可读取 `EXCEL_FILE_PATH` 中的测试用例，并在 `REPO_NAME` 仓库创建测试用例issue，并将测试用例issue关联到对应里程碑的测试用例看板。

### gitcode_batch_link_issues_to_dashboard

##### 步骤1：工具配置

```
# mian.py
"CURRENT_DASHBOARD": 指定里程碑版本。
"SCAN_REPOS_LIST": True表示扫描全量仓库列表，并检测尚未关联到看板的issue，同时耗时也较久（约25-30分钟）。False表示跳过扫描仓库，仅检测看板已有条目的自定义字段信息是否正确
"USER_ACCOUNT": GitCode用户账号，用于获取认证凭据
"USER_PASSWORD_WITH_HASH": GitCode用户密码(哈希)，用于获取认证凭据。获取方法：开启浏览器F12并手动登录GitCode，找到POST方法的login请求，拷贝请求body体中的password值。
"LOG_FILE_KEEP_DAYS": 日志文件保存的天数。超过该天数会被清理。
"LOG_DIR": 日志文件的存储路径。（需提前创建）
"MILESTONE_LIST": 该元组中需要包含当前里程碑版本号

# dashboard_field_ids.yaml
如果进入新的里程碑，需要在该文件创建新的版本号，并通过浏览器F12从看板中将id配置到该文件。
```

##### 步骤2：工具运行

执行 `python3 main.py` 即可自动扫描全量仓库列表（需要当前GitCode用户拥有对应仓库的查看权限），解析每个仓库中符合指定里程碑、且尚未关联的问题单issue，并将问题单issue关联到对应里程碑的问题单看板。关联完毕后，工具会扫描问题单看板中的自定义字段值，如果字段值有变动则刷新。

注：推荐通过Windows计划任务等方式周期循环运行本工具。