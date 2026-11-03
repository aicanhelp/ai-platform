# SIG相关配置的格式化存储

## 背景

本目录下存放的是openFuyao社区中，所有代码仓与特别兴趣小组（Special Interest Group，以下简称SIG）的配置信息。

本目录下每一个子目录代表一个SIG。每个SIG目录中存在一个`sig-info.yaml`文件，用来配置SIG的成员信息以及SIG管理的仓库。

## 配置管理方式

1. 每个SIG在`openfuyao/community`仓库的`sig`目录中都对应一个独立目录，目录名为`sig-<SIG名称>`
2. 原则上由技术委员会（Technical Committee，以下简称TC）修改和维护。各个SIG对应的`sig-info.yaml`的修改，由该SIG的Maintainer提交PR，经过TC审视后合入。
3. 各SIG目录下的`README.md`为SIG的信息展示区。其中SIG基本信息需按模板留空，由工具自动填充。

## 格式规范

### `sig-info.yaml`文件格式

`sig-info.yaml`文件使用`YAML`格式承载，包含如下属性：

| 字段 | 类型 | 说明 |
|--|--|--|
| `name` | string | SIG组名称，与目录名相同 |
| `description` | string | SIG组描述信息 |
| `mailing_list` | string | SIG组讨论邮件列表地址 |
| `maintainers` | sequence | SIG组所有Maintainer的信息 |
| `repositories` | sequence | SIG组管理的仓库信息 |

其中，`repositories`的每一个元素代表一个仓库组（逻辑概念），每个仓库组中的仓库具有相同的Committer、Member和Admin：

| 字段 | 类型 |  说明 |
|--|--|--|
| `repos` | sequence | 一组SIG仓库的信息 |
| `committers` | sequence | 仓库组所有Committer的信息 |
| `members` | sequence | 仓库组所有Member的信息 |
| `repo_admins` | sequence | 仓库组所有Admin的信息 |


根节点下`maintainers`的每一个元素代表SIG中的一位Maintainer，具有SIG下所有仓库的PR合入权限；`repositories`每个元素下的`committers`、`members`、`repo_admins`的每一个元素分别代表仓库组中的一位Committer、Member、Admin；Committer具有该仓库组下所有仓库的PR合入权限；Member具有该仓库组下所有仓库相关信息的查看权限；Admin具有该仓库组下所有仓库的管理权限，由于权限过大，需要对该角色严格管控。

| 字段 | 类型 |  说明 |
|--|--|--|
| `gitcode_id` | string | 用户gitcode ID |
| `name` | string | 用户姓名 |
| `email` | string | 用户email |

### sig-info.yaml 样例

```yaml
name: sig-Installation # SIG名称，与目录名保持一致
description: 负责openFuyao解决方案版本各类场景下安装部署能力的构建，支撑客户界面简单易用的、一键执行的全部署能力的提供
maintainers: # SIG maintainer
  - gitcode_id: tosuper # gitcode用户ID，个人主页URL末尾路径段（最后一个`/`后的部分）
    name: SHI Zilong
    email: shizilong1@huawei.com # 用户提交邮箱，与签署CLA的邮箱保持一致
repositories: # SIG仓库组（逻辑概念）；每个仓库组中的仓库具有相同的committer、member和repo_admin
  - repos:
      - openfuyao/cluster-api-provider-bke # gitcode仓库路径
      - openfuyao/docs
    committers: # 仓库committer
      - gitcode_id: zhangxyjlu
        name: ZHANG Xiaoyang
        email: zhangxiaoyang2@huawei.com
      - name: javadoors
        name: ZHANG Chao
        email: paul.zhang@huawei.com
    members: # 仓库member
      - gitcode_id: youou
        name: CUI Leilei
        email: cuileilei163@163.com
    repo_admins: # 仓库admin，非必要不添加
      - gitcode_id: zhangfan345
        display_name: ZHANG Fan
        email: zhangfan345@huawei.com
        
  - repos:
      - openfuyao/installer
    committers:
      - gitcode_id: zhangxyjlu
        name: ZHANG Xiaoyang
        email: zhangxiaoyang2@huawei.com
      - gitcode_id: javadoors
        name: ZHANG Chao
        email: paul.zhang@huawei.com
```


### 代码仓描述文件格式

配置文件存放路径为`sig/sig-<SIG组名称>/openfuyao/<仓库首字母>/<仓库名称>.yaml`，整体以yaml格式承载，包含如下基本元素：

| 名称 | 类型 | 说明 |
| :-- | :-- | :-- |
| name|字符串|仓库名称|
| description| 字符串 | 仓库包含组件的描述 |
| type|枚举类型，可选 public 或者 private | 仓库类型。private代码仓不提供开放访问|
| branches|清单|本代码仓下所有分支信息|

branches 清单中每个元素代表一个受管理的分支，以关系数组的方式呈现，需要包含以下子元素:
| 名称 | 类型 | 说明 |
| :-- | :-- | :-- |
| name| 字符串 | 分支名称 |
| type | 枚举类型，可选protected/readonly | 分支类型，protected 表示该分支可以被发布版本集成，readonly 表示该分支停止维护。 |
| create_from | 字符串 | 分支创建起点，默认分支可以为空；其他分支<font color=red>必须</font>指定已存在的分支名或tag名称 |

### 代码仓描述文件样例
```yaml
name: A-Tune
description: 'This is a repo for ……'
type: public
branches:
  - name: master
    type: protected
  - name: branch-v1
    type: protected
    create_from: master
  - name: branch-v2
    type: protected
    create_from: branch-v1
  - name: branch-v3
    type: protected
    create_from: tag-1

```