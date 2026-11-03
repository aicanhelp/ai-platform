# 如何贡献

## 提交issue

为项目做出贡献的一个好方法是在遇到问题时发送详细报告。我们始终欣赏写得好、详尽的错误报告和功能建议，并会为此而感激您！

### issue格式

提交issue时，请参考此格式：

```markdown
## 子系统
## 模块
## 特性
## 标签
## 标题
## 问题描述
> 清楚、简洁地描述问题所在。
## 预期行为
> 清楚、简洁地描述你期望发生的情况。
## 实际行为
> 清晰、简洁地描述实际发生的情况。
## 复现步骤
> 简洁的步骤，复现问题所需：
> 1. ...
> 2. ...
> 3. ...
## 相关日志
> 如果有日志输出，请提供相关日志信息。
## 环境信息
> - 操作系统 (例如，Linux, macOS, Windows):XXX
> - 是否为虚拟机:是/否
> - 是否有特殊的网络配置:XXX
> - Kubernetes 版本 (use `kubectl version`):XXX
> - 集群配置 (使用 `kubectl cluster-info dump`):XXX
> - 其他相关环境信息:XXX
## 严重程度
## 是否可重现
## 结果影响
## 期望解决日期
## 其它补充信息
```

## 提交PR请求

如果你是初学者，希望通过这个项目打开开源世界的大门，那么本教程是你的最佳选择之一。只要按照指导，你就会发现成为贡献者的乐趣。

### 步骤 1：fork代码仓库

在修改本项目之前，您需要确保本项目已经fork到您自己的仓库中。这意味着此仓库和您自己的仓库之间会并行开发，因此请注意避免这两个仓库之间出现不一致的情况。

### 第 2 步：克隆远程代码仓库

如果要将代码下载到本地机器，git最好的方法是：

```shell
git clone https://your_repo_url/repo.git
```

### 步骤 3：本地开发代码

为了避免多个分支之间不一致，我们建议切换到新分支：

```shell
git checkout -b new_branch_name origin/master
```

然后你就可以更改代码了。

### 步骤4：将代码推送到远程仓库

更新代码后，应该通过正式的方式推送更新：

```shell
git add .
git status (Check the update status)
git commit -m "feat: Your commit description"
git commit --amend (Add the concrete description of your commit)
git push origin new_branch_name
```

### 步骤 5：提交PR请求

最后一步，你需要拉取新分支和开发分支之间的比较请求。完成PR提交后，CI将自动设置以进行构建测试。

### pull request模板

#### 标题模板

标题模板 :type(product):XXX

样例:feat(installer):add offline install

type:

- fix：问题修复
- feat：新增功能
- docs：文档
- refactor：重构
- test：测试用例
- changlog：修复日志
- style：codecheck问题
- build：构建相关
- chore：其它问题

#### 正文模板

```text
Ref. #3064 #3097
All files owned by SIG testing were moved from `/devel` to the new folder `/devel/sig-testing`.

/sig contributor-experience
/cc @stakeholder1 @stakeholder2
/kind cleanup
/area developer-guide
/assign @approver1 @approver2 @approver3
```

这个 PR 里包含了：

- 第 1 行 - 提及其它有关的 issue 或 PR（#3064 #3097）。
- 第 2 行 - 对该 PR 做了什么事情的一个简短的描述。
- 第 4 行 - 使用 /sig contributor-experience 命令分配 SIG。
- 第 5 行 - 使用 /cc 命令指定可能对该 issue 或 PR 感兴趣的评审者。
- 第 6 行 - /kind cleanup 命令将该 issue 或 PR 添加了一个标签将它归类为清理代码、流程或者技术债务。
- 第 7 行 - /area developer-guide 命令将该 issue 或 PR 归类为与开发者指南有关。
- 第 8 行 - /assign 命令为该 PR 分配了一名批准者。k8s-ci-robot 会从 OWNERS 文件中的所有者列表中选择，推荐一名批准者。在评审结束后，他（她）将会使用 /approve 为该 PR 添加标签。
