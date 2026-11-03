# PR提交模板
## pull request模板

### 标题模板
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

### [正文模板](https://github.com/kubernetes/community/blob/master/contributors/guide/contributor-cheatsheet/README-zh.md#%E6%8F%90%E4%BA%A4-pull-request)
```
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

## ref
- https://github.com/kubernetes/community/blob/master/contributors/guide/pull-requests.md
- https://github.com/kubernetes/community/blob/master/contributors/guide/contributor-cheatsheet/README-zh.md#%E6%8F%90%E4%BA%A4-pull-request