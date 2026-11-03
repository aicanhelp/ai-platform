# NPU Operator 文档索引与维护规范

> ⚠️ **草案，待评审**：本规范为提案，经社区评审达成共识后生效；生效时移除此提示。

## 1. 文档索引

| 内容 | 位置 | 受众 |
|---|---|---|
| 架构总览 | [design/architecture.md](design/architecture.md) | 开发者 |
| CRD 字段参考 | [design/crd-reference.md](design/crd-reference.md) | 开发者 |
| Reconcile 逻辑 | [design/reconcile.md](design/reconcile.md) | 开发者 |
| 安装与使用 | [官网 User Guide](https://gitcode.com/openFuyao/docs/blob/master/docs/zh/user_guide/npu_operator.md#npu-operator)（独立仓库） | 用户 |
| 版本变更记录 | [CHANGELOG.md](../CHANGELOG.md) | 全部 |
| 贡献指引 | [CONTRIBUTING.md](../CONTRIBUTING.md) | 开发者 |

新开发者建议阅读顺序：README-zh.md → architecture.md → 按需查阅其余两篇。

设计文档统一使用中文撰写，采用 Markdown 与 Mermaid；每篇开头以一句话概述定位并链接相关文档。

## 2. 维护规范（草案）

**代码变更请同步更新对应文档。** 文档与代码保持一致是本项目的共同约定，建议在同一个 PR 中完成以下联动更新：

| 代码变更 | 对应文档 |
|---|---|
| CRD 字段（`api/v1/`） | crd-reference.md |
| Reconcile 阶段 / 渲染管线 / 冲突检测 | reconcile.md |
| 组件清单、拓扑、部署形态 | architecture.md |
| 尚未发布的的行为/特性 | CHANGELOG.md 的 Unreleased 段 |

其他建议：

- **新增文档**：置于 `docs/design/`，使用小写英文与连字符命名，并在本文档索引表中登记；优先扩充现有文档，避免新建重复内容。
- **评审**：设计文档建议由至少 1 名维护者评审；评审以事实与代码的一致性为主，文风不作要求。若文档与代码不一致，评审中指出对应位置即可，作者可随后补齐。
- **准确性**：字段、路径、组件名以代码为准，正文中注明事实来源；避免大段复制代码，以链接代替。

## 3. 发布流程

发布前请完成以下文档动作：

1. 将 CHANGELOG.md 的 Unreleased 段定稿为对应版本号（格式参考 [standard/CHANGELOG](https://github.com/standard/standard/blob/master/CHANGELOG.md)）；
2. 填写 release comment（变更摘要并指向 CHANGELOG 对应段落）；
3. 用户可见行为发生变化时，增量更新官网 User Guide；
4. 抽查设计文档与代码的一致性，偏差较大者安排专项修正。

## 4. 待评审事项

- [ ] 文档未联动更新时，如何在评审中提醒与跟进（如 reviewer 评论、后续补齐 PR）？
- [ ] 发布前文档抽查由谁负责？
- [ ] `design/` 是否按模块继续拆分（如 `device-plugin.md`），或维持现有三篇？

## TODO
- [ ] 校对 Reconcile 逻辑
- [ ] 校对 CRD 字段处理流程
