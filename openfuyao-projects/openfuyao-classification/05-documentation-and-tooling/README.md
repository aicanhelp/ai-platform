# 05 文档、社区、治理与工程工具

本目录收录 openFuyao 项目的文档、社区治理、SIG（Special Interest Group）提案、测试工程及开发者工具代码。它们不直接构成运行时的平台能力，但对理解项目组织、贡献流程、测试体系和发布策略很关键。

## 包含组件

| 目录 | 说明 |
|---|---|
| `community` | 社区治理文件、贡献指南、会议记录等。 |
| `docs` | 项目级文档。 |
| `ofep` | openFuyao Enhancement Proposals，增强提案。 |
| `security-committee` | 安全委员会文档与流程。 |
| `tc` | Technical Charter 或技术指导委员会文档。 |
| `sig-*` | 各特殊兴趣小组的工作目录：ai-inference、colocation、container-platform、distributed-framework、installation、large-scale-cluster、numa-affinity、orchestration-engine、qa、ub-enable。 |
| `e2e-auto-test` | 自动化端到端测试。 |
| `qa-tool` | 质量/测试辅助工具。 |
| `developer-experience` | 开发者体验工具、示例、脚本。 |
| `dependency-database` | 依赖版本/扫描/数据库工具。 |
| `release-management` | 版本发布规划、规则、报告等。 |

## 边界说明

- 这些目录不部署为生产服务，也不绑定昇腾或任何单一平台能力。
- `sig-*` 目录中可能包含少量原型代码，但本质仍属于社区治理/提案材料。
- 当后续需要把某个 SIG 中的原型升级为正式组件时，再考虑归入 01/02/03/04 中的某一类。
