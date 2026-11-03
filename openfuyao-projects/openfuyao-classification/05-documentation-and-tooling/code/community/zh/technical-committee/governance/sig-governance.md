# SIG角色说明

**社区成员**

本文简要描述了openFuyao社区中贡献者角色的各种职责。大部分角色的职责限于这些SIG (Special Interest Group)内：

| 角色       | 职责范围（简要描述） | 要求                                                  | 定义的文件                                                   |
| :--------- | -------------------- | ----------------------------------------------------- | ------------------------------------------------------------ |
| Member     | SIG成员              | SIG的积极贡献者                                       | openFuyao community仓库的sig-info.yaml 文件中的Member条目。  |
| Committer  | 审核其他成员的贡献   | SIG的积极贡献者，经验丰富，愿意投入精力参与到审核工作 | openFuyao community仓库的sig-info.yaml 文件中的Commiter条目。 |
| Maintainer | SIG Owner            | 经验丰富，富有责任心、出色的技术能力和管理能力        | openFuyao community仓库的sig-info.yaml 文件中的Maintainer条目。 |

说明：Maintainer和Committer在GitCode的权限上不做区分，两者的区分主要是集中在SIG治理的管理范围上。详细可以见下面的描述。

## 新的贡献者

欢迎新成员加入社区。我们有关于如何开始贡献的指导文档请参考： [openFuyao 贡献者指南](https://gitcode.com/openFuyao/community/blob/master/zh/contributors/ContributionDetail.md)

## 既有社区成员

既有的社区成员应证明能够遵守本文中的原则，熟悉SIG的组织、角色、政策、软件、约定等，以及相关的技术和/或写作能力。社区成员角色的期望、职责和要求，请参考下面的内容。

## SIG成员 Member

SIG成员是社区中持续活跃的贡献者，他们可以认领问题和PR，可以参与SIG组活动，并且可以为PR提交前完成测试。

**要求**

- GitCode上的注册会员

- 为SIG或社区做出多方面贡献，包括不限于：

  ```text
  在GitCode上提交或审核PR
   
  在GitCode上对问题进行归档或评论
   
  参与SIG或社区讨论          
  ```

- 已阅读 [贡献者指南](https://gitcode.com/openFuyao/community/blob/master/zh/contributors/ContributionDetail.md)

- 积极参与1个或多个SIG

- 提交3个有价值的PR

**责任与权利**

- 响应被分配的问题和PR

- 贡献的代码应该

  ```text
  经过良好的测试
   
  能够让测试用例始终通过
   
  解决后继发生的错误或问题     
  ```

- 可以分配问题或PR

注意：经常贡献代码的成员应积极的参与代码审查，并努力成为SIG的审核者Committer。

## 审核者 Committer

审核者能够在SIG或SIG的某些部分中审核代码的质量和正确性。审核者对代码库和软件工程原理非常了解。

**要求**

- 作为贡献者至少3个月
- 作为主要审阅者至少参与了6 次PR的审阅
- 审阅或合并至少20个基本PR到代码库
- 熟悉代码库
- 可以自我提名，或由该SIG的审核者或维护者提名

**责任与权力**

- 评审 PR：对Contributor提交的PR完成评审，评审可以参考社区的[编程规范](https://gitcode.com/openFuyao/security-committee/blob/master/zh/technical-committee/governance/README.md)。

- 和上游社区合作，包括：

  将所有变更推送到上游社区

  参与上游社区邮件列表

  获取上游社区的bug跟踪器的账户，并跟踪上游社区的重要bug

- 和测试团队合作，包括：

  在提交制品时，向质量检查人员提供如何调试/分类制品的信息，以供问题的分类

  提供基本功能的测试用例，用于测试回归

  提交制品更新时，提供有关更新中已经修复问题的测试用例，以供质量检查人员使用

## 维护者 Maintainer

维护者是SIG组的组长，可成为管理委员会或技术委员会成员，也是软件包的维护者，能够像Committer一样审查和批准代码贡献。代码审查的重点是代码质量和正确性，而批准的重点是对贡献的整体接受度。所有Committer 的责任与权力，Maintainer均具有。除此之外，Maintainer还承担了团队的技术路线、内外协调等工作。

**要求**

- 成为Committer至少3个月
- 作为主要Committer至少参与了12次PR的审核
- 审核或合并至少30个基本PR到代码库
- 熟悉代码库
- 可以自我提名，也可以由子项目Maintainer提名，并且没有其他子项目Maintainer的反对

**责任与权力**

- 确定SIG所负责项目的技术路线：包括规划和决策SIG技术方向、路标规划、架构演进。
- 制定SIG所负责项目的发布计划：确定SIG的关键需求和发布计划；参与社区的PM活动，并协调SIG计划和社区版本的里程碑时间表匹配。
- 参与社区协调活动：作为SIG的代表参与openFuyao技术委员会或理事会组织的活动和特定会议等。
- 召集SIG组会议：定期召集SIG会议，决策SIG内上升的争议。
