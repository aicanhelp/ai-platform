# 贡献指南

欢迎来到openFuyao！<br />
在参与社区贡献之前，请先阅读并遵守[openFuyao社区行为守则](https://openfuyao.cn/zh/communityRules/)。

## 1. 注册GitCode

openFuyao源代码托管在GitCode上：<https://gitcode.com/openfuyao>

请参考<https://docs.gitcode.com/docs/start/quick>注册您的GitCode账户，并在<https://gitcode.com/setting/email>设置您的主邮箱。

## 2. 签署CLA

在参与社区贡献前，您还需要签署[openFuyao社区贡献者许可协议（CLA）](https://cla.openfuyao.cn/sign/67e7c745ea9ceaa8feb65567)。

## 3. 选择SIG

### 3.1 了解SIG并找到您感兴趣的SIG

- 社区SIG组（Special Interest Group）是针对一个或多个主题，推动交付成果输出的团队。
- 社区SIG是开放的，欢迎任何人加入并参与贡献。您可以查看[SIG列表](https://openfuyao.cn/zh/sig/)找到感兴趣的SIG。每一个SIG在GitCode上拥有一个或多个代码仓库。您可以在SIG对应的代码仓库上提交Issue，参与Issue讨论，提交Pull Request，参与代码检视等。
- 项目（Project）：SIG会针对一个或多个主题树立目标，从而成立项目去实现目标，所以每一个SIG都至少有一个项目。
- 仓库（Repository，也即repo）：每一个项目都有一个或多个仓库（repository）用于保存任务的过程和结果。所以可以把项目等同于Repository。
- 团队成员：Maintainer和Committer是SIG的团队成员，他们定义在SIG的OWNERS文件里。团队成员可以被指派为项目的负责人，定义在项目的OWNERS文件里。

### 3.2 成立新的SIG

如果在已有的SIG列表里面没有找到您感兴趣的SIG，而您对某个方向有浓厚的兴趣，希望在社区成立一个新的相关SIG进行维护和发展，您可以参考[SIG申请流程](https://openfuyao.cn/zh/sig/sig-guidance/)创建新的工作组。

## 4. 开始贡献

### 4.1 提交Issue

#### 4.1.1 找到愿意处理的Issue

在您感兴趣的工作组项目GitCode主页内，单击“Issues”，您可以找到其[Issue列表](https://gitcode.com/org/openFuyao/issues?state=opened&scope=all&page=1)。如果您愿意处理其中的一个Issue，可以在Issue评论区积极评论，管理员将为您分配Issue。请您及时处理分配的Issue，若处理不及时，管理员可能会将Issue重新分配给其他人。或者您也可以在对应的团队或项目的repository内，进入Issue面板“新建Issue”。

新建Issue：

​    1. 进入Issue面板，单击“新建Issue”；

​    2. 单选下拉框将Issue类型设置成“需求”，系统会自动为您调出需求模板；

​    3. 标题栏简要描述需求的要点；

​    4. 在详细说明框内说明需求的场景和价值。

#### 4.1.2 参与Issue内的讨论

每个Issue下面可能已经有参与者们的交流和讨论，如果您感兴趣，也可以在评论框中发表自己的意见参与Issue讨论。

#### 4.1.3 提交Issue

如果您准备向社区上报Bug或者提交需求，或者为openFuyao社区贡献自己的意见或建议，请在社区对应的仓库上提交Issue（请参考[Issue](https://docs.gitcode.com/docs/help/home/org_project/issue/)）。为了吸引更广泛的注意，您也可以把Issue的链接附在邮件内，通过[邮件列表](https://mailweb.openfuyao.cn/)发送给所有人。

### 4.2 提交Pull-Request

#### 4.2.1 了解工作组和项目内的开发注意事项

每个工作组内的项目使用的编码语言、开发环境、编码约定等都可能存在差异的。如果您想了解并参与到编码类贡献，可以参考[编码规范](https://gitcode.com/openFuyao/security-committee/blob/main/docs/zh/developer-guide/SecurityCoding.md)和贡献者指南。这个指南一般是在该工作组的首页地址内，以CONTRIBUTING.md文件的形式提供，或者就直接在该项目的README.md内。

#### 4.2.2 下载代码和拉分支

如果要参与代码贡献，您还需要了解如何在GitCode下载代码，通过PR合入代码等，具体使用请参见[《Fork工作流》](https://docs.gitcode.com/docs/help/home/org_project/pullrequests/pr-fork)。该托管平台的使用方法类似GitHub，如果您以前使用过GitHub，本节的内容您可以大致了解。

#### 4.2.3 修改、构建和本地验证

在本地分支上完成修改后，进行构建和本地验证。

#### 4.2.4 提交一个Pull-Request

当您提交一个PR的时候，就意味您已经开始给社区贡献代码了（请参考[Pull Requests](https://docs.gitcode.com/docs/help/home/org_project/pullrequests/)）。为了使您的提交更容易被接受，您需要：

- 遵循工作组的编码约定；
- 准备完善的提交信息；
- 如果一次提交的代码量较大，建议将大型的内容分解成一系列逻辑上较小的内容，分别进行提交会更便于检视者理解您的想法；
- 使用适当的工作组和监视者标签去标记PR：社区机器人会发送给您消息，以方便您更好的完成整个PR的过程。

### 4.3 贡献原创开源项目

如果您想将自己原创的开源项目贡献到openFuyao社区，您首先需要遵循：

- 项目代码必须开源开放，遵循相关的[开源协议](https://opensource.org/license)；
- 通过邮件列表预约[TC（Technical Committee，即openFuyao技术委员会）例会](https://etherpad.openfuyao.cn/p/sig-Tc)，在会上申请将您的开源项目融入到社区中。

#### 4.3.1 经TC例会之后，您可以参考以下两种方式将您的作品贡献到openFuyao中

##### 4.3.1.1 在其他社区开发，集成到本社区中

假定您已经在github，gitlab或者gitcode上拥有了自己的项目，那么通过TC会议评审后，项目即可加入到本社区的repo仓。

##### 4.3.1.2 在本社区中开发，在本社区中集成

如果您有了一个很棒的idea想要在openFuyao社区里实现，那么您可以依照下面的过程来深度参与到社区中：

1. 在TC的例会中申请一个开源项目；
2. 如果TC委员认为这是一个很好的idea，并且认为值得去推进，那么我们会在GitCode中建立一个repo；
3. 这个项目在本社区中持续开发和孵化，直到项目逐渐成熟；
4. 经TC评审通过，最终这个原创的开源项目会随着本开源社区的发布版本走遍全世界，为世界人民所使用。

### 4.4 检视代码

openFuyao开源社区是一个开放的社区，我们希望所有参与社区的人都能成为活跃的检视者。当成为工作组的committer或maintainer角色时，便拥有审核代码的责任与权利。

[《补丁审核的柔和艺术》](https://sage.thesharps.us/2014/09/01/the-gentle-art-of-patch-review/)一文中提出了一系列检视的重点，说明代码检视的活动也希望能够促进新的贡献者积极参与，而不会使贡献者一开始就被细微的错误淹没，所以检视的时候，可以重点关注包括：

1. 贡献背后的想法是否合理；
2. 贡献的架构是否正确；
3. 贡献是否完善。<br/>

## 5. 和社区一起成长

社区不同角色对应不同的责任与权利，每种角色都是社区不可或缺的一部分，您可以通过积极贡献不断积累经验和影响力，并获得角色上的成长。请查看[社区角色说明](https://www.openfuyao.cn/zh/sig/role-description/)了解更详细介绍与责任权利描述。
