# SIG 管理指南

 目录

- [申请新SIG流程](#id1)
- [SIG变更申请流程](#id2)
  - [增删新项目或repository申请流程](#id2-1)
  - [变更团队成员申请流程](#id2-2)

<h2 id="id1">申请新SIG流程</h2>

**1、使用SIG模板创建自己的新SIG**

将 gitcode.com/openFuyao/community Fork到你的GitCode下,并在sig目录下创建你的sig文件夹。


```
git clone https://gitcode.com/YOURGITCODE/community

cd ./community/sig

mkdir sig-YOURSIGNAME

cd sig-YOURSIGNAME
```

**2、完成SIG相关信息的配置**

请在sig-info.yaml文件中完成对SIG成员及仓库的配置，可参考已有[样例](./../../../sig/sig-Large-Scale-Cluster/sig-info.yaml)。

```
vi sig-info.yaml

- name: sig-YOURSIGNAME
  repositories:
  - openFuyao/aaa
```

**3、提交PR**

将以上修改提交到GitCode上，并在GitCode上创建一个Pull Request。

**4、编写SIG创建提案**

将 gitcode.com/openFuyao/ofep Fork到你的GitCode下,并在ofeps目录下以创建新sig文件夹，在该文件夹下编写SIG创建提案。SIG创建提案可参考[示例文件](https://gitcode.com/openFuyao/ofep/blob/main/ofeps/sig-top/ofep-0003-%E8%AE%BE%E7%AB%8BAI%E6%8E%A8%E7%90%86SIG.md)。

**5、TC评审并反馈意见**

技术委员会例会申报议题，并在例会上讲解提案。技术委员会成员根据提案内容投票决定是否创建新的SIG。

**6.TC评审通过并合入**

技术委员会将通过合并Pull Request的方式来批准您的申请



<h2 id="id1">SIG变更申请流程</h2>

如果您要修改SIG团队成员、增删Repository等，您需要提交SIG变更申请流程。

<h3 id="id2-1">增删新项目或repository申请流程</h3>

**1、在sig文件夹的sig-info.yaml内添加新项目的repository信息或删除相关信息**

根据以上的信息，打开sig文件夹下sig-info.yaml文件，在`name`字段下面找到项目所属的sig，可以在该sig的`repo`的末尾添加项目的repository，或者找到待删除的repository进行删除。

```
vi sig-info.yaml

- name: sig-YOURSIGNAME
  repositories:
  - openeuler/aaa
  - src-openeuler/bbb
```

**2、提交PR**

将以上修改提交到GitCode上，并在GitCode上创建一个Pull Request。

**3、编写SIG变更提案**

将 gitcode.com/openFuyao/ofep Fork到你的GitCode下，在对应sig文件夹下编写SIG变更提案并提PR。

**4、TC评审并反馈意见**

技术委员会例会申报议题，并在例会上讲解提案。技术委员会成员根据提案内容投票决定是否变更SIG。

**5、TC评审通过并合入**

技术委员会将通过合并Pull Request的方式来批准您的申请



#### 内部变更申请流程

**只影响本SIG范围内的变更**，只需要SIG内的Maintainer达成一致，请走以下流程

**1、提交PR**

将以上修改提交到GitCode上，并在GitCode上创建一个Pull Request。

**2、在SIG内部发送邮件申请**

给您所对应的sig团队的邮箱列表发邮件申请，可以在正文中包含主题 *_[内部变更申请]_* 和PR的链接。如果之前在SIG团队内对此变更已经有讨论，可以省略该步骤。

**3、SIG内部评审并给出意见**

如果您的SIG内部已经有评审意见，可以省略该步骤。

**4.评审通过并合入**

SIG的Maintainer合并Pull Request来批准申请。



<h3 id="id2-2">变更团队成员申请流程</h3>

团队成员的刷新由SIG内部自己维护

**1.完成新SIG成员的配置**

请在`/community/sig`文件夹下找到您的sig文件夹，在sig-info.yaml文件中完成对SIG成员的配置，填写说明参考[博客](https://www.openeuler.org/zh/blog/georgecao/openEuler-sig-member-management.html)

```
vi sig-info.yaml

```

**2、提交PR**

将以上修改提交到GitCode上，并在GitCode上创建一个Pull Request。

**3、在SIG内部发送邮件申请**

给您所对应的sig团队的邮箱列表发邮件申请，邮件主题为 *_[变更团队成员申请]_*，邮件正文中应包含PR链接。如果之前在SIG团队内对此变更已经有讨论，可以省略该步骤。

**4、SIG内部评审并给出意见**

如果您的SIG内部已经有评审意见，可以省略该步骤。

**5.评审通过并合入**

SIG的Maintainer合并Pull Request来批准申请。