# 快速入门

## 概述

openFuyao文档采用Markdown格式编写，通过Git进行版本管理，托管于GitCode平台，文档修改通过Pull Request（PR）工作流进行审核与合并。openFuyao文档分仓存储，《发行说明》、《法律声明》、《资源获取》等基础特性文档存在openFuyao/docs仓，由DOC SIG维护。《在离线混部用户指南》等增量特性文档分别存在各特性代码仓，由各特性SIG组维护。

文档中心将手册按开发者之旅分类，常见的文档开发场景如下：

- 新增一级目录：需联系DOC SIG maintainer修改。
- 新增手册：新增功能特性时，需在对应一级目录下新增特性手册，除了需要在特性代码仓维护文档内容外，还需要配置openFuyao/docs仓对应目录的 _toc.yaml 文件。
- 修改手册：包括低错问题修复、章节增删等内容调整，直接在对应代码仓中更新文档即可。

## 通用流程

首先，要明确待修改文档所在的仓库。可依据手册名称，定位对应的文档仓库及具体存放目录。下面以《用户指南》为例，介绍新增文档、修改文档，删除文档三个任务场景的通用流程。

1. 克隆仓库。

    fork远程仓库，克隆fork仓库到本地，并将本地仓库与远程仓库关联。

    ```bash
    git clone git@gitcode.com:Xilyqaz/docs.git

    git remote add upstream git@gitcode.com:openFuyao/docs.git
    ```

2. 切换分支。

    依据所需修改文档的版本，切换到对应的分支。通常情况下，分支命名规则为`openFuyao-版本号`。此处以v25.06版本为例。

    ```bash
    git fetch upstream

    git checkout openFuyao-v25.06

    git rebase upstream/openFuyao-25.06

    git checkout -b "p-v25.06" # p-v25.06为个人创建的分支
    ```

3. 提交变更。

    文档变更具体操作参考[新增文档](#新增文档)、[修改文档](#修改文档)、[删除文档](#删除文档)章节。

    ```bash
    git add .

    git commit -m "提交原因"
    ```

4. 推送到远程仓库，并创建PR。

    ```bash
    git push origin p-v25.06
    ```

## 新增文档

新增文档旨在指导开发者新增一本完整的手册。

- 找到手册的[存放位置](./document_organization_structure.md#文档存放地址)，新增对应的内容文件，并修改手册的目录结构文件`_toc.yaml`。

    以在《用户指南》中增加“NPU Operator”手册为例，首先在《用户指南》的存放目录下，新增文档内容文件`NPU Operator.md`，并在《用户指南》的`_toc.yaml`下，增加对`NPU Operator.md`的引用：

    ```yaml
    isManual: true                                                  # ismanual：标识此文件是手册的目录结构文件
    sections:
      - label: User_Guide                                           # 一级目录名：《用户指南》
        sections:
    +     - label: npu_operator                                     # 新增手册名：NPU_Operator // [!code ++]
    +       href: ./npu_operator.md                                 # 新增手册的内容文件地址 // [!code ++]
    ```

## 修改文档

文档修改分为以下2种情况：

- 仅内容修改：在现有手册章节中修改文档内容，例如，在《用户指南》的NPU Operator.md中修改“应用场景”章节的内容。找到文档的[存放位置](./document_organization_structure.md#文档存放地址)，并对其内容进行相应调整，仅内容修改无需改动目录结构文件（_toc.yaml）。
  
- 手册在官网修改：找到所属文档的`_toc.yaml`并修改。

   例如，在《用户指南》一级目录下，若开发者希望将`web_terminal.md`调整至`npu_operator.md`前面显示，需找到用户指南的`_toc.yaml`文件并加以修改：

    ```yaml
    label: User_Guide                                      # 一级目录：用户指南
    sections:                                                       
   +  - href: ./web_terminal.md                            # web_terminal.md放在npu_operator.md上面 // [!code ++]
   +  - href: ./npu_operator.md                            # 手册：npu_operator.md // [!code ++]
   -  - href: ./web_terminal.md                            # 手册：web_terminal.md // [!code --]      
    ```

## 删除文档

删除文档章节旨在指导开发者删除一本完整的手册。若要在现有手册中删除内容，比如在《用户指南》中的NPU Operator.md删除“应用场景”章节，则参考[修改文档](#修改文档)。

以删除《用户指南》中的`NPU_Operator.md`手册为例，可参考以下操作步骤：

1. 首先，确定想要删除手册的[文档存放地址](./document_organization_structure.md#文档存放地址)，将其删除。

2. 在docs的 _toc.yaml文件中，删除对NPU_Operator.md的引用。

    ```yaml
    label: User_Guide                                             # 文档页
    sections:
    - - href: ./npu_operator.md                                   # 删除对npu_operator.md目录结构文件的引用 // [!code --]
      - href: ./web_terminal.md
      - href: ./registration_and_login.md 
    ```
