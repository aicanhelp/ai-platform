# 规范与门禁

## 文档写作规范

### 内容元素规范

[参见写作规范](./writing_specifications.md)

### 语言风格规范

【规则】提交内容必须是与openFuyao特性相关内容。

【规则】内容不能包含敏感信息、有强烈的种族歧视或性别歧视的内容。

【规则】提交的内容必须是原创内容，不得侵犯他人知识产权。

【规则】提交的内容必须客观、真实，不允许使用夸大宣传等词汇。

**文档贡献中不受欢迎的行为**

短时间内通过自动化工具，提交大量的PR，提交大量的处理诸如拼写错误，语法错误，日期错误，语句不通顺等“无害的错误”的修正。 

## 文档发布流水线门禁

为提升文档质量，openFuyao docs仓引入了自动化检视工具，对文档中低错问题进行排查。

开发者提交PR后，会自动触发门禁进行检查。

通过门禁检查是PR合入的必要条件之一，检查项提示错误可查看错误详细信息。

下面将对各个检查项进行介绍：

### Markdown Lint

Markdown Lint对markdown文档的格式进行检视。markdownlint规则介绍及本仓规则设置，请参考[**检查规则**](markdownlint-rules.md)。

### markdown/html-validation

markdown/html-validation将检查Markdown中HTML标签的语法规范与渲染兼容性。

反例：

  ```html
    <table>
      <tr>
        <th>Header 1
        <th>Header 2</th>   <!-- 错误1：未闭合的<th> -->
      <tr>                  <!-- 错误2：未闭合的<tr> -->
        <td>Data 1<td>      <!-- 错误3：自闭合<td> -->
        <td>Data 2</td>
    </table>                <!-- 错误4：未闭合的<tr>导致结构错乱 -->
  ```

正例：

  ```html
    <table>
      <tr>
        <th>Header 1</th>
        <th>Header 2</th>
      </tr>
      <tr>
        <td>Data 1</td>
        <td>Data 2</td>
      </tr>
    </table>
  ```

### markdown/link-validation

markdown/link-validation将检查文档中出现的所有链接是否有效。

反例：

  ```text
  ![错误官网](https://doc.openfuyao.cn/)  <!-- 拼写错误导致 404 -->
  ```

正例：

  ```text
  ![正确官网](https://docs.openfuyao.cn/)
  ```

### markdown/resource-existence

此检查项将确认本地图片或者链接是否有效。

反例：

  ```text
  ![ci图片](./ci检查结果.jpg)  <!-- 图片相对路径错误 -->
  ```

正例：

  ```text
  ![ci图片](./figures/ci检查结果.jpg)
  ```

### docs/toc-check

- 新增文档为确保能在openFuyao文档官网展示，需要在对应`_toc.yaml`文件增加所在章节位置，否则此检查项报错。新增文档场景写作流程可参考快速入门的[指导](./quick_start.md#新增文档)，`_toc.yaml`文件写作可参考[_toc.yaml文件写作规范](./document_organization_structure.md#目录结构文件格式)。

- 变更文档名称时，docs/toc-check会检查对应`_toc.yaml`文件是否同步修改文档名称。

- 在进行docs/toc-check检查时，门禁会对文档进行全量检查，检查每个`_toc.yaml`文件中所有的文件是否存在于文档对应的路径下，若`_toc.yaml`文件中记录的文件不存在，则会报错。

### codespell

codespell主要用于检查文档中的单词拼写错误，详细信息可参考[https://github.com/codespell-project/codespell](https://github.com/codespell-project/codespell)。

如果有特殊单词需要加入忽略清单，可联系[andxibei](https://gitcode.com/andxibei)或[Xilyqaz](https://gitcode.com/Xilyqaz)。
