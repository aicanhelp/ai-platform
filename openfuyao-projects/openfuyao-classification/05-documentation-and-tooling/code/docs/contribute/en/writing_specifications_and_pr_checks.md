# Writing Specifications and PR Checks

## Writing Specifications

### Content Element Specifications

[See the writing specifications](https://gitcode.com/openFuyao/docs/blob/master/contribute/%E5%86%99%E4%BD%9C%E8%A7%84%E8%8C%83.md).

### Language Style Specifications

[Rule] Submitted content must be related to openFuyao features.

[Rule] The content cannot contain sensitive information, or content with strong racial or gender discrimination.

[Rule] Submitted content must be original and must not infringe others' intellectual property rights.

[Rule] Submitted content must be objective and authentic. Words like exaggerated promotion are not allowed.

**Unwelcome behaviors in document contribution**

Submit a large number of PRs in a short period of time using automation tools to primarily address harmless errors, such as spelling errors, grammar errors, date errors, and awkward phrasing.

## PR Checks in the Document Release Pipeline

To improve document quality, the **openFuyao/docs** repository introduces automatic review tools to check basic errors in documents.

After a developer submits a PR, PR checks are automatically triggered.

Passing PR checks is one of the prerequisites for PR incorporation. If an error is displayed for a check item, you can view the error details.

The following subsections describe the check items.

### markdownlint

markdownlint checks the formats of Markdown documents. For details about markdownlint rules and the rule settings in this repository, see the [**check rules**](./markdownlint_rules.md).

### markdown/html-validation

markdown/html-validation checks the HTML tag closure issue in a document. Note that HTML tags in code blocks will not be scanned.

Negative example:

  ```html
    <table>
      <tr>
        <th>Header 1
        <th>Header 2</th>   <!-- Error 1: Unclosed <th> -->
      <tr>                  <!-- Error 2: Unclosed <tr> -->
        <td>Data 1<td>      <!-- Error 3: Self-closed <td> -->
        <td>Data 2</td>
    </table>                <!-- Error 4: Unclosed <tr>Causing structure disorder -->
  ```

Positive example:

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

markdown/link-validation checks whether all links in a document are valid.

Negative example:

  ```text
  ![Incorrect website](https://doc.openfuyao.cn/)  <!-- 404 error due to incorrect spelling -->
  ```

Positive example:

  ```text
  ![Correct website](https://docs.openfuyao.cn/)
  ```

### markdown/resource-existence

This check item checks whether the local images or links are valid.

Negative example:

  ```text
  ![CI image](./CI check result.jpg)  <!-- Incorrect relative path of the image -->
  ```

Positive example:

  ```text
  ![CI image](./figures/CI check result.jpg)
  ```

### docs/toc-check

- To ensure that a new document can be displayed in the openFuyao document center, you need to add the chapter where the new document is located to the corresponding `_toc.yaml` file. Otherwise, an error is reported. For details about how to write a new document, see the [document addition guide](./quick_start.md#adding-documents) in **Quick Start**. For details about how to write a `_toc.yaml` file, see [_toc.yaml Writing Specifications](./document_organization_structure.md#formats-of-directory-structure-files).

- When a document name is changed, docs/toc-check checks whether the document name is also changed in the corresponding `_toc.yaml` file.

- When docs/toc-check checks documents, a full document check is triggered. It checks whether all documents referenced in each `_toc.yaml` file exist in the corresponding document paths. If any document referenced in a `_toc.yaml` file does not exist, an error is reported.

### codespell

codespell is used to check spelling errors in documents. For details, see [https://github.com/codespell-project/codespell](https://github.com/codespell-project/codespell).

If any special words need to be ignored, contact [andxibei](https://gitcode.com/andxibei) or [Xilyqaz](https://gitcode.com/Xilyqaz) for adding the words to an exclusion list.
