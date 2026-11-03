# markdownlint Rules

This document describes markdownlint rules and rule settings in the **openFuyao/docs** repository. For details, see <https://github.com/markdownlint/markdownlint/blob/v0.12.0/docs/RULES.md>.

## Introduction to markdownlint

markdownlint is a tool for checking the format of Markdown files. It can comprehensively check Markdown files based on rules. You can use the markdownlint plug-in of tools such as VS Code to fix format issues.

## Rule Settings for the openFuyao/docs Repository

- openFuyao rules use the following scheme:

    - Set the `style` parameter to `atx` for MD003 (Header style).
    - Set the `style` parameter to `ordered` for MD029 (Ordered list item prefix).
    - Disable MD004 (Unordered list style).
    - Disable MD007 (Unordered list indentation).
    - Disable MD009 (Trailing spaces).
    - Disable MD013 (Line length).
    - Disable MD014 (Dollar signs used before commands without showing output).
    - Disable MD020 (No space inside hashes on closed atx style header).
    - Disable MD021 (Multiple spaces inside hashes on closed atx style header).
    - Disable MD024 (Multiple headers with the same content).
    - Disable MD025 (Multiple top level headers in the same document).
    - Disable MD027 (Multiple spaces after blockquote symbol).
    - Disable MD033 (Inline HTML).
    - Disable MD036 (Emphasis used instead of a header).
    - Disable MD046 (Code block style).

- Ruby files are written as follows:

   ```bash
   all
   rule 'MD003', :style => :atx
   rule 'MD029', :style => :ordered
   exclude_rule 'MD004'
   exclude_rule 'MD007'
   exclude_rule 'MD009'
   exclude_rule 'MD013'
   exclude_rule 'MD014'
   exclude_rule 'MD020'
   exclude_rule 'MD021'
   exclude_rule 'MD024'
   exclude_rule 'MD025'
   exclude_rule 'MD027'
   exclude_rule 'MD033'
   exclude_rule 'MD036'
   exclude_rule 'MD046'
   ```

## Rule Introduction

### MD001: Header levels should only increment by one level at a time

- **Negative example**

  ```text
  # Header1

  ### Header3
  ```

- **Positive example**

  ```text
  # Header1

  ## Header2

  ### Header3

  #### Header4
  ```

### MD002: First header should be a top level header

- **Parameters**

    - `level`: level of the top-level header. The default value is **1**.
- **Negative example**

  ```text
  ## This is not an H1 header

  ### Another header
  ```

- **Positive example**

  ```text
  # Start with an H1 header

  ## Then use an H2 for subsections
  ```

### MD003: Header style

- **Parameters**

    - `style`: document header style. The value can be `consistent`, `atx`, `atx_closed`, `setext`, or `setext_with_atx`. **This parameter has been set to `atx` for this repository.**

- **Negative example**

  ```text
  # ATX style H1

  ## Closed ATX style H2 ##

  Setext style H1
  ===============
  ```

- **Positive example**

  ```text
  # ATX style H1

  ## ATX style H2
  ```

### MD004: Unordered list style

**This rule has been disabled for this repository.**

- **Parameters**

    - `style`: unordered list style. The options are as follows: `consistent` for using consistent symbols in the same list, `asterisk` for using asterisks (*), `plus` for using plus signs (+), `dash` for using hyphens (-), and `sublist` for sub-items to use different symbols compared to their parent list. The default value is `consistent`.
- **Negative example**

  ```text
  * Item 1
  + Item 2
  ```

- **Positive example**

  ```text
  * Item 1
  * Item 2
  ```

### MD005: Inconsistent indentation of list items at the same level

- **Negative example**

  ```text
  * Item1
      * nested item 1
      * nested item 2
     * A misaligned item
  ```

- **Positive example**

  ```text
  * Item1
      * nested item 1
      * nested item 2
      * nested item 3
  ```

### MD006: Consider starting bulleted lists at the beginning of the line

- **Negative example**

  ```text
  Some text

      * List item
      * List item
  ```

- **Positive example**

  ```text
  Some text

  * List item
  * List item
  ```

### MD007: Unordered list indentation

**This rule has been disabled for this repository.**

- **Parameters**

    - `ident`: number of spaces used for indentation when an unordered list is nested. The default value is **2**.
- **Negative example**

  ```text
  * List item
    * Nested list item indented by 4 spaces
  ```

- **Positive example**

  ```text
  * List item
      * Nested list item indented by 2 spaces
  ```

### MD009: Trailing spaces

**This rule has been disabled for this repository.**

- **Parameters**
    - `br_spaces`: number of spaces that can be added at the end of a line. The default value is **0**. It is recommended that this parameter be set to a value greater than or equal to 2. If the value is less than 2, the default value **0** is used.

### MD010: No hard tabs (using spaces for indentation)

- **Parameters**

    - `code_blocks`: whether the rule is effective for code blocks. The value can be **true** or **false**. The default value is **true**.
- **Negative example**

  ```text
  Some text

      * hard tab character used to indent the list item
  ```

- **Positive example**

  ```text
  Some text

      * Spaces used to indent the list item instead
  ```

### MD011: Reverse link syntax

- **Negative example**

  ```text
  (Incorrect link syntax)[http://www.example.com]
  ```

- **Positive example**

  ```text
  [Correct link syntax](http://www.example.com)
  ```

### MD012: Multiple consecutive blank lines

- **Parameters**

    - `maximum`: maximum number of consecutive blank lines allowed in a document. The default value is **1**.

- **Negative example**

  ```text
  Some text here
  
  Some more text here
  ```

- **Positive example**

  ```text
  Some text here

  Some more text here
  ```

> [!NOTE]NOTE
> If there are multiple consecutive blank lines in a code block, this rule will not be triggered.

### MD013: Line length

**This rule has been disabled for this repository.**

- **Parameters**

    - `line_length`: maximum length of a line. The default value is **80**.
    - `heading_line_length`: maximum length of a heading line. The default value is **80**.
    - `code_blocks`: whether the rule is effective for code blocks. The value can be **true** or **false**. The default value is **true**.
    - `tables`: whether the rule is effective for tables. The value can be **true** or **false**. The default value is **true**.
    - `headings`: whether the rule is effective for headings. The value can be **true** or **false**. The default value is **true**.

### MD014: Dollar signs used before commands without showing output

**This rule has been disabled for this repository.**

- **Negative example**

  ```text
  sl
  cat foo
  less bar
  ```

- **Positive example**

  ```text
  ls
  cat foo
  less bar
  ```

  ```text
  $ ls
  foo bar
  $ cat foo
  Hello world
  $ cat bar
  baz
  ```

  In a code block, a dollar sign ($) is not required before a terminal command. However, if the code contains both a terminal command and command output, a dollar sign ($) can be placed before the terminal command.

### MD018: No space after hash on atx style header

- **Negative example**

  ```text
  #Header1

  ##Header2
  ```

- **Positive example**

  ```text
  # Header1

  ## Header2
  ```

### MD019: Multiple spaces after hash on atx style header

- **Negative example**

  ```text
  #  Header1

  ##  Header2
  ```

- **Positive example**

  ```text
  # Header1

  ## Header2
  ```

### MD020: No space inside hashes on closed atx style header

**This rule has been disabled for this repository.**

- **Negative example**

  ```text
  #Header1#

  ##Header2##
  ```

- **Positive example**

  ```text
  # Header1 #

  ## Header2 ##
  ```

### MD021: Multiple spaces inside hashes on closed atx style header

**This rule has been disabled for this repository.**

- **Negative example**

  ```text
  #  Header1  #

  ##  Header2  ##
  ```

- **Positive example**

  ```text
  # Header1 #

  ## Header2 ##
  ```

### MD022: Headers should be surrounded by blank lines

- **Parameters**

    - `lines_above`: number of blank lines above the header line. The default value is **1**.
    - `lines_below`: number of blank lines below the header line. The default value is **1**.
- **Negative example**

  ```text
  # Header1
  Some text

  Some more text
  ## Header2
  ```

- **Positive example**

  ```text
  # Header1

  Some text

  Some more text

  ## Header2
  ```

### MD023: Headers must start at the beginning of the line

- **Negative example**

  ```text
  Some text
    ## Indented header
  ```

- **Positive example**

  ```text
  Some text

  ## Header
  ```

### MD024: Multiple headers with the same content

**This rule has been disabled for this repository.**

- **Negative example**

  ```text
  # Some text

  ## Some text
  ```

- **Positive example**

  ```text
  # Some text

  ## Some more text
  ```

### MD025: Multiple top level headers in the same document

**This rule has been disabled for this repository.**

- **Parameters**

    - `level`: highest header level of the document. The default value is **1**.
- **Negative example**

  ```text
  # Top level header

  # Another top level header
  ```

- **Positive example**

  ```text
  # Title

  ## Header

  ### Another header
  ```

### MD026: Trailing punctuation in header

- **Parameters**

    - `punctuation`: punctuation marks that cannot appear at the end of a header. The default value is **".,;:!?"**
- **Negative example**

  ```text
  # This is a header.
  ```

- **Positive example**

  ```text
  # This is a header
  ```

### MD027: Multiple spaces after blockquote symbol

**This rule has been disabled for this repository.**

- **Negative example**

  ```text
  >  This is a block quote with bad indentation
  >  there should only be one
  ```

- **Positive example**

  ```text
  > This is a block quote with bad indentation
  > there should only be one
  ```

### MD028: Blank line inside blockquote

- **Negative example**

  ```text
  > This is a blockquote
  > which is immediately followed by

  > this blockquote. Unfortunately
  > in some parsers, this are treated as the same blockquote.
  ```

- **Positive example**

  ```text
  > This is a blockquote.
  >
  > This is the same blockquote.
  ```

### MD029: Ordered list item prefix

- **Parameters**

    - `style`: format of ordered list item prefixes. The options are as follows: `one` for using only 1 as the prefix, and `ordered` for using incrementing numbers starting from 1 as the prefix. The default value is `one`. **This parameter has been set to `ordered` for this repository.**
- **Negative example**

  ```text
  1. Do this
  1. Do that
  1. Done
  ```

- **Positive example**

  ```text
  1. Do this
  2. Do that
  3. Done
  ```

### MD030: Spaces after list markers

- **Parameters**

    - `ul_single`: number of spaces between the prefix symbol and text in a single paragraph of an unordered list. The default value is **1**.
    - `ol_single`: number of spaces between the prefix symbol and text in a single paragraph of an ordered list. The default value is **1**.
    - `ul_multi`: number of spaces between the prefix symbol and text in multiple paragraphs of an unordered list. The default value is **1**.
    - `ol_multi`: number of spaces between the prefix symbol and text in multiple paragraphs of an ordered list. The default value is **1**.
- **Negative example**

  ```text
  *  Foo
  *  Bar
  *  Baz
  ```

- **Positive example**

  ```text
  * Foo
  * Bar
  * Baz
  ```

### MD031: Fenced code blocks should be surrounded by blank lines (not required at the beginning or end of the document)

- **Negative example**

  ````text
  Some text
  ```

  Code block

  ```

  ```

  Another code block

  ```
  Some more text
  ````

- **Positive example**

  ````text
  Some text

  ```

  Code block

  ```

  ```

  Another code block

  ```

  Some more text
  ````

### MD032: Lists should be surrounded by blank lines (not required at the beginning or end of the document)

- **Negative example**

  ```text
  Some text
  * Some
  * List

  1. Some
  2. List
  Some text
  ```

- **Positive example**

  ```text
  Some text

  * Some
  * List

  1. Some
  2. List

  Some text
  ```

### MD033: Inline HTML

**This rule has been disabled for this repository.**

- **Negative example**

  ```text
  <h1>Inline HTML header</h1>
  ```

- **Positive example**

  ```text
  # Markdown header
  ```

### MD034: Bare URL used

- **Negative example**

  ```text
  For more information, see http://www.example.com/.
  ```

- **Positive example**

  ```text
  For more information, see <http://www.example.com/>.
  ```

### MD035: Horizontal rule style

- **Parameters**

    - `style`: method of creating a horizontal line. The value can be `consistent`, `***`, `---`, or other strings that specify a horizontal line. The default value is `consistent`.

### MD036: Emphasis used instead of a header

**This rule has been disabled for this repository.**

- **Parameters**

    - `punctuation`: punctuation marks used at the end of a line. If an emphasis ends with one of the specified punctuation marks, the emphasis is not considered as a header replacement. The default value is **".,;:!?"**
- **Negative example**

  ```text
  **My document**

  Lorem ipsum dolor sit amet...

  _Another section_

  Consectetur adipiscing elit, sed to eiusmd
  ```

- **Positive example**

  ```text
  # My document

  Lorem ipsum dolor sit amet...

  ## Another section

  Consectetur adipiscing elit, sed to eiusmd
  ```

### MD037: Spaces inside emphasis markers

- **Negative example**

  ```text
  Here is some ** bold ** text

  Here is some * italic * text

  Here is some more __ bold __ text

  Here is some more _ italic _ text
  ```

- **Positive example**

  ```text
  Here is some **bold** text

  Here is some *italic* text

  Here is some more __bold__ text

  Here is some more _italic_ text
  ```

### MD038: Spaces inside code span elements

- **Negative example**

  ```text
  ` some text `
  `some text `
  ` some text`
  ```

- **Positive example**

  ```text
  `some text`
  ```

### MD039: Spaces inside link text

- **Negative example**

  ```text
  [ a link ](http://www.example.com/)
  ```

- **Positive example**

  ```text
  [a link](http://www.example.com/)
  ```

### MD040: Fenced code blocks should have a language specified

- **Negative example**

  ````text
  #!/bin/bash
  echo Hello world
  ```

- **Positive example**

  ````bash
  ​```bash
  #!/bin/bash
  echo Hello world
  ​```
  ```

- **Common code block programming languages**

  | Supported Language         | Keyword   |
  | ---------------- | -------- |
  | Python           | python   |
  | C                | cpp      |
  | Java             | java     |
  | Shell            | bash     |
  | Markdown         | markdown |
  | JavaScript       | js       |
  | CSS              | css      |
  | SQL              | sql      |
  | PHP              | php      |
  | Text             | text     |
  | XML              | html     |
  | Bat              | bat      |
  | Protocol Buffers | protobuf |

### MD041: First line in file should be a top level header

- **Parameters**

    - `level`: highest header level of the document. The default value is **1**.
- **Negative example**

  ```text
  This is a file without a header
  ```

- **Positive example**

  ```text
  # File with header

  This is a file with a top level header
  ```

### MD046: Code block style

**This rule has been disabled for this repository.**

- **Parameters**

    - `style`: code block definition format. The options are as follows: `fenced` for using three backticks, `indented` for using indentation, and `consistent` for context consistency. The default value is `fenced`.
- **Negative example**

  ```text
  Some text.

      Code block

  Some more text.
  ```

- **Positive example**

  ````text
  Some text.

  ​```ruby
  Code block
  ​```
  Some more text.
  ```
  
  

### MD047: File should end with a single newline character

- **Negative example**

  ```text
  # Header

  This file ends without a newline.[EOF]
  ```

- **Positive example**

  ````text
  # Header

  This file ends with a newline.
  [EOF]
  ````

## Markdown Plug-in in VS Code

The markdownlint extension library contains a rule library for Markdown files to ensure consistency with Markdown standards. After the configuration is added, markdownlint can automatically check document errors.

### Installation

- Press `Ctrl+Shift+X` to open the extension tab.
- Enter `markdownlint` to find the extension.
- Click `Install` and then click `Enable`.

### Configuration

The markdownlint version used in VS Code is formulated by David Anson and is different from the official markdownlint version v0.12.0 used in the openFuyao repository. To keep the configuration consistent with that in the openFuyao repository, refer to the following configuration items:

- On the command panel (`Ctrl+Shift+P`), enter the `Open Settings (JSON)` command.
- Add the following configuration to the JSON object:

   ```bash
   "markdownlint.config":{
      "default":true,
      "MD003":{"style":"atx"},
      "MD029":{"style":"ordered"},
      "MD004":false,
      "MD007":false,
      "MD009":false,
      "MD013":false,
      "MD014":false,
      "MD020":false,
      "MD021":false,
      "MD024":false,
      "MD025":false,
      "MD033":false,
      "MD036":false,
      "MD042":false,
      "MD043":false,
      "MD044":false,
      "MD045":false,
      "MD046":false,
      "MD048":false,
      "MD049":false,
      "MD050":false,
      "MD051":false,
      "MD052":false,
      "MD053":false,
      "MD055":false,
      "MD056":false,
      "MD057":false
   }
   ```
