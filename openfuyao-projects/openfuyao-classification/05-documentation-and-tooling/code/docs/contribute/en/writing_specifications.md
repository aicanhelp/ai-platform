# Writing Specifications

This document describes the writing specifications of openFuyao documentation contributions.

## Naming Rules

If you want to add a document, add a Markdown document (file with the .md file name extension) to the corresponding file directory. Document names must be declared based on the document content.

- The name of a new document cannot be the same as that of an existing document. If the name is duplicate, rename the document.
- Document names cannot contain parentheses (), brackets [], braces {}, or angle brackets <>. Otherwise, the document directory cannot be properly displayed. You can use underscores (_) or hyphens (-) instead.

## Content Standards

The primary goal is to convey content concisely and clearly. Introductory documents should clearly and succinctly explain principles, architectures, and design rationale, and operational documents must specify key steps to provide practical assistance to other developers.

**Headings**

- Headings should be brief and clearly reflect the main content of sections. Do not omit necessary information. For example, instead of a vague heading like **Overview**, use **XXX Overview** to ensure that readers can immediately grasp the context.
- The headings of operational documents should be in the verb-object structure, for example, **Creating a Task**.
- The headings of the same level and type must be of the same structure.
- Headings cannot end with a punctuation mark.
- Parentheses () can be used in headings to provide supplementary information. Headings cannot contain other special characters, such as question marks (?).  
- A blank line must be used to separate a heading and the body text.
- Headings are written in # *Heading name* format. Note that a space must be used between the number sign (#) and the heading name. Only one heading level can be added at a time, and the first heading must be the top-level heading.

**Directories**

The .md files are stored at the following addresses:

- Official website: <https://docs.openfuyao.cn/>
- GitCode: <https://gitcode.com/openFuyao/docs>

**Spaces**

- Do not add spaces between Chinese and English characters when editing a document.
- Do not add spaces between Chinese characters and digits.
- Do not add spaces between digits and units.
- Do not add spaces between links and the body text.
- Do not add spaces between double-byte characters (DBCs) and other characters.

**Images**

- Images of a document are stored in the **figures** folder that is in the same directory as the document. For example, when documents in the directory reference images, relative paths are used for reference.
- Original images must be used to prevent intellectual property infringement.
- Images must be clear and legible, and image information must be complete. For example, the start and end points must be included in a flowchart.
- The image logic must be clear.
- Images and text must be used together. Do not separate images from text.
- The PNG format is preferred, and the JPG format is also acceptable.
- Chinese images must be used for Chinese documents, and English images must be used for English documents.
- Image paths cannot contain Chinese characters.
- The image file name extension must be in lowercase. For example, the file name extension cannot be .PNG.
- It is recommended that images be named based on their content. Using only a number sequence does not facilitate image inheritance.
- For screenshots, ensure that only relevant information within the allowed scope is retained. Key information in screenshots can be highlighted with red frames or text notes.

Image writing example:

```
Images are written in the ![](./figures/pic_name.png) format. The ./ part is mandatory. Otherwise, images cannot be displayed on the official website.
```

**Code Blocks**

Code examples show how to implement specific functions. Developers write and debug code based on code examples. Code requirements are as follows:

- The logic and syntax of code must be correct.

- It is recommended that the input and output of code be separated.

- Comments must be provided for key steps in code.

- Code must be derived from concrete examples.

- Inline code and command lines in documents must be enclosed in a pair of backticks (``), for example, `code line`.

- Block-level code is indented with three backticks (```) or four spaces (the **Tab** key cannot be used), with a blank line above and below for separation. The following shows an example:

  ```python
    #!/usr/bin/env python3
    print("Hello, World!")
    
  ```

  ```
    #include  <stdio.h>
    int main(void)
    {
        printf("Hello world\n");
    }
  ```

**Annotation Symbols**

The following annotation symbols are used in documents to indicate different application scenarios and levels of emphasis. You can select the corresponding annotation symbol based on the importance of the information to be highlighted for users.

| Symbol| Description                                                        | Reference Method                                  |
| ---- | ------------------------------------------------------------ | ------------------------------------------ |
| CAUTION| Indicates a hazard with a low level of risk which, if not avoided, could result in minor or moderate injury.| &gt; icon image **CAUTION**<br>&gt; Body text|
| NOTICE| Indicates a potentially hazardous situation which, if not avoided, could result in equipment damage, data loss, performance deterioration, or unanticipated results.<br>NOTICE is used to address practices not related to personal injury.| &gt; icon image **NOTICE**<br>&gt; Body text|
| NOTE| Supplements the important information in the main text.<br>NOTE is used to address information not related to personal injury, equipment damage, and environment deterioration.| &gt; icon image **NOTE**<br>&gt; Body text|

**Tables**

You are advised to create tables in Markdown documents using the following method, instead of the HTML table style.

```
|  Table header   | Table header  |
|  ----  | ----  |
| Cell  | Cell |
| Cell  | Cell |
```

Rules for using punctuation marks in cells:

1. If a column in the table contains only terms or phrases, do not use any punctuation mark.
2. If a column in the table contains only sentences, use periods (.).
3. In special cases, if the mixed use of phrases and sentences cannot be avoided, use periods (.).

**Document PR Review Checklist**

- Specifications issues and basic errors

  - Misspellings or punctuation errors
  - Incorrect links, empty cells, or incorrect formats
  - Chinese characters in English documents
  - Descriptions inconsistent with WebUIs, without affecting operations
  - Awkward phrasing that does not affect understanding
  - Unmatched version numbers, such as version numbers in software package names and version numbers on WebUIs
  - Inconsistent wording for the same content in the same document
  - Non-standard software names
  - Colloquial language in descriptions
  - Non-standard images or tables
  - Non-standard or misspelled proper nouns
  - Missing half of a pair of punctuation marks, or non-standard usage of paired punctuation marks
  - Inconsistent case of commands in headings and body text
  - Images not displayed, oversized, wider than the body text or page, distorted, missing, or redundant
  - Commands and command explanations written together
  - Repetitive description of the same content in the same location
  - Operation screenshots inconsistent with the actual environment
  - Commands in the step description inconsistent with those in the command output

- Usability issues

  - Incorrect or missing key steps, which cannot guide users to complete tasks
  - Missing prerequisites or precautions
  - Unclear operation scenarios
  - Obscure images, tables, or text
  - Unclear logic, such as the lack of categories, item lists, or detailed steps
  - Steps not easy to understand or learn, such as the lack of command description
  - Usage methods or constraints provided for scenarios that are explicitly stated as unsupported
  - Non-standard document names, making it difficult to quickly locate key points in manuals
  - Step description inconsistent with actual operations

- Accuracy issues

  - Description of technical principles, functions, specifications, mapping versions, or OSs incorrect or inconsistent with the software
  - Incorrect schematic diagrams or architecture diagrams
  - Incorrect commands, command parameters, or IP addresses
  - Commands incapable of implementing the corresponding functions
  - Screenshots inconsistent with WebUIs, affecting actual operations
  - Incorrect links, leading to redirection to unwanted versions or content
  - Accessible links pointing to outdated or removed content
  - Non-standard code formats, such as missing line breaks, incorrect indentation, or unmatched parentheses
  - Redundant spaces in commands, leading to command errors

- Risk warning issues

  - Lack of security warnings for risky operations on important data or the system

- Compliance issues

  - Violation of laws and regulations, or use of sensitive words related to politics and territorial sovereignty
  - Copyright infringement
