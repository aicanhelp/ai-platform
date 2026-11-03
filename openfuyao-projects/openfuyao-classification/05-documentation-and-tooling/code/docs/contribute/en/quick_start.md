# Quick Start

## Overview

openFuyao documents are written in Markdown format, managed using Git for version control and hosted on the GitCode platform. Document modifications are reviewed and merged using pull request (PR) workflows. openFuyao documents are organized into separate repositories. Basic feature documents such as **Release Notes**, **Legal Statement**, and **Obtaining openFuyao Resources** are stored in the **openFuyao/docs** repository and maintained by the Doc SIG. Incremental feature documents such as **Colocation** under **User Guide** are stored in the code repository of each feature and maintained by the SIG of each feature.

The document center categorizes manuals based on the developer journey. The common document development scenarios are as follows:

- Adding a level-1 directory: Contact Doc SIG maintenance personnel.
- Adding a manual: When adding a feature, you need to add a feature manual to the corresponding level-1 directory. In addition to maintaining the document content in the feature code repository, you need to configure the **_toc.yaml** file in the corresponding directory of the **openFuyao/docs** repository.
- Modifying a manual: For example, to fix basic errors or to add or delete sections, you can directly update the document in the corresponding code repository.

## General Procedure

First, you need to confirm the repository where the document to be modified is located. You can locate the repository and the specific storage directory based on the manual name. The following uses **User Guide** as an example to describe the general process of adding, modifying, and deleting documents.

1. Clone the repository.

    Fork the remote repository, clone the forked repository to the local machine, and associate the local repository with the remote repository.

    ```bash
    git clone git@gitcode.com:Xilyqaz/docs.git

    git remote add upstream git@gitcode.com:openFuyao/docs.git
    ```

2. Switch to a branch.

    Switch to the branch corresponding to the version of the document you need to modify. In normal cases, the branch name is in the format of `openFuyao-Version number`. v25.06 is used as an example.

    ```bash
    git fetch upstream

    git checkout openFuyao-v25.06

    git rebase upstream/openFuyao-25.06

    git checkout -b "p-v25.06" # p-v25.06 is a personal branch created by the user.
    ```

3. Submit changes.

    For details about how to make changes to documents, see [Adding Documents](#adding-documents), [Modifying Documents](#modifying-documents), and [Deleting Documents](#deleting-documents).

    ```bash
    git add .

    git commit -m "Reason for the submission"
    ```

4. Push the changes to the remote repository and create a PR.

    ```bash
    git push origin p-v25.06
    ```

## Adding Documents

The following guides developers to add a complete manual.

- Find the [storage location](./document_organization_structure.md#document-storage-locations) of the manual, add the corresponding document content file, and modify the directory structure file `_toc.yaml` of the manual.

    For example, to add the **NPU Operator** manual to **User Guide**, add the document content file `NPU_Operator.md` to the storage directory of **User Guide**, and add the reference to `NPU_Operator.md` into `_toc.yaml` of **User Guide**.

    ```yaml
    isManual: true                                                  # ismanual: identifies the file as the directory structure file of the manual.
    sections:
      - label: User_Guide                                           # Level-1 directory: User Guide
        sections:
    +     - label: npu_operator                                     # Name of the new manual: NPU_Operator // [!code ++]
    +       href: ./npu_operator.md                                 # Address of the content file of the new manual // [!code ++]
    ```

## Modifying Documents

Document modifications fall into the following two categories:

- Modifying only document content: Modify the document content in a specific section of the current manual. For example, modify the content of the "Application Scenarios" section in **NPU_Operator.md** under **User Guide**. Find the [storage location](./document_organization_structure.md#document-storage-locations) of the document and adjust the content accordingly. If only the content is modified, the directory structure file **_toc.yaml** does not need to be modified.
  
- Modifying a manual at the official website: Find the `_toc.yaml` file of the document and make modifications.

   For example, if you want to move `Web_Terminal.md` to the front of `NPU_Operator.md` in the level-1 directory **User Guide**, find the `_toc.yaml` file of **User Guide** and modify the file as follows:

    ```yaml
    label: User_Guide                                           # Level-1 directory: User Guide
    sections:                                                       
   +  - href: ./web_terminal.md                            # web_terminal.md is above npu_operator.md. // [!code ++]
   +  - href: ./npu_operator.md                            # Manual: npu_operator.md // [!code ++]
   -  - href: ./web_terminal.md                            # Manual: web_terminal.md // [!code --]      
    ```

## Deleting Documents

The following guides developers to delete a complete manual. To delete content from an existing manual, for example, to delete the "Application Scenarios" section from **NPU_Operator.md** in **User Guide**, see [Modifying Documents](#modifying-documents).

The following uses `NPU_Operator.md` in **User Guide** as an example to describe how to delete a manual.

1. Go to the [storage location](./document_organization_structure.md#document-storage-locations) of the manual to be deleted and delete it.

2. Delete the reference to **NPU_Operator.md** from the **_toc.yaml** file in the **docs** repository.

    ```yaml
    label: User_Guide                                             # Document page
    sections:
    - - href: ./npu_operator.md                                   # Delete the reference to the directory structure file of NPU_Operator.md. // [!code --]
      - href: ./web_terminal.md
      - href: ./registration_and_login.md 
    ```
