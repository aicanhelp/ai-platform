# Installation Tool Download and Verification

This document uses the ARM64 architecture as an example to describe installation package integrity verification. The AMD64 architecture is handled similarly.

## Introduction

To confirm whether an installation package is incomplete or tampered with during transmission due to network connection issues, storage device problems, malicious attacks, or other reasons, you need to perform integrity verification on the installation package after obtaining it. Only installation packages that pass the verification can be deployed.

Here, by comparing the checksum recorded in the checksum file with the checksum of the installation package file calculated manually, you can determine whether the installation package is intact. The legitimacy of the checksum file is verified through the signature file (`.asc`) and the PGP public key. The complete trust chain is as follows.

```
PGP public key (primary key displayed after import) <--> Primary key output from asc signature verification --> Checksum file --> Installation package
```

> ![icon](../figures/icon-note.gif) **Note:**<br />The community PGP primary key is updated upon key rotation, and the document no longer provides a static fingerprint. During verification, you only need to confirm that the primary key displayed after importing the public key on this machine is consistent with the primary key output by gpg when verifying the sha256 file using the `.asc` file. Once they match, the signature verification passes. The displayed key does not need to be matched against any fixed fingerprint or keys from other channels.

## Prerequisites

Before verifying the installation package integrity, prepare the following files.

- openFuyao community PGP public key: `openfuyao.gpg`, [click to download](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/openfuyao.gpg).
- Installation package file: `bkeadm_linux_arm64`, [click to download](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/bkeadm_linux_arm64).
- sha256 checksum file: `bkeadm_linux_arm64.sha256`, [click to download](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/bkeadm_linux_arm64.sha256).
- gpg signature file: `bkeadm_linux_arm64.sha256.asc`, [click to download](https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/bkeadm_linux_arm64.sha256.asc).

## Operation Guide

Place the above four files in the same directory. The file integrity verification steps are as follows.

1. If you are downloading the PGP public key for the first time, run the following command to import the public key.

   ```shell
   gpg --import openfuyao.gpg
   ```

   After the import is complete, run the following command to display the currently imported public key and its primary key fingerprint. The key information is subject to the actual output on this machine (dynamically loaded and updated with the public key file). Record the primary key fingerprint in the output for comparison in the next step.

   ```shell
   gpg --list-keys --with-fingerprint --with-subkey-fingerprint contact@openfuyao.cn
   ```

2. Run the following command to verify the sha256 file using the `.asc` signature file and dynamically read the primary key used for this signature from the verification output.

   ```shell
   gpg --verify bkeadm_linux_arm64.sha256.asc bkeadm_linux_arm64.sha256
   ```

   After the command is executed, the sha256 checksum file is legitimate if all the following conditions are met:

   - The output contains `Good signature from "openFuyao <contact@openfuyao.cn>"`.
   - The key fingerprint in the output is consistent with the primary key fingerprint displayed in step 1. The fingerprint field name varies by GPG version, commonly `Primary key fingerprint` or `Key fingerprint`. **Refer to the actual `gpg --verify` output**; for comparison, only the fingerprint value needs to be consistent, without relying on a fixed English field name.

   The displayed key does not need to match fixed fingerprints from the document or other channels. As long as the primary key displayed in step 1 is consistent with the primary key output by gpg verification in this step, the verification passes.

   If verification fails (for example, `BAD signature` appears, the signature cannot be verified, or the primary key fingerprint is inconsistent with that in step 1), proceed as follows:

   - Delete the already downloaded `openfuyao.gpg`, `bkeadm_linux_arm64.sha256`, and `bkeadm_linux_arm64.sha256.asc` from this directory.
   - Re-obtain the above files from the official download addresses in [Prerequisites](#prerequisites) of this document, and confirm that the file names and architecture match.
   - If an old public key has been imported on this machine before, you can delete the old key first and then re-import it:

     ```shell
     gpg --delete-keys contact@openfuyao.cn
     gpg --import openfuyao.gpg
     ```

   - Re-run the import and `gpg --verify` as described in step 1 and step 2 above.

   If verification still fails after re-obtaining and verifying the files following the above steps, contact the openFuyao sig-installation mailbox [installation@list.openfuyao.cn](mailto:installation@list.openfuyao.cn) or the openFuyao community mailbox [contact@openfuyao.cn](mailto:contact@openfuyao.cn), and attach the complete output of `gpg --verify`.

3. Run the following command to verify the installation package file integrity.

   ```shell
   sha256sum -c <(cat bkeadm_linux_arm64.sha256) < bkeadm_linux_arm64
   ```

   After the command is executed, if the output is `-: OK`, the installation package file integrity is intact. Otherwise, the installation package integrity has been compromised. Re-download the installation package and the corresponding `.sha256` / `.sha256.asc` files and verify again. If it still fails, contact the openFuyao sig-installation mailbox [installation@list.openfuyao.cn](mailto:installation@list.openfuyao.cn) or the openFuyao community mailbox [contact@openfuyao.cn](mailto:contact@openfuyao.cn).
