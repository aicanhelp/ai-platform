# OSCheckTool Tool Usage Instructions

In container scenarios, various workloads have greatly reduced dependencies on the OS runtime environment, but the container platform itself, being responsible for more content, has more dependencies on the OS runtime environment. oschecktool is a single-machine checking tool that checks whether specific runtime environments meet the environmental requirements of the container platform in the openFuyao scenario, to reduce scenarios of deployment failures or functional failures caused by environmental dependencies.

## Motivation

As an open-source ecosystem, there are numerous Linux distributions. Referencing [LinuxTimeLine](https://github.com/FabioLolix/LinuxTimeline), there are currently hundreds of public Linux distributions, and the number of internal custom versions based on certain distributions within enterprises is countless. The numerous Linux distributions bring challenges to applications running on them. To unify the Linux ecosystem, the Linux Foundation initiated the LSB project as early as 2001. However, the LSB project has had no activity since the release of LSB5.0 in 2015. The LSB project has de facto stopped, and various distributions no longer perform LSB certification, but the problem of inconsistent configuration and capabilities across distributions has not disappeared.

On the other hand, container technology was born in 2013. Container images themselves package the entire environment on which containers run, reducing dependencies on the underlying operating system and achieving "build once, run anywhere." However, as the foundation for container operation and orchestration, the CaaS platform itself cannot run completely within containers. Meanwhile, the container platform needs to be responsible for network and storage-related capabilities, and dependencies on system configuration and specific kernel modules cannot be reduced.

As an open ecosystem system, openFuyao may run on various Linux distribution systems in addition to the currently explicitly supported openEuler and Ubuntu systems. Considering the private Linux versions built by various ecosystem partners themselves, how to ensure that these runtime environments can meet the normal operation of openFuyao is a problem that must be considered for the healthy development of the openFuyao ecosystem.

Therefore, this tool is provided to check whether the current runtime environment meets the openFuyao runtime environment requirements.

## Usage Instructions

This tool is a single-machine tool. After obtaining the tool and decompressing it, enter the tool directory and run the tool oschecktool. The interface is as follows:

```sh
./oschecktool [-s check-set] [-t yaml|csv|plain] [-p key=value1,value2,...] [-o report-path] [-h]
    -h Print help information, including selectable check sets and specifiable -p parameters
    -t Specify output file format, supports yaml, csv, plain three forms, where:
        - yaml: Output yaml format report
        - csv: Output csv format report
        - plain: Output plain text format report, default is plain if not specified
    -s Specify check set, actually specifying check content
        - If not specified, default uses all check items in the check-items directory of the tool
        - If specified, only uses the specified check set
    -p Specify parameters, format is key=value1,value2,...
        - Each parameter is actually specified according to check item definition, multiple parameters supported, format is key=value
        - If a parameter has multiple values, use "," to separate
        - If the same parameter is bound multiple times, it will be merged into a list
        - If the specified parameter does not exist, no error will be reported, the parameter will just be ignored
    -o Specify report path, will generate report under this path, if not specified, default creates reports directory in the current directory and generates report under it (plain format report defaults to output to console)
```

Notes:

- Because the checking process may involve multiple directories with high privilege access, this tool needs to run with root privileges.
- In addition to generating reports in specified locations, the tool will create a log directory under the tool directory to store log files. Except for this, it has no other write actions to the system by default (such as creating files, directories, modifying files, etc.), but custom check items need to ensure no impact on the system themselves.
- When the tool runs, the commands called by check items default to running with sh, therefore depending on the sh command. Other commands need to be included according to specific check items. If command execution fails, the corresponding check item is Error. You can view specific error information through logs.

## Check Instructions

Overall, all content that needs to be checked is organized according to check items, and each check item can contain multiple sub-check items.
Check items are located in the tool's conf directory, where:

- The check-items directory contains configuration files for all check items.
- The check-sets directory contains configuration files for all check sets.

### Check Item Organization Structure

The check item organization structure is as follows:

@startuml

object "Check Set" as checkset{
    Name
    Included check set list
    Check item list
}

object "Check Item" as checkitem{
    Name
    Description [Description]
    Runtime parameters and sub-check item information
}

object "Sub Check Item" as subcheckitem{
    Name
    Description [Description]
    Expected value
}

checkset o-- checkset
checkset o-- checkitem
checkitem o-- subcheckitem

@enduml

Check sets themselves support nesting to meet the nested associations of various components within openFuyao.

### Check Item Configuration File Format Description

Archive directory:./conf/check-items/

Format description:

```yaml
name: sysctl-check
kind: kv-checker
doc: |
   Check whether sysctl meets Kubernetes runtime requirements through sysctl -a command
spec:
  ...
```

Where:

- `name`: Check item name, the unique identifier of the check item. Check sets apply check items through the `name` field, and the result will also contain the `name` field.
- `kind`: Check item type, distinguishing different check item types according to the `kind` field.
- `doc`: Check item description, providing detailed description of the check item. The report will also contain the `doc` field. You can explain the principle in detail in this field.
- `spec`: Specific parameters of each check item. Different `kind` check items have different `spec` structures.

Currently supported check item kind fields include:

- `kv-checker`: Parse data from specific sources into simple key-value pairs and check whether specific keys meet expectations, can also be simplified to `kv`.
- `command-checker`: Use commands as sub-check items and directly check command line output results, can also be simplified to `command`.
- `ping-checker`: Use ICMP protocol to check whether target host is reachable, can also be simplified to `ping`.
- `kernel-module-checker`: Check kernel module loading status and loading parameters, can also be simplified to `kernel-module`.

#### command-checker Type Check Item

This check item executes a command and checks output content (stdout, stderr merged). The spec field format definition example is as follows:

```yaml
spec:
  - name: echo command exist  # [Required] Name
    command: which echo && echo "Exist" || echo "Not Exist"
    type: string
    expect: "Exist"
    doc: Check whether {name} command exists, configuration process depends on echo command
```

Description of spec fields:

- `spec[].name`: [Required] Check item name.
- `spec[].command` [Required] Command and parameters to execute, will actually be used as parameters for sh -c.
- `spec[].doc` [Optional] Output to report for description.
- `spec[].type` & `spec[].expect` [Optional], same as kv-parser. If not specified, directly use the return value of the above command for judgment, 0 is success, non-0 is failure. Otherwise, check command output according to expect.

Where, `spec[].type` is the type of the above value, and `spec[].expect` is the expected result. The following types are supported:

- `string`: String type, directly compare whether strings are equal.
- `regex`: Regular expression type, use regular expression to check whether string matches.
- `int`: Integer type, check whether the above output is consistent with the expect numerical value.
- `version-range`: Version type, check whether the above output is consistent with the expect version range. Version numbers are represented in dot-separated decimal form, such as 1.1.1. expect is an interval, such as `(1.2.3, 1.5)`, indicating version number greater than 1.2.3 and less than 1.5, `[1.2.3, ]` indicates version number greater than or equal to 1.2.3.
- `int-range`: Integer range type, check whether the above output is consistent with the expect integer range. expect is an interval, such as `(1, 10]`, indicating integer greater than 1 and less than or equal to 10.

#### kv-checker Type Check Item

This check item parses the output of a command or specific file content into key-value pairs, then checks whether specific keys meet expectations. The spec field format definition example is as follows:

```yaml
spec:
  source: # Specify data source, supports command, file, each line of data is split into key, value by splitter
    file:
      - /etc/sysctl.conf
    command:
      - sysctl -a

  kvParser: # Optional, specify how to parse key and value from source. If not specified, default uses space characters for splitting. If splitting fails, skip this line
    reSplitter: = # Use regular expression to split key, value    
  subItems: # Sub-check items
    - name: kernel.threads-max
      key: kernel.threads-max  # If key is specified, use key to match, otherwise directly use name. Key itself supports regular expressions for wildcard matching
      type: int-range  # Integer value range
      expect: "[409600,)"
      doc: Minimum thread count < 409600, which may cause the system to fail to run normally
```

Where:

- `spec.source.file`: [Optional], specify the file path to read, contents of multiple files will be merged.
- `spec.source.command`: [Optional], specify the output of a command as the subsequent data source, outputs of multiple commands will be merged. If `spec.source.file` is specified, this parameter will be ignored.
- `spec.kvParser`: [Optional], specify how to parse key and value from source. If not specified, default uses space characters for splitting. If splitting fails, skip this line.
- `spec.kvParser.reGroup`: [Optional], use this regular expression to match each line, requiring two or more non-named groups in the regular expression. If there are more than two matching groups, the last group value will be used as value, and other group values will be connected with - as key.
- `spec.kvParser.reSplitter`: [Optional], specify the separator for key and value. The separator is a regular expression. If there are multiple matching values, split by the first matching value. If `spec.kvParser.reGroup` is specified, this parameter is invalid.
- `spec.kvParser.reIgnore`: [Optional], matched lines will be ignored and not parsed.
- `spec.subItems[]`: specific content to be checked. Where, `key` matches with the previously parsed key, and `expect` matches with the previously parsed value. If `key` is empty, use `name` as key. Other fields have the same meaning as fields with the same name in command-checker.

#### ping-checker Type Check Item

This check item uses ICMP protocol to check whether target host is reachable. Default checks 5 times, each timeout is 5 seconds. More than half success is considered success, otherwise failure. Supports IPv4 and IPv6. The `spec` field format example is as follows:

```yaml
spec:
  targets: 
    - 192.168.1.1@hostname
```

Where:

- `spec.targets[]` List of target hosts to check, format is `ip@node name`. If node name is not specified, default uses ip as node name. Node name is mainly for display in the report.

#### kernel-module-checker Type Check Item

This check item parses `/proc/modules` and `/sys/module` directories to check whether specified kernel modules are loaded and whether loading parameters meet expectations. The `spec` field format example is as follows:

```yaml
spec:
  - name: ip_vs
    doc: Check whether ip_vs module is loaded normally
    detail:
      - name: parameters/conn_tab_bits
        expect: "[0, )"
        type: int-range
  - name: not_exist_module
    doc: Test non-existent module, should fail check
    
  - name: nf_conntrack
    doc: Test existing module, but loading parameters do not meet requirements
    detail:
    - name: parameters/expect_hashsize
      expect: "(, 0)"
      type: int-range
```

Where:

- `spec[].name`: [Required], Kernel module name.
- `spec[].doc`: [Optional], Output to report for description.
- `spec[].detail[]`: [Optional], Module parameter check items. Each parameter check item is a key-value pair. Key is the parameter path (relative path value based on `/sys/module/<module name>` as the base path), `expect` is the expected value, `type` is the value type. `expect` and `type` are the same as fields with the same name in other check items. mentioned above.

#### Extended Parameters

To specify some values that cannot be specified in check items, oschecktool supports specifying some extended parameters in check items and binding assignments through -p parameters in commands. The example is as follows:

```yaml
name: ping_check
kind: ping
doc: Simulate checking whether target IP is connectable, specific target is input through -p parameter
params:
  - name: ping-addr
    desc: Check whether target IP is connectable
    multi: true
spec:
  targets: '{{ping-addr}}'
```

Where:

- `params[].name` [Required] Parameter name, valid character range: `A-Za-z0-9_-`.
- `params[].desc` [Optional] Description.
- `params[].multi` [Optional] Whether it is a multi-value parameter, default is false. Multi-value will be set as a string array, single value will be a string.

In `spec`, you can use `{{param.name}}` to reference parameter values, but only support setting a certain yaml field completely to the parameter value.

### Check Set Format Definition

Check sets are used to organize multiple check items together for convenient selection and execution. The check set format definition example is as follows:

```yaml
name: default
desc: default check set
include: 
  - other_check_set
items:
  - sysctl_check
```

Where:

- `name` [Required] Check set name.
- `desc` [Optional] Description.
- `include` [Optional] Include other check sets. The included check sets will be merged into the current check set. References the `name` field of other check sets.
- `items` [Optional] Check item list, referencing the name field of check items.

## Local Build

- When local GO development environment is available, you can directly use the build script for building in a Linux environment, as follows:

  ```sh
  sh <code_path>/.build/build.sh
  # Build result will be generated to <code_path>/output/
  ```

- When local GO development environment is not available, you can use docker or nerdctl for building, as follows:

  ```sh
  docker buildx build . -f   <code-path>/.build/oschecktool.dockerfile \
    -o type=local,dest=<out-path>,platform-split=true \
    --provenance=false \
    --platform=linux/amd64,linux/arm64 \
    --build-arg=GOPRIVATE=gopkg.openfuyao.cn 
  ```

  ```sh
    nerdctl build . -f   <code-path>/.build/oschecktool.dockerfile \
      -o type=local,dest=<out-path>,platform-split=true \
      --provenance=false \
      --platform=linux/amd64,linux/arm64 \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn
  ```
  
  Where:
  - `<out_path>` [Required] Build output path, needs to be replaced with actual output path.
  - `<code-path>` [Required] Project root directory path, needs to be replaced with actual project root directory path.
  - `GOPRIVATE` Configure Go language private repository, equivalent to GOPRIVATE environment variable value, used to specify private repository address.
  - `--platform=linux/amd64,linux/arm64` [Optional] Build target platform. If not specified, default builds current platform. If current container build environment does not support multi-platform building, this parameter needs to be removed.
