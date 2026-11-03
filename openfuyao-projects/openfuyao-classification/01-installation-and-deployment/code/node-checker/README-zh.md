# OSCheckTool工具使用说明

容器场景下，各种负载对OS运行环境的依赖得到很大减少，但容器平台自身反因负责内容更多而对OS运行环境有了更多依赖，oschecktool为一个单机检查工具，在openFuyao场景下检查特定运行环境是否满足容器平台的环境诉求，以减少各种因环境依赖造成的部署失败或功能失效的场景。

## 动机

作为一个开源的生态，Linux发行版数量众多，参考[LinuxTimeLine](https://github.com/FabioLolix/LinuxTimeline)，当前已有数百个公开的Linux发行版，而企业内部基于某发行版定制的内部使用版本更是不计其数，数量众多的Linux发行版给运行于其上的应用带来了挑战，为了统一Linux的生态，早在2001年，Linux基金会就发起了LSB项目，而LSB项目自2015年LSB5.0发布后已再无活动，LSB项目已事实停止，各发行版也不再进行LSB认证，但各发行版配置和能力不一致的问题并未消失。
另一方面2013年容器技术的诞生，容器镜像自身打包了整个容器运行依赖的环境，降低了对下层操作系统的依赖，实现“build once，run anywhere”。而CaaS平台本身作为容器运行和编排的基础，自身无法完全运行于容器中，同时容器平台需要负责网络和存储相关能力，对系统的配置和特定内核模块的依赖也无法消减。
openFuyao作为一个开放生态的系统，除当前已明确支持的openEuler、Ubuntu系统外，也可能会运行在其他各种Linux发行版系统上，再考虑到各生态伙伴自己构建的私有Linux版本，如何保证这些运行环境能够满足openFuyao的正常运行，是openFuyao生态健康发展必须要考虑的问题。

因此提供此工具，用于检查当前的运行环境是否满足openFuyao的运行环境诉求。

## 使用说明

本工具为单机工具，在获取工具并将工具解压后，进入工具目录，运行工具oschecktool，接口如下:

```sh
./oschecktool [-s chec-set] [-t yaml|csv|plain] [-p key=value1,value2,...] [-o report-path] [-h]
    -h 打印帮助信息，包括可以选择的检查集和可以指定的-p参数
    -t 指定输出文件格式, 支持yaml,csv, plain三种形式，其中:
        - yaml: 输出yaml格式的报告
        - csv: 输出csv格式的报告
        - plain: 输出纯文本格式的报告, 不指定则默认为plain
    -s 指定检查集，实际就是指定检查内容
        - 如果不指定，则默认使用工具check-items目录下的所有检查项
        - 如果指定，则只使用指定的检查集
    -p 指定参数，格式为key=value1,value2,...
        - 各参数实际按照检查项定义而指定，多个参数支持，格式为key=value
        - 如果参数有多个值，则使用","分割
        - 如果同名参数多次绑定，则会被合并到一个列表中
        - 如果指定的参数不存在，则也不会报错，只是会忽略该参数
    -o 指定report路径，将在该路径下生成报告，不指定则默认在在前目录下创建reports目录并在其下生成报告(plain格式的报告默认输出到控制台)
```

注意事项：
  
- 因为检查过程可能涉及多个高权限访问的目录，本工具需要以root权限运行。
- 工具除在指定的位置生成报告外，还会在工具目录下创建log目录存放日志文件，除此外默认对系统无其他写动作（如创建文件、目录、修改文件等），但自定义的检查项需要自行保证对系统无影响。
- 工具运行时调用检查项的命令默认使用sh运行，因此依赖sh命令，其他命令需要根据具体的检查项来包含，如果命令运行失败，则对应的检查项为Error，可以通过日志查看具体的错误信息。

## 检查说明

整体上，所有需要检查的内容按照检查项的方式进行组织，每个检查项可以包含多个子检查项。
检查项位于工具conf目录下，其中：

- check-items目录下为所有检查项的配置文件。
- check-sets目录下为所有检查集的配置文件。

### 检查项组织结构

检查项组织结构如下：

@startuml

object "检查集" as checkset{
    名称
    包含的检查集清单
    检查项列表
}

object "检查项" as checkitem{
    名称
    说明[说明]
    运行参数和子检查项信息
}

object "子检查项" as subcheckitem{
    名称
    说明[说明]
    期望值
}

checkset o-- checkset
checkset o-- checkitem
checkitem o-- subcheckitem

@enduml

检查集本身支持嵌套，以满足openFuyao内各组件的嵌套关联。

### 检查项配置文件格式说明

归档目录：./conf/check-items/

格式说明：

```yaml
name: sysctl-check
kind: kv-checker
doc: |
   通过sysctl -a命令检查sysctl是否符合kubernetes运行要求
spec:
  ...
```

其中：

- `name`：检查项名称，检查项的唯一标识，检查集通过`name`字段应用检查项，同时结果中也会包含`name`字段。
- `kind`：检查项类型，根据`kind`字段区分不同的检查项类型。
- `doc`：检查项说明，对检查项进行详细说明，报告中也会包含`doc`字段，可在这个字段中详细说明原理。
- `spec`：各检查项的具体参数，不同`kind`的检查项有不同的`spec`结构。

当前支持的检查项kind字段包括：

- `kv-checker`：将特定来源的数据解析为简单的key-value对，并对特定的key检查是否符合预期，也可简化为`kv`。
- `command-checker`：使用命令作为子检查项，并直接检查命令的命令行输出结果，也可以简化为`command`。
- `ping-checker`：使用icmp协议检查目标主机是否可达，也可以简化为`ping`。
- `kernel-module-checker`：检查内核模块的加载情况和加载参数，也可以为简化为`kernel-module`。

#### command-checker类型检查项

此检查项执行一个命令并检查输出内容（stdout、stderr合并），spec字段格式定义样例如下：

```yaml
spec:
  - name: echo command exist  #【必选】 名称
    command: which echo && echo "Exist" || echo "Not Exist"
    type: string
    expect: "Exist"
    doc: 检查{name}命令是否存在，配置过程依赖echo命令  
```

spec各字段说明：

- `spec[].name`：【必选】，检查项名称。
- `spec[].command`：【必选】，执行的命令和参数，实际会作为sh -c的参数。
- `spec[].doc`：【可选】，输出到报告中用作说明。
- `spec[].type` & `spec[].expect`：【可选】，同kv-parser, 如果不指定，则直接使用上述命令的返回值作为判断，0为成功，非0失败。否则按照expect检查命令输出。

其中，`spec[].type`为上面值的类型，`spec[].expect`为期望结果，支持如下类型：

- `string`：字符串类型，直接比较字符串是否相等。
- `regex`：正则类型，使用正则表达式检查字符串是否匹配。
- `int`：整数类型，检查上述输出与expect的数值是否一致。
- `version-range`：版本类型，检查上述输出与expect的版本范围是否一致，版本号使用点分数值形式表示，如1.1.1，expect的为一个区间，如`(1.2.3, 1.5)`，表示版本号大于1.2.3小于1.5，`[1.2.3, ]`表示版本号大于等于1.2.3。
- `int-range`：整数范围类型，检查上述输出与expect的整数范围是否一致，expect的为一个区间，如`(1, 10]`，表示整数大于1小于等于10。

#### kv-checker类型检查项

此检查项将命令的输出或特定文件内容解析为key-value对，然后检查特定的key是否符合预期，spec字段格式定义样例如下：

```yaml
spec:
  source:  #指定数据源，支持command、file，其中的每行数据都被splitter分割为key、value
    file:
      - /etc/sysctl.conf
    command:
      - sysctl -a

  kvParser: #可选，指定如何从source中解析出key和value，如果不指定，则默认使用空字符进行拆分。如果拆分呢失败，则跳过该行
    reSplitter: = # 使用正则表达式拆分key、value    
  subItems: #子检查项
    - name: kernel.threads-max
      key: kernel.threads-max  #如果指定了key，则使用key匹配， 否则直接使用name，key本身支持正则表达式进行通配
      type: int-range  #整型值范围
      expect: "[409600,)"
      doc: 最下线程数<409600，可能导致系统无法正常运行
```

其中：

- `spec.source.file`：【可选】，指定要读取的文件路径，多个文件的内容将会被合并。
- `spec.source.command`：【可选】，指定命令的输出作为后续的数据源，多个命令的输出将会被合并，如果指定了`spec.source.file`，此参数将被忽略。
- `spec.kvParser`：【可选】，指定如何从source中解析出key和value，如果不指定，则默认使用空字符进行拆分。如果拆分呢失败，则跳过该行。
- `spec.kvParser.reGroup`：【可选】，使用此正则表达式来匹配每一行，要求正则表达式中有两个或以上的非命名分组，如果有两个以上的匹配分组，则将最后一个分组值作为value，其他分组值使用-连接作为key。
- `spec.kvParser.reSplitter`：【可选】，指定key和value的分隔符，分割符作为正则表达式。如果有多个匹配值则以第一个匹配值分割，如果指定了`spec.kvParser.reGroup`，则此参数无效。
- `spec.kvParser.reIgnore`：【可选】，匹配的行将被忽略，不进行解析。
- `spec.subItems[]`：具体的待检查内容，其中，`key`与前述解析的key进行匹配，`expect`与前述解析的value进行匹配，如果`key`为空，则使用`name`作为key，其他字段与command-checker同名字段含义相同。

#### ping-checker类型检查项

此检查项使用ICMP协议检查目标主机是否可达，默认检查5次，每次超时5秒，超过半数成功则为成功，否则为失败，支持IPV4和IPV6，`spec`字段格式样例如下：

```yaml
spec:
  targets: 
    - 192.168.1.1@hostname
```

其中:

- `spec.targets[]`：检查的目标主机列表，格式为`ip@节点名`，如果不指定节点名，则默认使用ip作为节点名，节点名主要是为了在报告中显示。

#### kernel-module-checker类型检查项

此检查项解析`/proc/modules`和`/sys/module`目录，检查指定的内核模块是否加载，以及加载参数是否符合预期。`spec`字段格式样例如下：

```yaml
spec:
  - name: ip_vs
    doc: 检查ip_vs模块是否正常加载
    detail:
      - name: parameters/conn_tab_bits
        expect: "[0, )"
        type: int-range
  - name: not_exist_module
    doc: 测试不存在的模块，应该检查不通过
    
  - name: nf_conntrack
    doc: 测试存在的模块，但加载参数不符合要求
    detail:
    - name: parameters/expect_hashsize
      expect: "(, 0)"
      type: int-range
```

其中：

- `spec[].name`【必选】，内核模块名。
- `spec[].doc`【可选】，输出到报告中用作说明。
- `spec[].detail[]`【可选】，模块参数检查项，每个参数检查项为一个key-value对，key为参数路径（以`/sys/module/<模块名>`为基础路径的相对路径值）`expect`为期望的值，`type`为值的类型，`expect`、`type`与前述其他检查项同名字段相同。

#### 扩展参数

为了指定一些无法在检查项中指定指定值，oschecktool支持在检查项中指定一些扩展参数，并在命令中通过`-p`参数进行绑定赋值。样例如下：

```yaml
name: ping_check
kind: ping
doc: 模拟检测目标Ip是否可以联通, 具体目标通过-p 参数输入
params:
  - name: ping-addr
    desc: 检查目标Ip是否可以联通
    multi: true
spec:
  targets: '{{ping-addr}}'

```

其中：

- `params[].name`【必选】参数名，合法字符范围：`A-Za-z0-9_-`。
- `params[].desc`【可选】说明。
- `params[].multi`【可选】是否为多值参数，默认false，多值将会设置为字符串数组，单值则为字符串。

`spec`中可以使用`{{param.name}}`来引用参数值，但只支持将某个yaml字段完整设置为参数值。

### 检查集格式定义

检查集用于将多个检查项组织起来，方便选择执行。检查集格式定义样例如下：

```yaml
name: default
desc: default check set
include: 
  - other_check_set
items:
  - sysctl_check
```

其中:

- `name`【必选】检查集名。
- `desc`【可选】说明。
- `include`【可选】引入其他检查集，引入的检查集将合并到当前检查集中，引用的是其他检查集的`name`字段。
- `items`【可选】检查项列表，引用的是检查项的name字段。

## 本地构建

- 本地具备GO开发环境时，可直接在linux环境中使用构建脚本进行构建，如下：

  ```sh
  sh <code_path>/.build/build.sh
  #构建结果结果将生成到<code_path>/output/下
  ```

- 本地不具备GO开发环境，可以使用docker或nerdctl进行构建，如下：

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
  
  其中：
  - `<out_path>`【必选】构建输出路径，需替换为实际的输出路径。
  - `<code-path>`【必选】项目根目录路径，需替换为实际的项目根目录路径。
  - `GOPRIVATE` 配置Go语言私有仓库，相当于GOPRIVATE环境变量值，用于指定私有仓库的地址。
  - `--platform=linux/amd64,linux/arm64`【可选】构建目标平台，如果不指定则默认构建当前平台，如果当前容器构建环境不支持多平台构建，则需去掉此参数。
