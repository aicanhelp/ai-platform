# ologger日志框架

## 特性介绍

本特性旨在设计一个通用化、可扩展的日志框架，以统一支持社区各类组件的日志需求。在提供标准化、结构化的日志基础能力的同时，框架将保持灵活的扩展性，便于各组件根据自身场景进行定制化适配，实现日志功能的规范化管理与个性化输出。

### 应用场景

使用ologger日志框架，完成对开发组件的规范化日志打印。

### 能力范围

**表1** ologger日志框架功能表

|  功能类别   | 功能内容                                                                                                                                                                                                                                          |
|:-------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  日志内容   | <ul><li>结构化输出，分为`json`格式输出和`text`格式输出，支持选择是否输出至控制台和输出日志文件。</li><li>包含日志级别，默认分为CRITICAL、ERROR、WARNING、INFO、DEBUG、TRACE六个级别。</li><li>包含时间，精确到秒，且可以配置时区（utc/local）。</li><li>包含打印日志的文件及行数信息。</li><li>包含当前进程的`pid`和当前主机的`ip`。</li><li>可扩展，可以根据具体记录的需求添加多个字段，以提供更加全面的信息。</li></ul> |
|  日志可轮转  | 可根据日志文件最后修改日期和实际大小进行轮转，同时将旧日志文件进行压缩，减少磁盘空间占用。                                                                                                                                                                                                 |
| 运行时调整级别 | 可以在运行时动态调整日志内容输出级别。                                                                                                                                                                                                                           |
| 用户自定义级别 | 日志框架支持用户自定义级别并将其注册到框架中，同时提供统一入口打印对应自定义级别的日志内容。                                                                                                                                                                                                |
|  日志防注入  | 防止攻击者通过`\n`等控制符破坏日志格式，构造虚假日志内容。                                                                                                                                                                                                               |
| 敏感词过滤机制 | 将敏感词字段对应的值进行模糊处理，同时允许用户自定义敏感词，保障信息安全。                                                                                                                                                                                                         |
|  文件权限   | 配置文件权限为`644`，日志文件权限也为`644`，目录权限为`755`。                                                                                                                                                                                                        |
| 配置项值异常  | 提供`warning`日志信息，并写明回退的缺省值配置情况。                                                                                                                                                                                                                |
| 设置功能开关  | 设置敏感词过滤功能开关`enable_sanitize`，在不需要该功能时及时关闭，提高日志框架性能。                                                                                                                                                                                           |
|  其他事项   | <ul><li>在磁盘满的场景下自动关闭日志写入文件功能，防止机器重启影响业务。</li><li>日志文件删除后，应及时新建对应文件，保证日志框架可用性。</li></ul>                                                                                                                                                                          |

### 亮点特征

- **兼备标准化与可扩展性**：支持符合日志规范的结构化输出、用户自定义日志级别，同时允许用户运行时动态调整日志内容输出级别，满足多样化日志需求。
- **提供安全防护机制**：内置日志防注入与敏感词过滤机制，有效防御日志伪造与信息泄露风险。
- **日志框架高可用**：支持动态轮转与特殊场景实时调整，确保业务连续性。

### 实现原理

**总体流程：**

1. 引用ologger日志库。
2. 编写配置文件。
3. （可选）通过`NewLogger`创建自定义日志对象。
4. 调用框架中日志打印接口。

### 与相关特性的关系

各组件需适配ologger日志框架。


## 使用ologger日志框架

### 前提条件

开发组件引用ologger日志库。

### 背景信息

框架提供符合社区日志规范的日志打印接口及相关各项功能，若组件需适配社区日志规范，可采用本框架进行日志相关开发工作。

### 使用限制

框架仅提供开发过程中日志相关功能，其他规范涉及内容仍需人为遵守。

### 操作步骤

1. 执行如下命令，安装ologger日志库，并在代码中引用。

    ```bash
    go get gopkg.openfuyao.cn/common-modules/ologger@latest
    ```

    ```go
    import "gopkg.openfuyao.cn/common-modules/ologger/log"
    ```

2. 编写配置文件。

    配置文件默认存放于`/etc/openFuyao/ologger/ologger.yaml`中，若需要自定义路径，可以通过修改环境变量`OLOGGER_CONFIG`来进行配置。若配置文件不存在，ologger将使用缺省值和环境变量配置继续运行，并打印`INFO`级别提示日志；若配置文件成功加载，ologger将打印`INFO`级别的`ologger config loaded`日志，并携带`path`字段。

    ```bash
    export OLOGGER_CONFIG="/etc/openFuyao/ologger/config.yaml"
    ```

    配置文件模板如下。

    ```yaml
    # ologger 日志框架配置文件模板，复制此文件并填写实际值
    # 通过环境变量"OLOGGER_CONFIG"刷新日志框架配置项
    
    # ologger日志框架配置项
    # 日志文件存储路径 - 默认为 /var/log/openFuyao/ologger.log
    path: /yourpath/yourfile
    # 日志内容打印最低级别 - 默认为 INFO
    level: INFO
    # 日志输出格式 - json为JSON格式，text/console为文本模式
    format: json
    # 时区 - local/utc
    timezone: local
    # 日志内容是否包含 IP
    include_ip: true
    # 日志轮转 - 单个文件最大大小（MB）
    max_size_mb: max_size_num
    # 日志轮转 - 保留的历史日志文件数量上限
    max_backups: max_backups_num
    # 日志轮转 - 日志文件的最大保留天数
    max_age_days: max_age_num
    # 日志轮转 - 轮转后的旧日志是否压缩
    compress: true
    # 自定义级别
    custom_levels:
      level_name: level_num
    # 自定义清洗敏感词
    sensitive_words:
      - sensitive_word
    
    # ologger日志框架功能开关
    # 日志是否输出至控制台
    enable_console: true
    # 日志是否输出至指定文件
    enable_file: true
    # 是否开启运行时动态调整日志级别，默认关闭，如需开启请设置为 true
    watch_level: false
    # 是否开启敏感词清洗
    enable_sanitize: true
    ```

    上述配置项除自定义级别（custom_levels）外，均可以通过环境变量方式进行配置，列举如下。
    
    ```bash
    export OLOGGER_PATH="/var/log/openFuyao/xxx.log"
    export OLOGGER_LEVEL="INFO"
    export OLOGGER_FORMAT="json"
    export OLOGGER_TZ="utc"
    export OLOGGER_INCLUDE_IP="true"
    export OLOGGER_MAX_SIZE_MB="100"
    export OLOGGER_MAX_BACKUPS="5"
    export OLOGGER_MAX_AGE_DAYS="1"
    export OLOGGER_COMPRESS="true"
    export OLOGGER_SENSITIVE_WORDS="secret,token"
    
    export OLOGGER_CONSOLE="true"
    export OLOGGER_ENABLE_FILE="true"
    # 显式开启运行时动态调整日志级别
    export OLOGGER_WATCH_LEVEL="true"
    export OLOGGER_ENABLE_SANITIZE="true"
    ```

3. （可选）创建自定义日志对象。

    框架在初始化时提供默认全局日志对象，业务代码可直接使用包级API（见[步骤4](#调用日志打印接口)）。若需要独立日志实例（例如不依赖配置文件或环境变量的二进制分发场景），可通过`NewLogger`在代码中显式创建：

    ```go
    logger := log.NewLogger(log.Config{
        Path:   "/var/log/openFuyao/app.log",
        Level:  "INFO",
        Format: "json",
    })
    ```

    `NewLogger`的配置由传入的`Config`结构体控制，未设置的字段回退至库缺省值；不读取`OLOGGER_*`环境变量或配置文件，且不会自动启动`watch_level`级别文件侦听。多数场景可跳过本步骤，直接使用默认全局日志对象。

<a id="调用日志打印接口"></a>
4. 调用日志打印接口。

   框架提供默认全局日志对象与自定义日志对象两种调用方式，接口形态一致。

   - 框架中自定义级别需通过统一入口打印日志。

      默认全局日志对象：

      ```go
      log.Log(levelName string, msg string, args ...any)

      log.Logf(levelName string, template string, args ...any)
      ```

      自定义日志对象（步骤3中创建的`logger`）：

      ```go
      logger.Log(levelName string, msg string, args ...any)

      logger.Logf(levelName string, template string, args ...any)
      ```

   - 框架中默认级别除统一入口外，还可以通过框架中集成的**各级别特有的函数**打印日志内容（此处以`INFO`级别为例）。

      默认全局日志对象：

      ```go
      log.Info(msg string, args ...any)

      log.Infof(template string, args ...any)
      ```

      自定义日志对象：

      ```go
      logger.Info(msg string, args ...any)

      logger.Infof(template string, args ...any)
      ```

      使用示例如下。

      默认全局日志对象：

      ```go
      log.Info("This is msg!", "key", "value")
      ```

      自定义日志对象：

      ```go
      logger.Info("This is msg!", "key", "value")
      ```

### 后续操作

可采用日志分析工具，对格式化日志内容进行分析。

### 相关操作

#### 日志框架功能

- 日志轮转

    日志框架配置轮转参数，并调用`lumberjack`库实现具体功能。

    日志轮转配置项均有**合理配置范围**，具体如下。

    ```yaml
    # 日志轮转-单个文件最大大小（MB）
    max_size_mb: 100-10240
    # 日志轮转-保留的历史日志文件数量上限
    max_backups: 5-30
    # 日志轮转-日志文件的最大保留天数
    max_age_days: 1-14
    # 日志轮转-轮转后的旧日志是否压缩
    compress: true/false
    ```
    
    >![image](figures/icon-note.gif)**说明：**
    >
    >若`compress`设定为`true`，每一个旧日志文件都会被压缩成`.gz`文件，以减少磁盘空间占用。

- 运行时调整级别

    该功能默认关闭。若需要运行时动态调整日志级别，可将`watch_level`配置项设置为`true`，或设置环境变量`OLOGGER_WATCH_LEVEL="true"`，并在运行时通过调整`level`配置项来调整日志输出最低级别。
    
    如果`level`输入错误，默认不改变原配置，并打印`warning`日志；`level`默认配置为`INFO`级别，若首次配置`level`输入错误，`level`也会回退到`INFO`级别。
    
    ```yaml
    # 日志输出最低级别-默认为INFO
    level: INFO
    # 开启运行时动态调整日志级别
    watch_level: true
    ```
    
    同时，框架提供修改日志level的接口，供应用程序调用，接口如下。
    
    ```go
    SetLevel(level string) error
    ```

- 用户自定义级别

    通过`custom_levels`配置项配置级别名称和等级，日志框架将自动完成对这些级别的注册，示例如下。
    
    ```yaml
    # 自定义级别
    custom_levels:
      HINT: 1
      NIL: -9
      Fatal: 16
    ```
    
    框架中默认级别如下，在自定义级别时**不要发生等级冲突**，否则会覆盖原级别。
    
    ```yaml
    TRACE      →     -8
    DEBUG      →     -4
    INFO       →      0
    WARNING    →      4
    ERROR      →      8
    CRITICAL   →      12
    ```

- 日志打印

  - 打印至控制台，通过`enable_console`配置项配置。

      ```yaml
      # 日志是否输出至控制台
      enable_console: true
      ```

  - 打印至文件，通过`enable_file`配置项配置。

      ```yaml
      # 日志是否输出至指定文件
      enable_file: true
      ```
    >  ![image](figures/icon-note.gif)**说明：**
    >
    >  - 若开启打印至文件，但`path`为空，则回退至默认配置，输出至`/var/log/openFuyao/ologger.log`文件中。
    >
    >  - 若`path`存在，则在日志文件不存在时会新建文件，否则自动创建目录和文件。

- 通过With打印日志

    使用`With`添加日志字段，示例如下。
    
    ```go
    log.With("with_key", "with_value").Info("msg")
    ```
    
    输出结果示例如下。
    
    ```json
    {"time":"2026-02-02T19:45:42+08:00","level":"INFO","msg":"msg","source":{"function":"main","file":"main.go","line":48},"pid":136360,"ip":"192.168.105.9","with_key":"with_value"}
    ```

#### 日志框架安全防护

- 日志框架各配置项缺省值

    ```yaml
    # ologger日志框架配置项
    path: /var/log/openFuyao/ologger.log
    level: INFO
    format: json
    timezone: local
    include_ip: true
    max_size_mb: 100
    max_backups: 30
    max_age_days: 14
    compress: true
    custom_levels: {}
    sensitive_words: []
    
    # ologger日志框架功能开关
    enable_console: true
    enable_file: true
    watch_level: false
    enable_sanitize: true
    ```
    
    配置生效级别：配置文件配置 > 环境变量配置 > 缺省值。

- 文件权限

    配置文件权限为`644`，日志文件权限也为`644`，新建日志文件目录权限为`755`。

    >![image](figures/icon-note.gif)**说明：**
    >
    > - 若配置文件`chmod`执行异常，则会打印一条`warning`日志，配置将完全回退至缺省值（或环境变量配置）。
    > 
    > - 若日志文件`chmod`执行异常，错误将被忽略，日志正常写入，不影响程序运行。

- 配置项值异常

    提供`warning`日志信息如下。
    
    ```json
    {"time":"2026-02-03T15:12:03+08:00","level":"WARNING","msg":"ologger config invalid, keeping existing value","source":{"function":"doSlow","file":"C:/Program Files/Go/src/sync/once.go","line":78},"pid":104256,"ip":"192.168.105.9","item":"format","kept":"json"}
    ```

- 敏感词过滤机制

  - 默认过滤常见敏感词`pwd`、`passwd`、`password`，采用正则表达式进行匹配和内容清洗。
  - 提供敏感词自定义配置项`sensitive_words`，支持特性敏感词清洗，会与默认敏感词合并后一同清洗。
  - 设置敏感词过滤功能开关`enable_sanitize`，在不需要该功能时及时关闭，提高日志框架性能。
  - 敏感词清洗大小写不敏感。

    示例如下。
    
    ```json
    {"time":"2026-02-02T19:45:42+08:00","level":"INFO","msg":"the PassWord=****** is correct","source":{"function":"main","file":"main.go","line":48},"pid":136360,"ip":"192.168.105.9","Secret":"******"}
    ```

- 其他注意事项

  - 在磁盘满的场景下自动关闭日志写入文件功能，防止机器重启影响业务。

  - 日志文件删除后，应及时新建对应文件，保证日志框架可用性。

    >  ![image](figures/icon-note.gif)**说明：**
    >
    >  恢复日志文件时，会触发一次日志轮转产生空的旧日志文件压缩包，属于预期行为。

