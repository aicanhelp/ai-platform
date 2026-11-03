# ologger Log Framework

## Feature Introduction

This feature aims to design a universal, extensible log framework to uniformly support the log requirements of various community components. While providing standardized, structured basic log capabilities, the framework maintains flexible extensibility, facilitating custom adaptation by each component based on its own scenarios, achieving standardized management and personalized output of log functionality.

### Application Scenarios

Use the ologger log framework to complete standardized log printing for developed components.

### Capability Scope

**Table 1** ologger Log Framework Feature Table

|  Feature Category   | Feature Content                                                                                                                                                                                                                                          |
|:-------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  Log Content   | <ul><li>Structured output, divided into `json` format output and `text` format output, supports choosing whether to output to console and output log files.</li><li>Contains log level, default divided into CRITICAL, ERROR, WARNING, INFO, DEBUG, TRACE six levels.</li><li>Contains timestamp, accurate to seconds, and timezone can be configured (utc/local).</li><li>Contains the file and line number information of the log print.</li><li>Contains the current process's `pid` and current host's `ip`.</li><li>Extensible, can add multiple fields based on specific recording requirements to provide more comprehensive information.</li></ul> |
|  Log Rotation  | Rotation can be performed based on log file last modification date and actual size, while simultaneously compressing old log files to reduce disk space usage.                                                                                                                                                                                                 |
| Runtime Level Adjustment | Log content output level can be dynamically adjusted at runtime.                                                                                                                                                                                                                           |
| User Custom Level | The log framework supports user custom levels and registers them into the framework, while simultaneously providing a unified entry point to print log content corresponding to custom levels.                                                                                                                                                                                                |
|  Log Injection Prevention  | Prevents attackers from disrupting log format through control characters such as `\n`, constructing false log content.                                                                                                                                                                                                               |
| Sensitive Word Filtering | Blurs the values corresponding to sensitive word fields, while simultaneously allowing users to customize sensitive words to ensure information security.                                                                                                                                                                                                         |
|  File Permissions   | Configuration file permissions are set to `644`, log file permissions are also `644`, and directory permissions are `755`.                                                                                                                                                                                                        |
| Configuration Value Exception  | Provides `warning` log information and specifies the fallback default value configuration.                                                                                                                                                                                                                |
| Feature Switch Settings  | Sets the sensitive word filtering feature switch `enable_sanitize`, promptly disabling when this feature is not needed to improve log framework performance.                                                                                                                                                                                           |
|  Other Notes   | <ul><li>Automatically disabling log file writing functionality when disk is full, preventing machine restart from affecting services.</li><li>After log file deletion, a corresponding file should be promptly created to ensure log framework availability.</li></ul>                                                                                                                                                                          |

### Highlight Features

- **Combining standardization with extensibility**: Supports structured output conforming to log specifications and user custom log levels, while simultaneously allowing runtime dynamic adjustment of log content output level, satisfying diverse log requirements.
- **Security protection mechanism**: Built-in log injection prevention and sensitive word filtering mechanisms, effectively defending against log forgery and information leakage risks.
- **High availability of log framework**: Supports dynamic rotation and real-time adjustment in special scenarios, ensuring business continuity.

### Implementation Principle

**Overall Process:**

1. Import the ologger log library.
2. Write the configuration file.
3. (Optional) Create a custom log object via `NewLogger`.
4. Call the log print interface in the framework.

### Relationship with Related Features

Each component needs to adapt to the ologger log framework.


## Using the ologger Log Framework

### Prerequisites

Developed components import the ologger log library.

### Background Information

The framework provides log print interfaces and related functionalities conforming to community log specifications. If a component needs to adapt to community log specifications, it can use this framework for log-related development work.

### Usage Restrictions

The framework only provides log-related functionalities during development. Other specification content still requires manual compliance.

### Operation Steps

1. Run the following command to install the ologger log library and import it in code.

    ```bash
    go get gopkg.openfuyao.cn/common-modules/ologger@latest
    ```

    ```go
    import "gopkg.openfuyao.cn/common-modules/ologger/log"
    ```

2. Write the configuration file.

    The configuration file is stored by default in `/etc/openFuyao/ologger/ologger.yaml`. If a custom path is needed, it can be configured by modifying the environment variable `OLOGGER_CONFIG`. If the configuration file does not exist, ologger continues with default values and environment variables, and prints an `INFO`-level notice. If the configuration file is loaded successfully, ologger prints an `INFO`-level `ologger config loaded` log with the `path` field.

    ```bash
    export OLOGGER_CONFIG="/etc/openFuyao/ologger/config.yaml"
    ```

    The configuration file template is as follows.

    ```yaml
    # ologger log framework configuration file template, copy this file and fill in actual values
    # Refresh log framework configuration items through environment variable "OLOGGER_CONFIG"
    
    # ologger log framework configuration items
    # Log file storage path - default is /var/log/openFuyao/ologger.log
    path: /yourpath/yourfile
    # Minimum log content print level - default is INFO
    level: INFO
    # Log output format - json for JSON format, text/console for text mode
    format: json
    # Timezone - local/utc
    timezone: local
    # Whether log content includes IP
    include_ip: true
    # Log rotation - maximum size per file (MB)
    max_size_mb: max_size_num
    # Log rotation - maximum number of historical log files retained
    max_backups: max_backups_num
    # Log rotation - maximum retention days for log files
    max_age_days: max_age_num
    # Log rotation - whether to compress rotated old logs
    compress: true
    # Custom levels
    custom_levels:
      level_name: level_num
    # Custom sensitive word filtering
    sensitive_words:
      - sensitive_word
    
    # ologger log framework feature switches
    # Whether to output logs to console
    enable_console: true
    # Whether to output logs to specified file
    enable_file: true
    # Whether to enable runtime dynamic log level adjustment, disabled by default, set to true to enable
    watch_level: false
    # Whether to enable sensitive word filtering
    enable_sanitize: true
    ```

    All configuration items above except custom levels (custom_levels) can be configured through environment variables, listed as follows.
    
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
    # Explicitly enable runtime dynamic log level adjustment
    export OLOGGER_WATCH_LEVEL="true"
    export OLOGGER_ENABLE_SANITIZE="true"
    ```

3. (Optional) Create a custom log object.

    The framework provides a default global log object during initialization. Business code can directly use package-level APIs (see [step 4](#call-the-log-print-interface)). If an independent log instance is needed (e.g., for binary distribution scenarios that don't depend on configuration files or environment variables), you can explicitly create one via `NewLogger` in code:

    ```go
    logger := log.NewLogger(log.Config{
        Path:   "/var/log/openFuyao/app.log",
        Level:  "INFO",
        Format: "json",
    })
    ```

    The configuration of `NewLogger` is controlled by the passed `Config` struct. Unset fields fall back to library defaults; it does not read `OLOGGER_*` environment variables or configuration files, and does not automatically start `watch_level` file monitoring. For most scenarios, you can skip this step and directly use the default global log object.

<a id="call-the-log-print-interface"></a>
4. Call the log print interface.

    The framework provides two calling approaches: the default global log object and the custom log object, with a consistent interface.

   - Custom levels in the framework must be printed through the unified entry point.

      Default global log object:

      ```go
      log.Log(levelName string, msg string, args ...any)
    
      log.Logf(levelName string, template string, args ...any)
      ```

      Custom log object (the `logger` created in step 3):

      ```go
      logger.Log(levelName string, msg string, args ...any)
    
      logger.Logf(levelName string, template string, args ...any)
      ```

   - Default levels in the framework, besides the unified entry point, can also be printed through **level-specific functions** integrated in the framework (here using `INFO` level as an example).

      Default global log object:

      ```go
      log.Info(msg string, args ...any)
    
      log.Infof(template string, args ...any)
      ```

      Custom log object:

      ```go
      logger.Info(msg string, args ...any)
    
      logger.Infof(template string, args ...any)
      ```

      Usage example is as follows.

      Default global log object:

      ```go
      log.Info("This is msg!", "key", "value")
      ```

      Custom log object:

      ```go
      logger.Info("This is msg!", "key", "value")
      ```

### Next Steps

Log analysis tools can be used to analyze formatted log content.

### Related Operations

#### Log Framework Features

- Log Rotation

    The log framework configures rotation parameters and calls the `lumberjack` library to implement specific functionality.

    Log rotation configuration items all have **reasonable configuration ranges**, as follows.

    ```yaml
    # Log rotation - maximum size per file (MB)
    max_size_mb: 100-10240
    # Log rotation - maximum number of historical log files retained
    max_backups: 5-30
    # Log rotation - maximum retention days for log files
    max_age_days: 1-14
    # Log rotation - whether to compress rotated old logs
    compress: true/false
    ```
    
    >![image](figures/icon-note.gif)**Note:**
    >
    >If `compress` is set to `true`, each old log file will be compressed into a `.gz` file to reduce disk space usage.

- Runtime Level Adjustment

    This feature is disabled by default. To dynamically adjust the log level at runtime, set the `watch_level` configuration item to `true` or set the `OLOGGER_WATCH_LEVEL="true"` environment variable, and adjust the `level` configuration item at runtime to change the minimum log output level.
    
    If `level` input is incorrect, the original configuration is not changed by default, and a `warning` log is printed. The default `level` configuration is `INFO`. If the first `level` configuration input is incorrect, `level` will also fall back to `INFO`.
    
    ```yaml
    # Minimum log output level - default is INFO
    level: INFO
    # Enable runtime dynamic log level adjustment
    watch_level: true
    ```
    
    Additionally, the framework provides an interface for modifying the log level for application invocation, as follows.
    
    ```go
    SetLevel(level string) error
    ```

- User Custom Level

    Configure level names and levels through the `custom_levels` configuration item. The log framework will automatically complete registration of these levels. Example is as follows.
    
    ```yaml
    # Custom levels
    custom_levels:
      HINT: 1
      NIL: -9
      Fatal: 16
    ```
    
    Default levels in the framework are as follows. When customizing levels, **do not create level conflicts**, otherwise original levels will be overwritten.
    
    ```yaml
    TRACE      →     -8
    DEBUG      →     -4
    INFO       →      0
    WARNING    →      4
    ERROR      →      8
    CRITICAL   →      12
    ```

- Log Printing

  - Print to console, configured through the `enable_console` configuration item.

      ```yaml
      # Whether to output logs to console
      enable_console: true
      ```

  - Print to file, configured through the `enable_file` configuration item.

      ```yaml
      # Whether to output logs to specified file
      enable_file: true
      ```
    >  ![image](figures/icon-note.gif)**Note:**
    >
    >  - If printing to file is enabled but `path` is empty, it falls back to the default configuration and outputs to the `/var/log/openFuyao/ologger.log` file.
    >
    >  - If `path` exists, a new file will be created when the log file does not exist; otherwise, the directory and file will be automatically created.

- Printing Logs with With

    Use `With` to add log fields. Example is as follows.
    
    ```go
    log.With("with_key", "with_value").Info("msg")
    ```
    
    The output example is as follows.
    
    ```json
    {"time":"2026-02-02T19:45:42+08:00","level":"INFO","msg":"msg","source":{"function":"main","file":"main.go","line":48},"pid":136360,"ip":"192.168.105.9","with_key":"with_value"}
    ```

#### Log Framework Security Protection

- Default Values of Log Framework Configuration Items

    ```yaml
    # ologger log framework configuration items
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
    
    # ologger log framework feature switches
    enable_console: true
    enable_file: true
    watch_level: false
    enable_sanitize: true
    ```
    
    Configuration effective priority: Configuration file configuration > Environment variable configuration > Default values.

- File Permissions

    Configuration file permissions are `644`, log file permissions are also `644`, and newly created log file directory permissions are `755`.

    >![image](figures/icon-note.gif)**Note:**
    >
    > - If the configuration file `chmod` execution encounters an exception, a `warning` log will be printed, and the configuration will completely fall back to default values (or environment variable configuration).
    > 
    > - If the log file `chmod` execution encounters an exception, the error is ignored, logs are written normally, and program running is not affected.

- Configuration Value Exception

    `warning` log information is provided as follows.
    
    ```json
    {"time":"2026-02-03T15:12:03+08:00","level":"WARNING","msg":"ologger config invalid, keeping existing value","source":{"function":"doSlow","file":"C:/Program Files/Go/src/sync/once.go","line":78},"pid":104256,"ip":"192.168.105.9","item":"format","kept":"json"}
    ```

- Sensitive Word Filtering Mechanism

  - Default filtering of common sensitive words `pwd`, `passwd`, `password`, using regular expressions for matching and content cleaning.
  - Provides the sensitive word custom configuration item `sensitive_words`, supporting specific sensitive word cleaning, which will be merged with default sensitive words for combined cleaning.
  - Sets the sensitive word filtering feature switch `enable_sanitize`, promptly disabling when this feature is not needed to improve log framework performance.
  - Sensitive word cleaning is case-insensitive.

    Example is as follows.
    
    ```json
    {"time":"2026-02-02T19:45:42+08:00","level":"INFO","msg":"the PassWord=****** is correct","source":{"function":"main","file":"main.go","line":48},"pid":136360,"ip":"192.168.105.9","Secret":"******"}
    ```

- Other Notes

  - Automatically closes log file writing functionality when disk is full, preventing machine restart from affecting services.

  - After log file deletion, a corresponding file should be promptly created to ensure log framework availability.

    >  ![image](figures/icon-note.gif)**Note:**
    >
    >  When restoring the log file, a log rotation will be triggered, producing an empty old log file compression package, which is expected behavior.

