# 代码仓库目标布局

## go代码仓库布局

### 代码相关部分

```text
base_dir
 ├──cmd //main函数的目录，如cmd/{component_name}/xxx.go,go文件内部有main函数
 │  └── app1
 │     └── main.go
 │  └── app2
 │      └── main.go
 ├──conf //配置文件目录，存储程序运行时所需要的配置文件
 ├──pkg //可选，公共基础库目录，可复用且外部工程也可能会使用
 │  └──app_name //模块名
 │     ├──XXX.go //源代码文件
 │     └──XXX_test.go //单元测试代码
 ├──internal //内部调用API函数目录，不希望被外部工程调用
 ├──test //测试数据和系统测试文件目录，mock用的数据等，不要放单元测试代码
 ├──deployments //部署用的编排模板文件目录，如terraform,aos,k8s模板和helm
 ├──build //构建
 │   ├──package //打包脚本目录,用于打包docker,rpm,deb,zip,tar.gz
 │   └──ci //ci脚本、ut和静态检查文件目录
 ├──scripts //其余脚本，可选
 ├──examples //例子代码，可选
 ├──assets //工程资源，如图片，可选
 ├──scripts //其余脚本，可选
 ├──plugins //可扩展插件，可选
 ├──generated //代码生成器生成的代码，非默认，可选
 ├──vendor //go第三方依赖包存放路径，可选
 └──output //工程输出目录
```

### 说明部分

```text
├── readme.md
├── LICENSE
├── code-of-conduct.md
├── docs
│   ├── contributors # 贡献文档
│   ├── design       # 设计文档
│   ├── dev          # 开发文档
│   ├── user         # 用户文档
└── .gitignore
```

### 构建相关

```text
demo
├── Makefile
├── Dockerfile
├── .ci-operator.yaml
├── bundles # operator打包
├── manifests # yaml清单
└── images
```

### 其它

```text
demo
├── scripts
├── hack
```

## javascript代码仓库布局

```text
base_dir
 ├──package.json //工程配置文件
 ├──scripts //构建脚本
    ├──build.suffix //可选，构建入口文件，suffix为对应脚本后缀
 ├──scr //源码目录
    ├──i18n //国际化
    ├──environments //环境变量
    ├──theme //主题
 ├──app //页面目录
    ├──style.less //约定的全局样式文件，自动引入
    ├──main.ts //入口ts文件
    ├──index.html //入口html文件
 ├──dist //目标文件目录
 ├──test //测试文件目录
```

## ref

- [ascend-device-plugin](https://gitee.com/ascend/ascend-device-plugin)
- [openshift console-operator](https://github.com/openshift/console-operator)
