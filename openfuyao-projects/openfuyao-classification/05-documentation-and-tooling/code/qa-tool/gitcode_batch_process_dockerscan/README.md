### gitcode_batch_process_dockerscan

- 工具介绍：镜像批量拉取、打包、打镜像和上传到指定仓库的脚本（batch_push_images.sh）；浏览器界面自动化创建Docker安全扫描任务的脚本（main.py）

## 使用方法

##### 步骤1：需要处理的镜像

根据images.txt文件的格式整理需要处理的镜像

##### 步骤2：工具配置

```
# batch_push_images.sh
前期准备：openFuyao系统、bke工具、Docker工具
配置参数：
"SOURCE_REGISTRY"： 源镜像仓库（如果镜像名已包含仓库，可留空）
"TARGET_REGISTRY": 目标镜像仓库
"TARGET_USER"： 目标仓库用户名
"TARGET_PASSWORD"： 目标仓库密码（可留空，会提示输入或从环境变量获取）
"IMAGES_LIST": 需要上传的镜像列表文件

# mian.py
前期准备：浏览器驱动（edge driver）、自动化框架（selenium）
配置参数：
"HARBOR_BASE_URL"： 镜像仓地址 
"HARBOR_USER"： 镜像仓用户名              
"HARBOR_PWD"： 镜像仓密码            
"WEB_URL"： 浏览器url
"WEB_USER"： 浏览器登录用户名，无需登录则留空
"WEB_PWD"： 浏览器登录密码，无需登录则留空
"TARGET_REGION"： 目标区域
"SCAN_ENGINES"： 扫描引擎
"NEED_TEST_CONNECTION"： 是否勾选连接测试
"IMAGES_LIST"： 镜像列表路径
"EDGE_DRIVER_PATH"： 驱动路径
```

##### 步骤3：工具运行

执行 `bash batch_push_images.sh`即可对images.txt中的镜像进行批量拉取、打包、打镜像，并且上传到指定仓库的脚本；

执行 `python3 main.py` 即可在浏览器界面自动化执行批量创建Docker安全扫描任务（代码需要根据自己处理的浏览器界面进行修改！！！）