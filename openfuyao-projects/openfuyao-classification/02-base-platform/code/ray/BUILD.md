Ray的编译工作是由Bazel工具实现的。Bazel是一个强大的构建和测试工具，适用于大型代码库，并支持多种编程语言。通过Bazel可以一个命令做到依赖下载、编译及WHL打包等工作。

# 1、环境搭建

Ray最低使用GCC10编译。大部分GCC10编译生成的二进制文件都依赖于`libstdc++.so.6.0.28`，较低版本的系统默认并不支持，Ubuntu到20才默认包含这个库文件。而CentOS里的devtoolset10编译生成的二进制文件依赖`libstdc++.so.6.0.19`，支持范围更广。

* Ubuntu20+GCC10：OpenEuler>20.03，Ubuntu>=20。
* CentOS7+devtoolset10：OpenEuler基本全支持，Ubuntu>=14。

此文档介绍CentOS7+devtoolset10的环境搭建，其它系统的大同小异。

**注意：**Ubuntu20+GCC10生成的WHL包也可以在低版本系统里运行，只是需要额外的配置和修改。

## 1.1      容器创建

```
docker pull centos:7
docker run -dit -u root --ipc=host --network=host --name ray_build_env \
-v /home/liuxh:/home/liuxh docker.io/library/centos:7 /bin/bash
```

## 1.2      源配置

```
[base]
name=CentOS-$releasever - Base - mirrors.huaweicloud.com
baseurl=https://mirrors.huaweicloud.com/centos/$releasever/os/$basearch/
#mirrorlist=https://mirrorlist.centos.org/?release=$releasever&arch=$basearch&repo=os
gpgcheck=1
gpgkey=https://mirrors.huaweicloud.com/centos/RPM-GPG-KEY-CentOS-7

#released updates
[updates]
name=CentOS-$releasever - Updates - mirrors.huaweicloud.com
baseurl=https://mirrors.huaweicloud.com/centos/$releasever/updates/$basearch/
#mirrorlist=https://mirrorlist.centos.org/?release=$releasever&arch=$basearch&repo=updates
gpgcheck=1
gpgkey=https://mirrors.huaweicloud.com/centos/RPM-GPG-KEY-CentOS-7

#additional packages that may be useful
[extras]
name=CentOS-$releasever - Extras - mirrors.huaweicloud.com
baseurl=https://mirrors.huaweicloud.com/centos/$releasever/extras/$basearch/
#mirrorlist=https://mirrorlist.centos.org/?release=$releasever&arch=$basearch&repo=extras
gpgcheck=1
gpgkey=https://mirrors.huaweicloud.com/centos/RPM-GPG-KEY-CentOS-7

#additional packages that extend functionality of existing packages
[centosplus]
name=CentOS-$releasever - Plus - mirrors.huaweicloud.com
baseurl=https://mirrors.huaweicloud.com/centos/$releasever/centosplus/$basearch/
#mirrorlist=https://mirrorlist.centos.org/?release=$releasever&arch=$basearch&repo=centosplus
gpgcheck=1
enabled=0
gpgkey=https://mirrors.huaweicloud.com/centos/RPM-GPG-KEY-CentOS-7

```

```
vi  /etc/yum.conf sslverify=false  # 解决Peer's Certificate issuer is not recognized.
yum clean all
yum makecache     # 更新YUM缓存
```

## 1.3 GCC安装

```bash
yum install -y devtoolset-10 devtoolset-10-gcc devtoolset-10-gcc-c++
source /opt/rh/devtoolset-10/enable
```

## 1.4 Java安装

毕昇JDK下载站点：[https://mirrors.huaweicloud.com/kunpeng/archive/compiler/bisheng_jdk/](https://mirrors.huaweicloud.com/kunpeng/archive/compiler/bisheng_jdk/)

```bash
vi  ~/.bashrc
export JAVA_HOME=/usr/local/jdk-17.0.10
```

## 1.5 Bazel安装

Bazel是一款快速、可靠地构建和测试任何规模的软件。Bazel官方提供了带JDK和不带JDK两个版本，由于公司内网环境访问外网不便，需要导入证书，使用外置JDK更方便一些。下载后将二进制文件复制到`/usr/local/bin/`，并重命名为bazel，加上可执行权限，以便后面调用。

Ray 2.30.0之前版本使用Bazel 5.4.1版本。

[https://mirrors.huaweicloud.com/bazel/5.4.1/bazel_nojdk-5.4.1-linux-x86_64](https://mirrors.huaweicloud.com/bazel/5.4.1/bazel_nojdk-5.4.1-linux-arm64)

[https://mirrors.huaweicloud.com/bazel/5.4.1/bazel_nojdk-5.4.1-linux-arm64](https://mirrors.huaweicloud.com/bazel/5.4.1/bazel_nojdk-5.4.1-linux-arm64)

从Ray 2.30.0开始，Ray使用的Bazel版本升级到6.5.0

[https://mirrors.huaweicloud.com/bazel/6.5.0/bazel_nojdk-6.5.0-linux-x86_64](https://mirrors.huaweicloud.com/bazel/6.5.0/bazel_nojdk-6.5.0-linux-x86_64)

[https://mirrors.huaweicloud.com/bazel/6.5.0/bazel_nojdk-6.5.0-linux-arm64](https://mirrors.huaweicloud.com/bazel/6.5.0/bazel_nojdk-6.5.0-linux-arm64)

Ray源码里有很多依赖包，Bazel会在构建时下载，但有些包下载速率并不快。Bazel提供了`--repository_cache`选项可以将依赖包下载到本地制作成缓存，方便下次构建编译，但此选项并不适用于所有依赖，仅限于配置SHA256的依赖文件。Ray源码里，rules_perl使用git下载，go_sdk的链接没有配置SHA256，这两个文件不能缓存。

将`--repository_cache`选项加入`.bazelrc`配置文件中。

```
build --repository_cache=/tmp/bazel/cache
```

此外，在测试过程中有时候会切换分支，在新分支构建后切回来还需要重新构建，这会浪费很多资源和时间。Bazel考虑到这种情况提供了缓存方案，同一条件下编译生成的结果也会一样的，Bazel会将结果保存成缓存，下次相同条件编译到相同位置时可以直接调用缓存结果。Bazel提供了两种缓存方式，本地缓存和远程缓存，下面是本地缓存的使用方式。

```
build --disk_cache=/tmp/bazel/cache
```

## 1.6 NPM工具安装

Ray的dashboard编译使用Node.js的14版本。下载解压后，将其加入环境变量。

[https://nodejs.org/dist/v14.21.1/node-v14.21.1-linux-arm64.tar.xz](https://nodejs.org/dist/v14.21.1/node-v14.21.1-linux-arm64.tar.xz)

[https://nodejs.org/download/release/v14.21.3/node-v14.21.3-linux-arm64.tar.gz](https://nodejs.org/download/release/v14.21.3/node-v14.21.3-linux-arm64.tar.gz)

## 1.7 Python安装

### 1.7.1 源码安装

```bash
yum install zlib-devel bzip2-devel openssl-devel ncurses-devel sqlite-devel readline-devel tk-devel libffi-devel xz-devel
./configure --prefix=/usr/local/python3.8 --enable-shared --with-openssl=/usr  --enable-optimizations
make -j180
make install

# 在.bashrc里添加环境变量
export PATH=/usr/local/python3.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/python3.8/lib:$LD_LIBRARY_PATH

pip install --upgrade pip
pip install psutil numpy auditwheel patchelf setuptools==67.8.0 wheel
```

`--enable-optimizations`选项会启动优化，比较耗时，可能会影响性能，可以不加。
Python 3.10版本最低需要openssl-1.1.1版本，CentOS7的默认源里没有，需要下载源码编译，然后用--with-openssl指定安装位置，并添加`--with-openssl-rpath=auto`。

export LDFLAGS="-Wl,-rpath,/usr/local/openssl/3.0.13/lib"

### 1.7.2 二进制安装

网络上有已经编译好的二进制包，我们可以下载解压即可使用。

[indygreg/python-build-standalone: Produce redistributable builds of Python](https://github.com/indygreg/python-build-standalone)


## 1.8 Git工具安装

```
yum install git

# 在.bashrc里添加环境变量，禁用SSL
export GIT_SSL_NO_VERIFY=false
```

# 2、Ray源码编译

## 2.1 编译dashboard

```bash
cd python/ray/dashboard/client
npm ci
npm run build
```

## 2.2 WHL打包

此过程会编译Ray源码，并打包。

```
cd python
python setup.py bdist_wheel
```

**注意：**为了使WHL包适用更多系统，一般会使用auditwheel工具处理WHL包。auditwheel工具会把一些依赖文件复制到WHL包内，但经过分析发现Ray是静态编译的，除了标准库外并不依赖其它动态库，所以这一步就省略了。

## 2.3 代码编译

在ray代码目录下执行下面的命令就可以开始源码编译过程，这个过程只会编译，不会打包。

```shell
bazel build --compilation_mode=opt -c fastbuild //:ray_pkg
```

## 2.4 Java编译

Ray打包时默认是不包含Java的，若是想要构建Java代码，需要配置环境变量`RAY_INSTALL_JAVA=1`。

```
apt install unzip
```

```
RAY_COMMIT_SHA=$(git rev-parse HEAD)
sed -i "s/{{RAY_COMMIT_SHA}}/${RAY_COMMIT_SHA}/g" ray/_version.py
```
