## openFuyao安全编译选项实施要求(Go)

建议按如下方式使用安全编译选项，提高版本安全质量。

### G.GO.SEC.01 打开栈保护选项

【级别】

要求

【描述】

作用范围：使用了CGO的场景

用法：纯go编译时不涉及；cgo编译时添加CGO_CFLAGS="-fstack-protector-strong“或CGO_CFLAGS="-fstack-protector-all"

说明：栈保护的作用是避免栈溢出被攻击者利用，通过在缓冲区和控制信息间插入一个canary word来实现。当缓冲区被溢出时，在返回地址被覆盖之前canary word会首先被覆盖。通过检查canary word的值是否被修改，就可以判断是否发生了溢出攻击。

（1）buildmode为c-archive、c-shared、plugin时隐含使用CGO，其他buildmode根据实际情况判断是否使用CGO。

（2）GCC4.9版本及以上落地-fstack-protector-strong，GCC4.9版本以下落地-fstack-protector-all，因为strong带来的性能损失要小于-fstack-protector-all,而GCC4.9版本才支持strong

（3）对于性能原因不能落地的需要详细的数据提供给可信构建C-TMG 编译选项工作组审核。

### G.GO.SEC.02 打开地址随机化选项

【级别】

要求

【描述】

#### a. 使用命令 echo 2 >/proc/sys/kernel/randomize_va_space 打开系统随机化配置

作用范围：堆、栈、内存映射区（mmap基址、shared libraries、vdso页）

用法：echo 2 >/proc/sys/kernel/randomize_va_space

ASLR是一种针对缓冲区溢出的安全保护技术，通过对堆、栈、共享库映射等线性区布局的随机化，增加攻击者预测目的地址的难度，防止攻击者直接定位攻击代码位置，达到阻止溢出攻击的目的。randomize_va_space等于1时，栈、数据段、VDSO会随机化，randomize_va_space等于2时堆地址也会随机化。

需要ASLR开启的级别为最高级别，即randomize_va_space等于2

#### b. 动态库、静态库、plugin等文件通过设置相应的buildmode来实现随机加载

作用范围：动态库

用法：buildmode为shared、c-shared、plugin以及c-archive时默认开启

说明：

地址无关选项将发生在代码段的重定位移到数据段实现，so文件加载时代码段不会发生任何变化，做到所有进程共用一个代码段副本。

go语言中完成PIC功能的编译选项是-shared，对于使用go tool compile命令生成的.o文件，编译时添加-shared选项来开启PIC

为了确保.a文件链接成动态库或可执行文件时不出错，go在编译时为buildmode为c-archive的.a文件添加了地址无关代码选项。

#### c. 可执行文件需要打开PIE选项实现随机加载

作用范围：可执行程序

用法：go编译时添加-buildmode=pie；cgo编译时添加CGO_CFLAGS="-fPIE（或-fpie）" CGO_LDFLAGS="-pie"，或者-ldflags "-linkmode 'external' -extldflags '-pie'"

说明：

具备PIE的可执行文件，在加载执行时可像共享库一样随机加载。有研究表明：PIE可有效降低固定地址类攻击、缓冲溢出类攻击的成功概率。

（1）建议使用-buildmode=pie。

（2）关注对应的热补丁版本是否支持PIE选项，不支持的场景下不建议使用该选项

（3）-fPIE编译选项，-pie链接选项

（4）-fPIE产生代码稍大，-fpie产生代码相对较小

### G.GO.SEC.03 打开GOT表重定位只读选项

【级别】

要求

【描述】

#### a.部分重定向只读选项

作用范围：动态库、可执行程序

用法：该选项默认开启。手动开启方式：go编译时添加-ldflags "-linkmode 'external' -extldflags '-Wl,-z,relro'"；cgo编译时添加-ldflags "-linkmode 'external' -extldflags '-Wl,-z,relro'"或者CGO_LDFLAGS="-Wl,-z,relro"

说明：

动态链接的二进制文件默认开启RELRO选项。动态链接的ELF二进制程序使用称为全局偏移表（GOT）的查找表去动态解析位于共享库中的函数。攻击者通过缓冲区溢出修改GOT表项的函数地址值来达到攻击的目的。通过增加RELRO选项，可以防止GOT表被恶意重写。

#### b.立即绑定选项

作用范围：动态库、可执行程序

用法：go编译时添加-ldflags "-linkmode 'external' -extldflags '-Wl,-z,now'"；cgo编译时添加-ldflags "-linkmode 'external' -extldflags '-Wl,-z,now'"或者CGO_LDFLAGS="-Wl,-z,now"

说明：

（1）开启部分重定项只读保护后，再开启立即绑定可实现全部重定向只读保护，即：全部重定向只读(GOT表全保护)

（2）可较好对ret2plt的攻击进行防护，而对诸如缓冲区溢出等攻击无法防范。

（3）对于大量使用共享库中的函数代码产品，在一定程度上会导致程序装载（启动）阶段缓慢，而运行时性能不会有影响。

### G.GO.SEC.04 打开堆栈不可执行选项实现堆栈不可执行保护

【级别】

要求

【描述】

作用范围：动态库、可执行程序

用法：该选项默认开启。手动开启方式：go编译时添加-ldflags "-linkmode 'external' -extldflags '-Wl,-z,noexecstack'"；cgo编译时添加-ldflags "-linkmode 'external' -extldflags '-Wl,-z,noexecstack'"或者CGO_LDFLAGS="-Wl,-z,noexecstack"

说明：缓冲区溢出成功后都是通过执行shellcode来达到攻击的目的，而shellcode一般都是在缓存区里，只要操作系统限制堆栈中的数据只可读写，不可执行，一旦堆栈中的数据被执行立即报告错误并退出，那么溢出成功后也不能执行shellcode，从而达到对应用安全防护的目的。

### G.GO.SEC.05 使用strip工具去除符号表

【级别】

要求

【描述】

作用范围：动态库、可执行程序

用法：strip工具

说明：

符号在链接过程中，发挥着至关重要的作用，链接过程的本质就是把多个不同的目标文件“粘”到一起，符号可看作链接的粘合剂，整个链接过程正是基于符号才正确完成的。链接完成后，符号表对可执行文件运行已经无任何作用，反而会成为攻击者构造攻击的工具，因此删除符号表可防御黑客攻击。事实上删除符号表除防攻击外，还可对文件减肥，降低文件大小。

（1）对于静态库，可重定位文件（.o）不能strip，否则出现编译错误，只涉及ELF可执行文件和动态库交付的产品才可以去除符号表

（2）仅交付给产品，并不直接参与公司外部发布的组件和平台，需提供正式机制通知下游产品在发布阶段统一执行删除符号表操作

（3）因为strip会影响产品定位网上问题和热补丁，构建流程上需要保证strip前后的版本同步，即需要产品本地保留未strip符号表的版本供补丁制作和网上调测使用。如可采用以下方案：

a. 执行机在编译的时候，生成未剥离符号表的可执行文件和动态库的版本,版本归档到VMP上(CMC)供产品做热补丁

b. 使用strip工具对动态库和可执行文件删除符号表

c. 剥离符号的可执行文件和动态库压缩到启动大包

（4）strip工具和-ldflags="-s"选项可达到一样的去除符号表的效果，基于-ldflags="-s"选项会造成版本两次编译构建，建议直接使用strip工具，如strip bin.out。

### G.GO.SEC.06 打开ftrapv选项来检测整数溢出

【级别】

建议

【描述】

作用范围：可重定位文件（.o）、动态库、可执行程序

用法：go语言不涉及该选项；cgo编译时添加CGO_CFLAGS=“-ftrapv”

说明：

（1）buildmode为c-archive、c-shared、plugin时隐含使用CGO，其他buildmode根据实际情况判断是否使用CGO。

（2）使用了-ftrapv选项后，执行带符号的整数间的加、减、乘运算时，不是通过CPU的指令，而是用包含在GCC附属库libgcc.c里的函数来实现。

（3）性能影响较大，建议在Debug版本中实施，Release版本不实施。

### G.GO.SEC.07 启用FORTIFY_SOURCE编译宏来打开FS选项

【级别】

建议

【描述】

作用范围：可重定位文件、动态库、可执行程序

用法：纯go编译时不涉及；cgo编译时添加CGO_CFLAGS=“-D_FORTIFY_SOURCE=2 -O2”

说明：

（1）buildmode为c-archive、c-shared、plugin时隐含使用CGO，其他buildmode根据实际情况判断是否使用CGO。

（2）程序中使用到静态的固定大小的缓冲区，增加了该选项之后，编译器或运行时库会对相关函数的调用在编译时或运行时进行检查。

（3）原则上推荐级别为-O2（基于性能优化效果优于O1）,若产品基于O2的风险性允许使用-O1。

（4）先在分支版本添加，重点做性能测试，根据测试结果取舍。

## 2.2 其他选项

### G.GO.OTH.01 使用-trimpath 将目标二进制中符号的源代码路径信息清除

【级别】

要求

用法：-trimpath

trimpath选项可以删除go生成的二进制文件中所有的源码路径信息，以提高构建可重复性。

### G.GO.OTH.02 混合C/C++语言编译场景，打开C/C++警告选项

【级别】

要求

【描述】具体的警告选项，参见C/C++警告选项条款。例如打开C/C++的"-Wall"选项，检查有用的警告选项集。

### G.GO.OTH.03 显式设置build-id来保证构建结果二进制一致性

【级别】

建议

【描述】

用法：-Wl,--build-id=none / user-defined-value

go默认生成的build-id可能是不一致的，用户可通过显式设置build id为none或者一个用户自定义的值来覆盖go默认生成的build id，避免由build id不同导致的构建生成的二进制文件不一致的问题。

参考：

自定义值user-defined-value参考格式：0xabcdef