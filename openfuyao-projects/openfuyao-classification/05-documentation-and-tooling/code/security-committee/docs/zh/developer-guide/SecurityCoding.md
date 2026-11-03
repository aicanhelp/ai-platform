# openFuyao安全编程指南

本文档基于Go语言提供一些安全编程建议，用于指导开发实践。

## 1. 规则名称：确保使用完毕的锁被正确释放
**问题级别**：严重

### 正确示例：
##### 场景1：加锁后未释放
- 修复示例：加锁后正确解锁

```go
var m sync.Mutex

func main() {
    var x sync.Mutex
    x.Lock()
    ...
    x.Unlock()  // 符合：使用完解锁
}
```

### 错误示例：
##### 场景1：加锁后未释放
- 错误示例：加锁后未释放

```go
var m sync.Mutex

func main() {
    var x sync.Mutex
    x.Lock()  // 加锁后，一直未解锁
    ...
}
```

### 修改建议：
使用完锁后应及时释放锁。
在错误或异常条件下，程序可能会提前退出，退出前要确保正确释放当前持有的锁。

---

## 2. 规则名称：需要字符转义时应优先使用raw string
**问题级别**：一般

### 正确示例：
##### 场景1：字符串存在转义未使用raw string
- 修复示例：

  ```go
  const prefix = `package p; import "unsafe"; type P *struct{}; type I interface{}; type T ` // 符合：使用了Raw字符串去掉了反斜杠，字符串更容易阅读
  ```

### 错误示例：
##### 场景1：字符串存在转义未使用raw string
- 错误示例：

  ```go
  const prefix = "package p; import \\"unsafe\\"; type P *struct{}; type I interface{}; type T " // 不符合：这是一句用于解析go程序的字符串定义
  ```

### 修改建议：
在字符串包含字符转义时，可直接使用raw string。

---

## 3. 规则名称：正式交付给客户的代码不应包含TODO/TBD/FIXME注释
**问题级别**：提示

### 正确示例：
##### 场景1：正式交付给客户的代码中未解决TODO问题
- 修复示例：把待处理问题解决后删除注释。

  ```go
  // 符合
  func accum(fee uint) {
      doSomething()
  }
  ```

### 错误示例：
##### 场景1：正式交付给客户的代码中未解决TODO问题
- 错误示例：

```go
// 不符合：未解决TODO问题
func accum(fee uint) {
    // TODO: ...
    total += fee
}
```

### 修改建议：
合理规划产品功能，尽量通过其他功能记录而不是在代码中记录。

---

## 4. 规则名称：注释符与注释内容间要有1个空格
**问题级别**：提示

### 正确示例：
##### 场景1：注释符与注释内容之间无空格
- 修复示例1：在注释符与注释之间加空格

```go
// 符合：注释符和注释之间有一个空格

// 单行注释

/* 单行注释 */
```

### 错误示例：
##### 场景1：注释符与注释内容之间无空格
- 错误示例：

```go
//不符合：注释符和注释之间无空格

//单行注释

/*单行注释*/
```

### 修改建议：
保持注释符与注释内容之间留有1个空格的格式。

---

## 5. 规则名称：错误信息不应大写或以标点符号结尾
**问题级别**：提示

### 正确示例：
##### 场景1：出现字符串中间有大写字母开头的单词
- 修复示例：

  ```go
  fmt.Errorf("something bad") // 符合，该字符串和上下文拼接后，不会出现违反语法的情况.
  ```
##### 场景2：字符串以标点符号结尾
- 修复示例：

  ```go
  fmt.Errorf("something bad") // 符合
  ```

### 错误示例：
##### 场景1：出现字符串中间有大写字母开头的单词
- 错误示例：

  ```go
  fmt.Errorf("Something bad") // 不符合，和上下文拼接后，出现字符串中间有大写字母开头的单词的情形
  ```
##### 场景2：字符串以标点符号结尾
- 错误示例：

  ```go
  fmt.Errorf("something bad.") // 不符合，字符串以标点符号结尾
  ```

### 修改建议：
避免错误字符串以大写字母开头，或以标点符号结尾。

---

## 6. 规则名称：无LICENSE文件
**问题级别**：致命

---

## 7. 规则名称：不要在迭代集合数据结构的过程中添加或删除元素
**问题级别**：严重

### 正确示例：
##### 场景1：迭代slice时添加了元素。
- 修复示例：迭代时不添加或删除元素。

```go
func main() {
    var a []int = []int{0, 1}
    for i := 0; i < len(a); i++ {
        ... // 符合：不添加或删除元素
    }
    n := len(a)
    for j := 0; j < n; j++ {
        a = append(a, 1, 2, 3)
    }
    ...
}
```

### 错误示例：
##### 场景1：迭代slice时添加了元素。
- 错误示例：迭代slice时添加了元素。

```go
func main() {
    var a []int = []int{0, 1}
    for i := 0; i < len(a); i++ {
        a = append(a, 1, 2, 3) // 不符合：在迭代slice过程中添加了元素
        ...
    }
    ...
}
```

### 修改建议：
不要在迭代slice的过程中添加或删除元素。

---

## 8. 规则名称：相关"if/else if"语句不应有相同的条件
**问题级别**：一般

### 正确示例：
```GO

func example(condition1, condition2 bool) {
  if condition1 {
  } else if condition2 { // 符合
  }
}

```

```GO

func SwitchWithMultipleConditions(param int) {
  switch param {
  case 1, 2, 3:
    fmt.Println(">1")
  case 4, 5: // 符合
    fmt.Println("<1")
  }
}

```

### 错误示例：
```GO

func example(condition1, condition2 bool) {
  if condition1 {
  } else if condition1 { // 不符合
  }
}

```

```GO

func SwitchWithMultipleConditions(param int) {
  switch param {
  case 1, 2, 3:
    fmt.Println(">1")
  case 3, 4, 5: // 不符合；3重复
    fmt.Println("<1")
  }
}

```

---

## 9. 规则名称：慎用goto语句
**问题级别**：一般

### 正确示例：
##### 场景1：使用了goto语句。
- 修复示例：非必须场景，不应使用goto语句，可使用结构化的控制语句（如if、for、while等）来实现程序逻辑。

```go
for m := 1; m < 5; m++ { // 符合
    fmt.Println(m)
}
```

### 错误示例：
##### 场景1：使用了goto语句。
- 错误示例：

```go
    m := 1
LOOP:
    if m < 5 {
        fmt.Println(m)
        m++
        goto LOOP // 不符合：使用了goto语句
    }
```

### 修改建议：
非确实需要不应使用goto语句。

---

## 10. 规则名称：不用的代码段直接删除，不要注释掉
**问题级别**：一般

### 正确示例：
##### 场景1：不用的代码未删除，而是注释了
- 修复示例：删除不用的代码

  ```go
  func TestUnaryOperation() { // 符合
  }
  ```

### 错误示例：
##### 场景1：不用的代码未删除，而是注释了
- 错误示例：

  ```go
  func TestUnaryOperation() {
      // a++
      // b--
  }
  ```

### 修改建议：
不用的代码建议直接删除，不要注释。

---

## 11. 规则名称：无README.OpenSource
**问题级别**：严重

---

## 12. 规则名称：goto语句只能向下跳转
**问题级别**：一般

### 正确示例：
##### 场景1：特殊场景下使用向上跳转的goto语句。
- 修复示例：特殊场景下使用goto语句，goto语句不要向上跳转，只能向下跳转。

```go
    // ...

    goto label // 符合：向下跳转
    // ...

label:
    // something
```

### 错误示例：
##### 场景1：特殊场景下使用向上跳转的goto语句。
- 错误示例：

```go
    // ...

label:
    // something

    // ...
    goto label // 不符合：向上跳转
```

### 修改建议：
特殊场景下使用了goto语句只能向下跳转。

---

## 13. 规则名称：避免调用os.Exit()和runtime.Goexit()退出函数
**问题级别**：提示

### 正确示例：
##### 场景1：调用`os.Exit()`函数退出。
- 修复示例：

```go
func CreateFile() error {
    f, err := ioutil.TempFile("/opt/pwm", "prefix")
    if err != nil {
        fmt.Println("Create temp file failure.")
        return err
    }
    defer closeAndRemove(f)
    _, err2 := f.Write([]byte("This is a example."))
    if err2 != nil {
        return err2  // 符合
    }
    return nil
}

func closeAndRemove(f *os.File) {
    f.Close()
    os.Remove(f.Name())
}
```
##### 场景2：调用`runtime.Goexit()`退出`goroutine`。
- 修复示例：

```go
func foo() error {
    ...
}

func bar() {
    go func() {
        defer func() {
            fmt.Println("defer!")
            fmt.Println("recovered = ",recover())
        }()

        ...
        err = foo()
        if err != nil {
           return // 符合
        }
    }()

    fmt.Println("bar")
}
```

### 错误示例：
##### 场景1：调用`os.Exit()`函数退出。
- 错误示例：

```go
func CreateFile() {
    f, err := ioutil.TempFile("/opt/pwm", "prefix")
    if err != nil {
        fmt.Println("Create temp file failure.")
        return
    }
    defer closeAndRemove(f)
    _, err2 := f.Write([]byte("This is a example."))
    if err2 != nil {
        os.Exit(0) // 不符合：会导致临时文件无法清除
    }
}

func closeAndRemove(f *os.File) {
    f.Close()
    os.Remove(f.Name())
}
```

##### 场景2：调用`runtime.Goexit()`退出`goroutine`。
- 错误示例：

```go
func foo() error {
    ...
}

func bar() {
    go func() {
        defer func() {
            fmt.Println("defer!")
            // Goexit 导致无法捕获到`err`错误，error会为nil
            fmt.Println("recovered = ",recover())
        }()

        ...
        err = foo()
        if err != nil {
           runtime.Goexit() // 不符合
        }
    }()

    fmt.Println("bar")
}
```

### 修改建议：
+ 不要在main函数外调用os.Exit();
+ 尽量不要使用runtime.Goexit()。

---

## 14. 规则名称：禁止在闭包中直接使用循环控制变量
**问题级别**：一般

### 正确示例：
##### 场景1：在循环中启动goroutine，在goroutine中直接使用循环控制变量。
- 修复示例：如果要在闭包中使用循环控制变量，可通过入参把循环控制变量显示地引入。

```go
func main() {
    runtime.GOMAXPROCS(runtime.NumCPU())

    var group sync.WaitGroup

    for i := 0; i < 5; i++ {
        group.Add(1)
        go func(j int) {
            defer group.Done()
            // 符合：对循环语句的goroutine需要进行显式地索引变量调用，这样才能得到类似"0 1 2 3 4"期望结果
            fmt.Printf("%-2d", j) // 闭包内部使用局部变量
        }(i) // 符合：把循环控制变量显式地传给goroutine
    }

    group.Wait()
}
```

### 错误示例：
##### 场景1：在循环中启动goroutine，在goroutine中直接使用循环控制变量。
- 错误示例：

```go
func main() {
    runtime.GOMAXPROCS(runtime.NumCPU())
    var group sync.WaitGroup

    for i := 0; i < 5; i++ {
        group.Add(1)
        go func() {
            defer group.Done()
            // 不符合： 输出结果为"5 5 5 5 5"，由于多个goroutine同时使用变量i，goroutine调用的函数闭包对外部变量使用
            // 了引用的方式，而Go运行时调度goroutine的实际时间不确定，这个结果并不是我们所期望的
            fmt.Printf("%-2d", i)
        }()
    }

    group.Wait()
}
```

### 修改建议：
闭包不要直接使用循环控制变量。

---

## 15. 规则名称：避免类型可能不为空的接口变量和nil直接进行比较
**问题级别**：一般

### 正确示例：
##### 场景1：赋值给接口变量的对象的类型不为空
- 修复示例：

```go
type Writer interface {
    write()
}

type S struct {
    x int
}

func (p *S) write()  {

}

func Interface_Get_Nil_good1(m *S) Writer {
    if m == nil { // 符合：判断对象值是否为nil，如果是nil，应直接返回无类型的nil
        return nil // 无类型的nil
    }
    return m
}
```
##### 场景2：类型不为空的变量作为返回值赋值给接口变量
- 修复示例：

```go
type MyError struct {

}

func (this *MyError) Error() string {
    return ""
}

func Return_Typeless_Nil_good1(s *MyError) error {
    if s == nil { // 符合：判断值为nil后，使用无类型的nil作为返回值赋值给接口变量
        return nil // 无类型的nil
    }
    return s
} 
```
##### 场景3：未使用error作为错误返回类型
- 修复示例：

```go
type Errort struct {

}

func (p Errort) Error() string {
    return ""
}

// 符合: 使用error作为函数的错误返回值类型
func Use_Error_As_RetType_good1() error {
    var x Errort
    return x
}
```

### 错误示例：
##### 场景1：赋值给接口变量的对象的类型不为空
- 错误示例：

```go
type Writer interface {
    write()
}

type S struct {
    x int
}

func (p *S) write()  {

}

func Interface_Get_Nil_bad1() Writer {
    var m *S = nil
    // 不符合: 值为空，类型不为空
    return m
}
```
##### 场景2：类型不为空的变量作为返回值赋值给接口变量
- 错误示例：

```go
type MyError struct {

}

func (this *MyError) Error() string {
    return ""
}

func Return_Typeless_Nil_bad1() error {
    var s *MyError = nil
    // 不符合: 返回的类型为*MyError，值为nil，不符合要求
    return s
}
```
##### 场景3：未使用error作为错误返回类型
- 错误示例：

```go
type Errort struct {

}

func (p Errort) Error() string {
    return ""
}

// 不符合: 使用Errort作为错误返回类型，这个类型不正确
func Use_Error_As_RetType_bad1() Errort {
    var x Errort
    return x
}
```

### 修改建议：
类型不为空的变量不能作为返回值赋值给接口变量，可使用无类型的nil。赋值给接口变量的对象指针有效，否则应为无类型的nil。用error接口作为函数的错误返回类型。

---

## 16. 规则名称：重复代码[GO]
**问题级别**：一般

---

## 17. 规则名称：许可证头
**问题级别**：致命

---

## 18. 规则名称：避免使用全局变量
**问题级别**：一般

### 正确示例：
##### 场景1：导出的全局变量可能导致并发问题
- 修复示例1：将其改为不导出的全部变量，并检查该变量是否存在并发读写问题

  ```go
  // 符合：全局变量未导出
  var mySumValue uint = 0
  ```

### 错误示例：
##### 场景1：导出的全局变量可能导致并发问题
- 错误示例：

  ```go
  // 不符合：全局变量禁止导出
  var MySumValue uint = 0
  ```

### 修改建议：
避免使用全局变量，且禁止将全局变量导出。包内使用全局变量，应进行封装。

---

## 19. 规则名称：条件结构中的所有分支不应出现相同的实现
**问题级别**：一般

### 错误示例：
```GO

if b == 0 {  // 不符合
  doOneMoreThing()
} else {
  doOneMoreThing()
}

switch i {  // 不符合
  case 1:
    doSomething()
  case 2:
    doSomething()
  case 3:
    doSomething()
  default:
    doSomething()
}

```

---

## 20. 规则名称：一个包内的如果存在多个init函数，不要有任何的依赖关系
**问题级别**：一般

### 正确示例：
##### 场景1：包内init函数间存在依赖
- 修复示例：
  ```go
  // 符合：修改逻辑，解除包内init函数间存在的依赖

  // file1
  package multiple_independent1

  var kb int

  func init() {
      kb = 1024
  }

  // file2
  package multiple_independent1

  var mb int

  func init() {
      kb := 1024
      mb = 1024 * kb
  }
  ```

### 错误示例：
##### 场景1：包内init函数间存在依赖
- 错误示例：kb在file1和file2之间存在依赖

  ```go

  // flie1
  package multiple_dependent1

  var kb int

  func init() {
      kb = 1024
  }

  // file2
  package multiple_dependent1

  var mb int

  func init() {
      mb = 1024 * kb // 不符合：kb的初始化在另一个init函数中
  }
  ```

### 修改建议：
避免包内init函数之间存在依赖。

---

## 21. 规则名称：删除无效或永不执行的代码
**问题级别**：一般

### 正确示例：
##### 场景1：无效或永不执行的代码未删除
- 修复示例：重新整理逻辑，删除无效或永不执行的代码。

  ```go
  // 符合
  func Foo(condition int) int {
      var i *int
      if condition == 1 {
          i = new(int)
          return 0
      }
      return 0
  }
  ```
##### 场景2：if/else if语句序列中相同的条件会自动导致无效代码
- 修复示例：修改重复的分支条件

  ```go
  // 符合
  func Foo() {
      if param == 1 {
          DoSomething1()
      } else if param == 2 {
          DoSomething2()
      } else if param == 3 { // 符合
          DoSomething3()
      } else {
          ...
      }
  }
  ```
##### 场景3：if条件中的逻辑操作包含常量操作数false
- 修复示例：修改常量操作数的相关逻辑

  ```go
  func Foo() {
      var condition = false
      ... // 包含condition的赋值逻辑
      if a < b && condition {
          doSomething()
      }
  }
  ```
##### 场景4：变量自己赋值给自己
- 修复示例：删除或重新优化代码。

  ```go
  // 符合
  func Foo() {
      var a int32
      ...
  }
  ```

### 错误示例：
##### 场景1：无效或永不执行的代码未删除
- 错误示例：

  ```go
  func Foo(condition int) int {
      var i *int
      if condition == 1 {
          i = new(int)
          return 0
      }
      if i != nil {
          ... // 不符合：变量i为nil，此处代码不可达
      }
      return 0
  }
  ```
##### 场景2：if/else if语句序列中相同的条件会自动导致无效代码
- 错误示例：

  ```go
  func Foo() {
      if param == 1 {
          DoSomething1()
      } else if param == 2 {
          DoSomething2()
      } else if param == 1 { // 不符合： 条件重复，永远不成立
          DoSomething3()
      } else {
          ...
      }
  }
  ```
##### 场景3：if条件中的逻辑操作包含常量操作数false
- 错误示例：

  ```go
  func Foo() {
      const condition = false
      ...
      if a < b && condition { // 不符合：if条件中的逻辑操作包含常量操作数false
          doSomething()
      }
  }
  ```
##### 场景4：变量自己赋值给自己
- 错误示例：

  ```go
  func Foo() {
      var a int32
      a = a // 不符合：需删除无效代码
      ...
  }
  ```

### 修改建议：
审核代码逻辑，对删除无效或永不执行的代码。

---

## 22. 规则名称：避免在循环体中修改循环控制变量
**问题级别**：严重

### 正确示例：
##### 场景1：在循环体中修改循环变量
- 修复示例：重构代码，使之符合不要在循环体中修改循环变量`i`的规则

```go
func main() {
    for i := 0; i < 10; i++ {
        ...
    }
}
```

### 错误示例：
##### 场景1：在循环体中修改循环变量
- 错误示例：存在异向修改，循环将执行多少次取决于代码中if的条件在循环过程中的满足情况，很可能永远无法退出循环

```go
func main() {
    for i := 0; i < 10; i++ {
        ...
        if ... {
            i -= 2  // 不符合
        }
    }
}
```

### 修改建议：
检查循环中异向修改循环参数，防止出现死循环。

---

## 23. 规则名称：文件头注释包含版权说明
**问题级别**：一般

### 正确示例：
##### 场景1：文件头注释没有首先包含“版权说明”
- 修复示例：文件头注释首先包含“版权说明”，然后包含其他可选内容

```go
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * ...
 */

/* 其它说明：... */
```
##### 场景2：版权说明中公司和注释时间顺序错误
- 修复示例：文件头注释首先包含“版权说明”，版权说明的内容及格式必须和规范规定的一致，然后包含其他可选内容

```go
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * ...
 */
```

### 错误示例：
##### 场景1：文件头注释没有首先包含“版权说明”
- 错误示例：

```go
/* 其它说明：... */
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * ...
 */
```
##### 场景2：版权说明中公司和注释时间顺序错误
- 错误示例：

```go
/*
 * 2021-2021. Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
 * ...
 */
```

### 修改建议：
版权说明的内容及格式如下，其内容可根据实际情况做调整，参见下面的**详细说明**。

中文版：
`版权所有 (c) 华为技术有限公司 2012-2018`

英文版：
`Copyright (c) Huawei Technologies Co., Ltd. 2012-2018. All rights reserved.`

**编写文件头注释应注意**

- 文件头注释应该从文件顶头开始（如果包含“关键资产说明”类注释，则文件头注释应紧随其后）
- 文件头注释首先包含“版权说明”，然后包含其他可选内容
- 可选内容应按需添加，避免空有格式，却没有内容
- 保持统一格式，具体格式由项目或更大范围统一制定
- 保持版面工整，换行时应注意对齐

---

## 24. 规则名称：始终使用len(s) == 0检查slice是否为空
**问题级别**：一般

### 正确示例：
##### 场景1：使用`s== nil`来检查slice是否为空，校验返回的结果可能错误
- 修复示例：

```go
func isEmpty(s []string) bool {
    return len(s) == 0  // 符合
}
```

### 错误示例：
##### 场景1：使用`s== nil`来检查slice是否为空，校验返回的结果可能错误
- 错误示例：

```go
func isEmpty(s []string) bool {
    return s == nil     // 不符合
}
```

### 修改建议：
使用len(s) == 0或者len(s) == 0 && s == nil检查slice是否为空（slice中没有元素）。

---

## 25. 规则名称：禁止解引用空的接口和内置引用类型变量
**问题级别**：严重

### 正确示例：
##### 场景1：解引用空接口
- 修复示例：接口使用前判空

```go
type Mystruct struct {

}

func (this *Mystruct) foo() {

}

type MyInterface interface {
     foo()
}

func main() {
    var i MyInterface
    if i != nil {    // 符合：合法性校验
        i.foo()
    }
}
```

##### 场景2：解引用slice
- 修复示例：

```go
func main() {
    var x []int8 = make([]int8, 6) // 符合：初始化
    x[3] = 10
    fmt.Println(x)
}
```

##### 场景3：解引用chan
- 修复示例：

```go
func main() {
    var ch chan int = make(chan int, 6) // 符合：初始化
    go func() {
        for i := 3; i >= 0; i-- {
            ch <- i
            time.Sleep(time.Second)
        }
    }()
    for data := range ch {
        fmt.Println(data)
        if data == 0 {
            break
        }
    }
}
```

##### 场景4：解引用map
- 修复示例：

```go

func main() {
    var my_map map[string]string = make(map[string]string) // 符合：初始化
    my_map["name"] = "Alice"
    fmt.Println(my_map["name"])
}
```

### 错误示例：
##### 场景1：解引用空接口
- 错误示例：

```go
type Mystruct struct {

}

func (this *Mystruct) foo() {

}

type MyInterface interface {
    foo()
}

func main() {
    var i MyInterface
    // 不符合：解引用空接口
    i.foo()
}
```

##### 场景2：解引用slice
- 错误示例：

```go
func main() int {
    ...
    var x []int8
    // 不符合：slice 未初始化被调用
    x[3] = 10
    fmt.Println(x)
}
```

##### 场景3：解引用chan
- 错误示例：

```go
func main() {
    var ch chan int
    // 不符合：chan 未初始化被调用
    go func() {
        for i := 3; i >= 0; i-- {
            ch <- i
            time.Sleep(time.Second)
        }
    }()
    for data := range ch {
        fmt.Println(data)
        if data == 0 {
            break
        }
    }
}
```

##### 场景4：解引用map
- 错误示例：

```go
func main() {
    var my_map map[string]string
    // 不符合：my_map 未初始化被调用
    my_map["name"] = "Alice"
    fmt.Println(my_map["name"])
}
```

### 修改建议：
禁止解引用空的接口和内置引用类型变量。

对slice、chan、map内置引用类型的变量，在访问变量的元素值前，必须使用make显式初始化（对值为nil的slice进行append操作除外，此操作将产生一个新的slice）。
对interface类型变量，在解引用或调用变量方法之前，应判断接口变量不为空。

---

## 26. 规则名称：避免将Go内存指针直接传入C函数使用
**问题级别**：一般

### 正确示例：
##### 场景1：直接将Go指针地址传入C函数。
- 修复示例：在需要将Go的字符串传入C语言时，先通过C.CString将Go语言字符串对应的内存数据复制到新创建的C语言内存空间上，再将C语言的内存传入C函数。请注意，使用该方式可以确保内存使用安全，但会导致效率下降。

```go
package main

/*
#include "stdio.h"
#include "stdlib.h"

void printString(const char* s) {
    printf("%s", s);
}
*/
import "C"
import "unsafe"

func printString(s string) {
    cs := C.CString(s)
    defer C.free(unsafe.Pointer(cs))
    C.printString(cs) // 符合： 将Go内存通过C.String()复制一个C语言的内存再传入C函数
}

func main() {
    s := "hello"
    printString(s)
}
```

### 错误示例：
##### 场景1：直接将Go指针地址传入C函数。
- 错误示例：

```go
package main

/*
#include "stdio.h"

void printString(const char* s) {
    printf("%s", s);
}
*/
import "C"
import "unsafe"

func printString(s string) {
    C.printString((*C.char)(unsafe.Pointer(&s))) // 不符合： 直接将Go指针地址传入C函数
}

func main() {
    s := "hello"
    printString(s)
}
```

### 修改建议：
先通过C.CString将Go语言字符串对应的内存数据复制到新创建的C语言内存空间上。

---

## 27. 规则名称：超大源文件[GO]
**问题级别**：一般

---

## 28. 规则名称：避免直接将C代码嵌入在Go文件中
**问题级别**：一般

### 正确示例：
##### 场景1：C代码和Go代码在同一个文件中。
- 修复示例：建议C代码和Go代码在不同文件中，将对应的C文件编译打包为静态库或动态库文件供Go文件使用。

```go
// foo.h
int sum(int a, int b)
```

### 错误示例：
##### 场景1：C代码和Go代码在同一个文件中。
- 错误示例：

```go
package main

/*
static int sum(int a, int b) {
    return a + b;
}
*/
import "C"

func main() {
    println(C.sum(1, 1)) // 不符合： sum函数源码也在Go文件中
}
```

### 修改建议：
建议采用打包为静态库/动态库的方式。

---

## 29. 规则名称：函数不应为空
**问题级别**：严重

### 正确示例：
```GO

func doNothing() {
  // 因为X和Y，什么都不做。
}

```

### 错误示例：
```GO

func doNothing() { // 不符合
}

```

---

## 30. 规则名称：版权头
**问题级别**：致命

---

## 31. 规则名称：使用[]byte而不是string存储敏感数据，敏感数据使用结束后应立即清0
**问题级别**：一般

### 正确示例：
##### 场景1：使用string保存敏感数据
- 修复示例1：

```go
const pwdMaxLen = 20
func getPassword() (*[pwdMaxLen]byte, error) {
    f, err := os.Open("/secure_area/secue_dat")
    ...
    // 符合：使用[]byte存储敏感数据，使用结束后立即清0
    pwd := new([pwdMaxLen]byte) // 注：如直接定义数组，必须返回数组地址，以保证pwd数组本身会被创建在堆上
    _, err = f.Read(pwd[:])
    return pwd, err
}

func doSomething(){
    password, err := getPassword()
    if err != nil {
        ... // log & return
    }
    verifyPassword(password)
    // 注：如果 getPassword() 到清除位置上可能导致异常，应确保异常分支敏感内存数据也得到清理
    for i := 0; i < len(password); i++{
        password[i] = 0
    }
}

func verifyPassword(pwd []byte) {
    ...
}
```

- 修复示例2：

```go
func main() {
    password := flag.String("pwd", "", "")
    flag.Parse()
    ...
    fmt.Printf("before clear=%s\
", *password)
    // 符合：三方件中使用string存储敏感数据，可在使用结束后通过非类型安全指针（unsafe）方式进行清0
    clearStringMemory(*password)
    fmt.Printf("after clear=%s\
", *password)
}
func clearStringMemory(s string){
    /* 对单个字符表示的字符串，Go的runtime实现的字符串共享字符数据，位于统一的一个staticbytes数组中，
     * 所以禁止清除它，否则程序后面运行时所有从单个字符解析出的字符串为'\\0'值；避免设置口令、密码等敏感信息的长度小于等于1
     */
    if len(s) <= 1 {
        return
    }
    bs := *(*[]byte)(unsafe.Pointer(&s))
    for i := 0; i < len(bs); i++{
        bs[i] = 0
    }
}
```

### 错误示例：
##### 场景1：使用string保存敏感数据
- 错误示例：

```go
func doSomething() {
    pass := getPassword()  // 不符合：变量`pass`属于敏感信息
    verifyPassword(password)
}

func getPassword() string {
    ...
}

func verifyPassword(pwd string) {
    ...
}
```

### 修改建议：
使用[]byte存储敏感信息，使用完成后立即清0。

---

## 32. 规则名称：函数不应有相同的实现
**问题级别**：一般

### 正确示例：
```GO

func fun1() (x, y int) {
  a, b := 1, 2
  b, a = a, b
  return a, b
}

func fun2() (x, y int) {  // 符合
  return fun1()
}

```

### 错误示例：
```GO

func fun1() (x, y int) {
  a, b := 1, 2
  b, a = a, b
  return a, b
}

func fun2() (x, y int) {  // 不符合；fun1和fun2有相同的实现
  a, b := 1, 2
  b, a = a, b
  return a, b
}

```

---

## 33. 规则名称：禁止在日志中保存敏感数据
**问题级别**：严重

### 错误示例：
```go
func foo() {
    sss := "opt/../limit/ssa/focus/etc"
    IP := "255.255.255.3"
    pwd := "p2222s"
    log.Printf("information is %v   %v  %v\n", pwd, IP, sss)// 将敏感信息打印到日志中,所以告警
}
```

错误信息被打印出来
```go
func Sensitive_Leak_bad1() {
    f, err := os.Open("") // os.Open 在配置表的函数
    defer f.Close()
    if err != nil {
        fmt.Println("Widows delete file error.", err.Error()) // 把error类型变量err打印，不合法，告警
    }
}
```

```go
type CreditCard struct {
    Id     int    `json:"card_id"`
    Bank   string `json:"bank_name"`
    Owner  string `json:"owner_id"`
    Pwd string `json:"password"` // 敏感信息
}

func Foo() {
    var card CreditCard
    card.Id = 132004473002056
    card.Bank = "China bank"
    card.Owner = "golang"

    // 此处有HardCode告警
    card.Pwd = "Tell4U"
    serialStr, err := json.Marshal(card) // 【错误】直接将敏感信息序列化
    if err != nil {
        fmt.Println("Serialization error.")
        return
    }
    fmt.Println(string(serialStr), err)
}
```

### 修改建议：
1. 不要将敏感信息保存在日志中或将其输出到控制台及串口;
2. 不要打印敏感信息;
3. 序列化或输出敏感信息应先加密。

---

## 34. 规则名称：确保发生异常时程序尝试恢复到合理状态并记录日志
**问题级别**：严重

### 正确示例：
##### 场景1：在defer-recover恢复机制中忽略发生的异常
- 修复示例：

```go
func process() {
    // 符合：添加defer-recover恢复机制，当存在panic运行时异常，程序将恢复执行
    defer func() {
        if e := recover(); e != nil {
            if re, ok := e.(runtime.Error); ok && re != nil {
                // 尝试为运行时异常做些事情，如：记录日志；做些可靠性的设计，必要时可继续将服务请求交给
                // 另一个就绪的程序做处理 ...
                fmt.Printf("Recover-for-RuntimeError: %v", re)
            } else {
                // 尝试为其他类型的异常做些事情，如继续判断错误的类型，处理特定类型的错误...
                fmt.Printf("Recover-for-No-RuntimeError: %v", e)
            }
        }
    }()
    // ... 特定业务的处理代码
}
```

### 错误示例：
##### 场景1：在defer-recover恢复机制中忽略发生的异常
- 错误示例：

```go
func process() {
    defer func() {
        if e := recover(); e != nil {} // 不符合，忽略发生的异常。
    }()
    ... // 特定业务的处理代码
}
```

### 修改建议：
在发生异常时，在defer语句中恢复到合理状态。

---

## 35. 规则名称：IP地址不能硬编码
**问题级别**：一般

### 正确示例：
```GO

config, err := ReadConfig("properties.ini")

ip := config["ip"]
port := config["ip"]

SocketClient(ip, port)

```

### 错误示例：
```GO

var (
  ip   = "192.168.12.42"
  port = 3333
)

SocketClient(ip, port)

```

### 修改建议：
不要在源代码中硬编码IP地址，要通过环境变量、配置文件或类似方法去进行配置。另外，如果不需要保密，也可以使用域名，因为域名可以快速更改目的地，无需重建软件。

---

## 36. 规则名称：包名应该与所在目录名一致
**问题级别**：一般

### 正确示例：
##### 场景1：目录名与包名不一致
- 修复示例：修改目录名或包名，两者命名保持一致。

  ```go
  // 符合：目录名与包名一致
  md5         // 目录名

  package md5 // 包名
  ```

### 错误示例：
##### 场景1：目录名与包名不一致
- 错误示例：

  ```go
  // 不符合：目录名与包名不一致
  md5_hash    // 目录名

  package md5 // 包名
  ```

### 修改建议：
修改包名使之与所在目录名一致。

---

## 37. 规则名称：避免资源泄漏
**问题级别**：严重

### 正确示例：
##### 场景1：goroutine未及时退出，发生泄露
- 修复示例：

```go
func Goroutine_Exit_good(name string, second int, signal chan int) {
    go func() {
        for {
            select {
            case <-time.Tick(time.Duration(second) * time.Second):
                fmt.Println(name, " is ready!")
            case <-signal:
                fmt.Println(name, " close goroutine.")
                return; // 符合：退出
            }
        }
    }()
}
```

##### 场景2：未释放资源
- 修复示例：

```go
func insert(db *sql.DB) {
    stmt, err := db.Prepare("insert into UserTbl(user,password) values(?,?)")
    if err != nil {
        fmt.Printf("stmt err:%s\n", err)
        return
    }
    defer stmt.Close() // 符合：通过defer语句，确保stmt在函数返回前被关闭
    stmt.Exec("user1", "user1")
}
```

### 错误示例：
##### 场景1：goroutine未及时退出，发生泄露
- 错误示例：
```go
func Goroutine_Exit_bad(name string, second int, signal chan int) {
    go func() {
        for {
            select {
            case <-time.Tick(time.Duration(second) * time.Second):
                fmt.Println(name, " is ready!")
            case <-signal:
                fmt.Println(name, " close goroutine.") // 不符合:未退出
            }
        }
    }()
}
```

##### 场景2：未释放资源
- 错误示例: 
```go
func insert(db *sql.DB) {
    stmt, err := db.Prepare("insert into UserTbl(user,password) values(?,?)") // 不符合：未正确释放资源
    if err != nil {
        fmt.Printf("stmt err:%s\n", err)
        return
    }
    stmt.Exec("user1", "user1")
}
```

### 修改建议：
正确释放使用的资源。

---

## 38. 规则名称：临时文件使用完毕必须及时删除
**问题级别**：严重

### 正确示例：
```go
package main
import (
"fmt"
"io/ioutil"
"os"
)
func closeAndRemove(f *os.File) {
    f.Close()
    os.Remove(f.Name()) // 符合：显式地调用os.Remove()删除临时文件
    }
    func main() {
    f, err := ioutil.TempFile("/opt/pwm", "prefix")
    if err != nil {
        fmt.Println("Create temp file failure.")
        return
    }

    defer closeAndRemove(f) // 符合：调用关闭并删除函数
    _, err = f.Write([]byte("This is a example."))
    if err != nil {
        // error process 省略
    }
}
```

### 错误示例：
```go
func main() {
    f, err := ioutil.TempFile("/opt/pwm", "prefix")
    if err != nil {
        fmt.Println("Create temp file failure.")
        return
    }
    defer f.Close() // 不符合: 临时文件虽然被关闭，但未被删除

    _, err = f.Write([]byte("This is a example."))
    if err != nil {
        // error process 省略
    }
}
```

### 修改建议：
临时文件及时删除。

---

## 39. 规则名称：禁止使用math/rand包提供的函数产生用于安全用途的伪随机数
**问题级别**：严重

### 正确示例：
##### 场景1：math/rand用于盐值的随机数生成。
- 修复示例：在安全用途场景下，应禁止使用`math/rand`包提供的函数产生的随机数，必须使用`crypto/rand`提供的安全的随机数产生方式。

```go
package main

import (
    "crypto/rand" // 安全随机数
    "fmt"
)

func main() {
    saltLength := 32
    salt := make([]byte, saltLength)
    // 符合： 使用rand.Read()生成的盐值是安全的
    if _, err := rand.Read(salt); err != nil {
        fmt.Printf("rand.Read() error : %v", err)
        return
    }
    ...
}
```

### 错误示例：
##### 场景1：math/rand用于盐值的随机数生成。
- 错误示例：

```go
package main

import (
    "math/rand" // 不安全随机数
    "fmt"
)

func main() {
    saltLength := 32
    salt := make([]byte, saltLength)
    // 不符合： 使用math/rand生成的盐值是不安全的
    if _, err := rand.Read(salt); err != nil {
        fmt.Printf("rand.Read() error : %v", err)
        return
    }
    ...
}
```

### 修改建议：
在安全场警下需要调用rand相关函数时，使用crypto/rand包替代math/rand包。

---

## 40. 规则名称：二进制文件
**问题级别**：严重

---

## 41. 规则名称：使用chan时确保对chan的操作有效
**问题级别**：一般

### 正确示例：
##### 场景1：从chan中读取的数据未校验是否有效
- 修复示例1：

```go
func Chan_Close_good1()  {
    var cc = make(chan int)
    for {
        select {
        case c, ok := <-cc:
            if !ok { // 符合：增加对chnnel接收数据的有效性判断，防止死循环，ok == false表示未成功读取到有效数据
                return
            }
            fmt.Println("continue ", ok, c)
        }
        <-cc
    }
}
```
- 修复示例2：

```go
// 省略代码
    for {
        select {
        case _, _ = <-cc:
            // 符合：只要有消息就退出协程，不会有死循环，所以不用判断chan是否成功读取到有效数据
            fmt.Println("channel closed")
            return
        case <-time.After(5 * time.Second):
            fmt.Println("timeout")
        }
    }
    // 省略代码
```

##### 场景2：向已关闭的chan发送数据
- 修复示例：

```go
func Send_Close_Chan_Data_good1()  {
    ch1 := make(chan int) // 符合
    close(ch1)
}
```

### 错误示例：
##### 场景1：从chan中读取的数据未校验是否有效
- 错误示例：

```go
func Chan_Close_bad1(x int)  {
    var cc = make(chan int)
    for {
        select {
        // 不符合: 当channel cc被关闭后如果不做检查则造成死循环
        case c, ok := <-cc:
            fmt.Println("continue ", ok, c)
        }
        <-cc
        if x == 5 {
            return
        }
    }
}
```

##### 场景2：向已关闭的chan发送数据
- 错误示例：

```go
func Send_Close_Chan_Data_bad1()  {
    ch1 := make(chan int)
    close(ch1)
    // 不符合: 向closed的chanl发送数据会产生panic
    ch1 <- 5
}
```

### 修改建议：
1. 通过comma ok的方式检查chan是否成功读取到有效数据的场景。
2. 不向已经close的chan中发送数据。

---

## 42. 规则名称：避免使用短声明定义一个空的slice
**问题级别**：一般

### 正确示例：
##### 场景1：使用短声明定义一个空的slice
- 修复示例：

  ```go
  func Filter(list []int) []int{
      var filtered []int // 符合：如果可以预估结果的大小时，建议使用make指定容量
      for _, v := range list {
          if v > 10 {
              filtered = append(filtered, v)
          }
      }
      return filtered
  }
  ```

### 错误示例：
##### 场景1：使用短声明定义一个空的slice
- 错误示例：

  ```go
  func Filter(list []int) []int{
      filtered := []int{} // 不符合：避免短声明定义一个空的slice
      for _, v := range list {
          if v > 10 {
              filtered = append(filtered, v)
          }
      }
      return filtered
  }
  ```

### 修改建议：
使用var定义空slice。

---

## 43. 规则名称：不应使用八进制值
**问题级别**：致命

### 正确示例：
```GO

func printTen() {
	myNumber := 10
	fmt.Println(myNumber)
}

```

### 错误示例：
```GO

func printTen() {
	myNumber := 010 // 不符合。myNumber将保存8，而不是10。符合预期吗？
	fmt.Println(myNumber)
}

```

---

## 44. 规则名称：表达式的比较，应该遵循左侧倾向于变化、右侧倾向于不变的原则
**问题级别**：一般

### 正确示例：
##### 场景1：常量在比较的左边
- 修复示例：

  ```go
  if p != nil && p.v < Max {   // 符合
      // ...
  }
  ```

### 错误示例：
##### 场景1：常量在比较的左边
- 错误示例：

  ```go
  if nil != p && Max > p.v { // 不符合："nil != p"和"Max > p.v"都将常量放左边
      // ...
  }
  ```

### 修改建议：
尽量遵循变量放在左边，常量放在右边的方式。

---

## 45. 规则名称：冗余或未定义的LICENSE文件
**问题级别**：致命

---

## 46. 规则名称：不应嵌套"switch"语句
**问题级别**：严重

### 正确示例：
```GO

func foo(x,y int) {
	switch x {
	case 0:
		bar(y)
	case 1:
		// ...
	default:
		// ...
	}
}

func bar(y int) {
	switch y {
	// ...
	}
}

```

### 错误示例：
```GO

func foo(x,y int) {
	switch x {
	case 0:
		switch y { // 不符合；嵌套了switch语句
		// ...
		}
	case 1:
		// ...
	default:
		// ...
	}
}

```

---

## 47. 规则名称：设计函数时，优先使用返回值而不是输出参数
**问题级别**：一般

### 正确示例：
##### 场景1：使用输出参数的方式返回数据。
- 修复示例：不要使用输出参数，要使用返回值。

```go
func nextInt(b []byte, i int) (value int, nextIndex int) { // 符合：返回多个需要的返回值
    for ; i < len(b) && !isDigit(b[i]); i++ {
    }
    x := 0
    for ; i < len(b) && isDigit(b[i]); i++ {
        x = x*10 + int(b[i]) - '0'
    }
    return x, i
}

func main() {
    b := []byte{'a', '1', '2', 'c'}
    for x, i := 0, 0; i < len(b); {
        x, i = nextInt(b, i)
        fmt.Println(x)
    }
}
```

### 错误示例：
##### 场景1：使用输出参数的方式返回数据。
- 错误示例：nextInt函数可从字节数组中的指定的位置开始扫描读取下一个整数值，并返回整数值以及下一个待读取的非整数字符的位置。整数值使用输出参数返回。

```go
func isDigit(r byte) bool {
    return '0' <= r && r <= '9'
}

func nextInt(b []byte, i int, x *int) int { // 不符合：通过传入的指针"x"来修改外部实参值
    for ; i < len(b) && !isDigit(b[i]); i++ {
    }
    *x = 0
    for ; i < len(b) && isDigit(b[i]); i++ {
        *x = *x*10 + int(b[i]) - '0'
    }
    return i
}

func main() {
    b := []byte{'a', '1', '2', 'c'}
    for x, i := 0, 0; i < len(b); {
        i = nextInt(b, i, &x)
        fmt.Println(x)
    }
}
```

### 修改建议：
使用返回值替换使用输出参数。

---

## 48. 规则名称：应删除无用的"if(true) {...}"和"if(false){...}"块
**问题级别**：一般

### 正确示例：
```GO

doSomething();
...

```

### 错误示例：
```GO

if true {
	doSomething()
}

if false {
	doSomething()
}

```

---

## 49. 规则名称：布尔值字面量不应冗余
**问题级别**：一般

### 正确示例：
```GO

if boolFunc() {
    // ...
}

flag := x

```

### 错误示例：
```GO

if boolFunc() || false {
    // ...
}

flag := x && true

```

---

## 50. 规则名称：循环必须有显式退出的条件
**问题级别**：一般

### 正确示例：
##### 场景1：for语句不包含显式的退出条件
- 修复示例1：

```go
for {
    ...
    // 符合：应确保someCondition一定能为True，从而退出循环
    if someCondition {
        break
    }
    ...
}
```

- 修复示例2：

```go
// 符合：确保loopCondition在一定次数后失效，从而退出循环
for loopCondition {
    ...
}
```

##### 场景2：循环条件不可达
- 修复示例：

```go
// 符合：每次循环后更新循环变量，退出条件可达
for i := 0; i < 10; i++ {
    fmt.Println("waiting...")
}
```

### 错误示例：
##### 场景1：for语句不包含显式的退出条件
- 错误示例：

```go
for {
    ... // 不符合：此代码模块没有显式的退出条件
}
```

##### 场景2：循环条件不可达
- 错误示例：

```go
// 不符合：循环变量未被修改，退出条件不可达
for i := 0; i < 10; {
    fmt.Println("waiting...")
}
```

### 修改建议：
循环语句中要保证有退出条件。

---

## 51. 规则名称：超大圈复杂度[GO]
**问题级别**：一般

---

## 52. 规则名称：控制流语句"if"、"for"和"switch"不要嵌套太深
**问题级别**：提示

### 错误示例：
默认阈值为3时: 

```GO

if condition1 { // 符合。深度 = 1
	/* ... */
	if condition2 { // 符合。深度 = 2
		/* ... */
		for i := 1; i <= 10; i++ { // 符合。深度 = 3，没有超出范围
			/* ... */
			if condition4 { // 不符合。深度 = 4
				if condition5 { // 深度 = 5，超出范围，但只在深度 = 4时报告缺陷
					/* ... */
				}
				return
			}
		}
	}
}

```

---

## 53. 规则名称：返回已被修改大小的slice参数对象
**问题级别**：一般

### 正确示例：
##### 场景1：参数slice经append后未返回
- 修复示例：函数包含slice参数（而不是slice指针）并对其进行append操作，必须返回修改后的slice。

```go
func AppendXXXList(s []int) []int { // 符合： 返回slice，调用者负责更新slice实参的原版本（其中包含了旧的大小和地址）
    s[0] = 10 // 符合：只是修改内容但没有修改规模
    s2 := append(s, 11);  // 修改规模
    return s2 // 符合: 函数返回s
}
func main() {
    array := []int{1, 2, 3, 4, 5}
    array = AppendXXXList(array) // 符合：因为后面重用了array，需要重新给它赋值
    // 继续使用array，如下的打印，查看被修改的`slice`的内容
    // ...
    fmt.Println(array) // 打印结果为[10 2 3 4 5 11]
}
```

### 错误示例：
##### 场景1：参数slice经append后未返回
- 错误示例：下面代码直接对函数的slice类型入参进行处理，调用者期望入参是不受任何影响的，但实际上经过处理后，slice的值已发生变化。

```go
 func AppendXXXList(s []int) { // 不符合: 修改了slice的大小, 因而必须返回此slice
     s[0] = 10 // 符合: 只是修改内容但没有修改大小
     s = append(s, 11) // 修改大小，如函数不返回s则不符合规则
 }
 func main() {
     array := []int{1, 2, 3, 4, 5}
     AppendXXXList(array)
     // 继续使用array，如下的打印，查看被修改的slice的内容
     // ...
     fmt.Println(array) // 打印结果为[10 2 3 4 5], 不是期望的[10 2 3 4 5 11]
 }
```

### 修改建议：
返回已被修改大小的slice参数对象。

---

## 54. 规则名称：必须处理类型断言的失败
**问题级别**：一般

### 正确示例：
##### 场景1：未处理类型断言的失败
- 修复示例1：使用类型断言判断是否为对应类型的变量：

```go
func Foo(a interface{}) {
    value, ok := a.(string) // 符合：使用类型断言判断是否为对应类型的变量
    if !ok {
        fmt.Println("It's not ok for type string")
        return
    }
    fmt.Println("The value is ", value)
}
```

- 修复示例2：对于模块的接口函数，在使用的外部输入的接口变量时，当它封装的目标类型为引用/指针的类型，在使用引用/指向的值之前，还应判断引用/指针值不为nil：

```go
func Foo(a interface{}) {
    pvalue, ok := a.(*int) // 符合：使用类型断言判断是否为对应类型的变量
    if !ok || pvalue == nil { // 符合：除了判断断言成功，同时，判断指针值不为nil
        fmt.Println("It's not ok for type *int")
        return
    }
    fmt.Println("The value is ", *pvalue)
}
```

- 修复示例3：下述类型断言的例子，如果类型断言失败，则走default分支。

```go
func Bar(t interface{}) {
    // other process
    switch t := t.(type) {
    default: // 符合：必须有`default`分支用于处理类型断言匹配失败的情况
        fmt.Printf("unexpected type %T", t) // %T prints whatever type t has
    case bool:
        fmt.Printf("boolean %t\n", t) // t has type bool
    case int:
        fmt.Printf("integer %d\n", t) // t has type int
    case *bool:
        fmt.Printf("pointer to boolean %t\n", *t) // t has type *bool
    case *int:
        fmt.Printf("pointer to integer %d\n", *t) // t has type *int
    }
    // other process
}
```

### 错误示例：
##### 场景1：未处理类型断言的失败
- 错误示例：如下代码如果a不为string类型，会触发panic.

```go
    fmt.Println("Where are you,Jonny?", a.(string)) // 不符合
```

### 修改建议：
处理类型断言失败的场景。

---

## 55. 规则名称：创建文件时必须显式指定合适的文件访问权限
**问题级别**：一般

### 正确示例：
通过Chmod方法显式更改文件权限
```Golang
import "os"
import "fmt"
func main() {
    const umask = 0o027 // 其它组无访问权限，以确保使用的umask值符合安全要求
    oldUmask := syscall.Umask(umask)
    ...
    f, err := os.Create("/opt/pwm/limit.txt") // 未应用umask前，默认的文件权限为0666；应用
    umask(027)后为文件权限为0640
    if err != nil {
        fmt.Println("Create limit.txt file failure.")
        return
    }
    defer f.Close()
    f.Chmod(0600) // 符合：使用Chmod()函数显式更改文件权限
    f.Write([]byte("This is a example."))
}
```

### 错误示例：
未指定文件权限
```Golang  
import "os"
import "fmt"
func main() {
    // 不符合: 没有显式指定文件权限
    f, err := os.Create("/opt/pwm/unlimit.txt")
    if err != nil {
        fmt.Println("Create unlimit.txt file failure.")
        return
    }
    defer f.Close()
    f.Write([]byte("This is a example."))
}
```

### 修改建议：
创建文件要显式指定文件权限。

---

## 56. 规则名称：禁止使用浮点数作为循环计数器
**问题级别**：一般

### 正确示例：
##### 场景1：直接使用了浮点数作为循环控制变量
- 修复示例：

```go
// 符合：使用了整型作为循环控制变量。
for i := uint32(2000000000) ; i < uint32(2000000001); i++{
    fmt.Println("uint32", i)
}
```

### 错误示例：
##### 场景1：直接使用了浮点数作为循环控制变量
- 错误示例：直接使用了浮点数作为循环控制变量

```go
for i := float32(2000000000) ; i < float32(2000000001); i++{
    fmt.Println("float32", i) // 因为float32(2000000000)和float32(2000000001)都是2e+09，不满足循环条件，所以不会进入循环
}
```

### 修改建议：
使用整型数据作为循环控制变量。

---

## 57. 规则名称：注意init函数内进行初始化的场景限制
**问题级别**：一般

### 正确示例：
##### 场景1：init函数中包含文件打开操作
- 修复示例

  ```go
  // 符合：将文件打开操作删除或移出init函数

  package huaweifunction08

  import "fmt"

  func init(i int) {
      fmt.Println("init in main.go ")
  }
  ```
##### 场景2：函数中调用可能抛出异常的函数
- 修复示例

  ```go
  // 符合：将可能抛出异常的函数调用移出init函数

  package huaweifunction08

  import "fmt"

  func init(i int) {
      fmt.Println("init in main.go ")
  }
  ```

### 错误示例：
##### 场景1：init函数中包含文件打开操作
- 错误示例：

  ```go
  package huaweifunction08

  import (
      "fmt"
      "io/ioutil"
  )

  func init() {
      fileInfos, err := ioutil.ReadDir("./dir/") // 不符合
      if err != nil {
          return
      }
      for _, info := range fileInfos {
          if info.IsDir() {
              fmt.Println(info.Name(), "\\\\")
          }
      }
  }
  ```
##### 场景2：函数中调用可能抛出异常的函数
- 错误示例：
  ```go

  package huaweifunction08

  import "fmt"

  func init(i int) {
      foo() // 不符合：此函数可能抛异常
      fmt.Println("init in main.go ")
  }

  func foo() {
      panic("error")
  }
  ```

### 修改建议：
init函数中应只进行简单的初始化操作。

---

## 58. 规则名称：三方软件版本
**问题级别**：一般

---

## 59. 规则名称：使用统一的命名风格
**问题级别**：提示

### 正确示例：
##### 场景1：未使用驼峰命名风格
- 修复示例1：按照驼峰命名风格修改

  ```go
  // 符合：结构体和成员命名皆遵照驼峰命名风格
  type tWow struct {
      xDamn int
      url   *url.URL
  }
  ```

### 错误示例：
##### 场景1：未使用驼峰命名风格
- 错误示例：

  ```go
  // 不符合：结构体和成员命名未遵照驼峰命名风格
  type t_wow struct {
      x_damn int
      Url    *url.URL
  }
  ```

### 修改建议：
命名应遵循驼峰命名风格。

---

## 60. 规则名称：禁止解引用空指针
**问题级别**：严重

### 正确示例：
##### 场景1：解引用空指针
- 修复示例：

```go
type Student struct {
    Name string
    Age int
}

func (s *Student) Foo(){
    fmt.Printf("Student: name=%s, age=%d \n", s.Name, s.Age)
}

func main() {
    var s *Student
    if len(os.Args) > 1 {
        s = &Student{Name:"test", Age:10}
    }
    if s == nil { // 增加判空处理
        fmt.Println("Failed.")
        return
    }
    s.Foo()  // 符合：使用s变量前，做了空指针判断
}
```

### 错误示例：
##### 场景1：解引用空指针
- 错误示例：

```go
type Student struct {
    Name string
    Age int
}

func (s *Student) Foo(){
    fmt.Printf("Student: name=%s, age=%d \n", s.Name, s.Age)
}

func main() {
    var s *Student // 声明后默认值是nil
    if len(os.Args) > 1 {
        s = &Student{Name:"test", Age:10}
    }
    s.Foo()  // 不符合：s可能为nil
}
```

### 修改建议：
不解引用空指针：指针类型变量使用前进行校验。

---

## 61. 规则名称：确保用key读取到map的元素有效
**问题级别**：严重

### 正确示例：
##### 场景1：未判断用key读取到map的元素是否有效
- 修复示例1：使用多重赋值来分辨key不存在的情况。

```go
// average 计算输入学生的平均年龄
func (c class) average(students []name) (float32, error) {
    cnt := 0
    sum := age(0)
    for _, student := range students {
        if studentAge, ok := c[student]; !ok { // 符合：判断key对应的值存在
            continue
        } else {
            sum += studentAge
            cnt++
        }
    }
    if cnt == 0 {
        return 0, errors.New("input is empty or not found")
    }
    return float32(sum) / float32(cnt), nil
}
```

- 修复示例2：对取得的零值做相应的异常处理。

```go
// average 计算输入学生的平均年龄
func (c class) average(students []name) (float32, error) {
    cnt := 0
    sum := age(0)
    for _, student := range students {
        studentAge := c.getAge(student)
        if studentAge == age(0) { // 符合：对取得的零值做异常处理
            continue
        }
        sum += studentAge
        cnt++
    }
    if cnt == 0 {
        return 0, errors.New("input is empty or not found")
    }
    return float32(sum) / float32(cnt), nil
}
```

### 错误示例：
##### 场景1：未判断用key读取到map的元素是否有效
- 错误示例：

```go
type (
    age uint
    name string
    class map[name]age
)

func (c class) getAge(name name) age {
    return c[name] // 不符合：这里如果查找失败，缺省会返回0，造成外部计算错误
}

// average 计算输入学生的平均年龄
func (c class) average(students []name) (float32, error) {
    if len(students) == 0 {
        return 0, errors.New("input is empty")
    }

    sum := age(0)
    for _, student := range students {
        sum += c.getAge(student)
    }
    return float32(sum) / float32(len(students)), nil
}
```

### 修改建议：
应该通过其他辅助条件去判断map中是否存在特定的值，然后在传回结果。

---

## 62. 规则名称：应删除多余的括号对
**问题级别**：一般

### 正确示例：
```GO

func foo(a bool, y int) int {
  x := (y / 2 + 1)

  if a && (x+y > 0) {
    //...
  }

  return (x + 1)
}

```

### 错误示例：
```GO

func foo(a bool, y int) int {
  x := (y / 2 + 1)   //符合，即使括号被编译器忽略

  if a && ((x+y > 0)) {  // 不符合
    //...
  }

  return ((x + 1))  // 不符合
}

```

---

## 63. 规则名称：禁止使用相对路径导入包
**问题级别**：一般

### 正确示例：
##### 场景1：使用相对路径导入
- 修复示例：

  ```go
  import "github.com/repo/proj/src/net" // 符合：使用完整的路径
  ```

### 错误示例：
##### 场景1：使用相对路径导入
- 错误示例：

  ```go
  import "../net"   // 不符合：使用了相对路径
  ```

### 修改建议：
改为绝对路径导入，建议直接用go mod管理。

---

## 64. 规则名称：导入包时避免使用别名
**问题级别**：一般

### 正确示例：
##### 场景1：使用别名导入包
- 修复示例：

  ```go
  import (
      "fmt"
      "os"
      "runtime/trace" // 符合：避免使用别名，直接导入包

      "aproject.com/oaservice" // 符合：避免使用别名，直接导入包
  )
  ```

### 错误示例：
##### 场景1：使用别名导入包
- 错误示例：

  ```go
  import (
      "fmt"
      "os"
      rtrace "runtime/trace" // 不符合：使用别名

      oas "aproject.com/oaservice" // 不符合：使用别名
  )
  ```

### 修改建议：
直接导入包，不使用别名，除非包名冲突。

---

## 65. 规则名称：禁止使用 . 来简化导入包
**问题级别**：一般

### 正确示例：
##### 场景1：使用“.”简化导入包
- 修复示例：

  ```go
  import "pubcode/api/broker"    // 符合
  ```

### 错误示例：
##### 场景1：使用“.”简化导入包
- 错误示例：

  ```go
  import . "pubcode/api/broker"  // 不符合：使用了 .  简化导入包
  ```

### 修改建议：
不要用"."来导入包，直接导入包即可。

---

## 66. 规则名称：每个包都应该有包注释
**问题级别**：提示

### 正确示例：
##### 场景1：包注释未紧贴package语句且开头不符合要求
- 修复示例：

  ```go
  // 符合：包注释紧贴package语句且开头（"Package"首字母大写）符合要求

  // Package list 实现了一个单向链表.
  package list
  ```

### 错误示例：
##### 场景1：包注释未紧贴package语句且开头不符合要求
- 错误示例：

  ```go
  // 不符合：包注释未紧贴package语句且开头不符合要求

  // package list 实现了一个单向链表.

  package list
  ```

### 修改建议：
增加包注释。

---

## 67. 规则名称：确保用于数组和slice的下标不越界
**问题级别**：严重

### 正确示例：
场景1：循环的上界可导致数组越界访问
- 修复示例：

```go
func Index_OutRange_good1() {
    var buffer [BufSize]byte
    var i = 0
    for i = 0; i < len(buffer) - 1; i++ { // 符合：循环上界可保证合法访问数组
        buffer[i] = buffer[i+1]
    }
}
```

### 错误示例：
场景1：循环的上界可导致数组越界访问
- 错误示例：

```go
func Index_OutRange_bad1() {
    var buffer [BufSize]byte
    var i = 0
    for i = 0; i < len(buffer); i++ {
        /* POTENTIAL FLAW: 循环的上界可导致数组越界访问 */
        buffer[i] = buffer[i+1]
    }
}
```

### 修改建议：
用于数组和slice的下标使用前校验，确保不越界。

---

## 68. 规则名称："switch"语句应包含"default"子句
**问题级别**：提示

### 正确示例：
```GO

switch tag {
case 0, 1, 2, 3:
	foo()
case 4, 5, 6, 7:
	bar()
default:
	qix()
}

```

### 错误示例：
```GO

switch tag { // 不符合 - 缺少default case
case 0, 1, 2, 3:
	foo()
case 4, 5, 6, 7:
	bar()
}

```

---

## 69. 规则名称：函数功能要单一
**问题级别**：一般

### 正确示例：
##### 场景1：函数的代码块嵌套深度超过4层
- 修复示例：

  ```go
  // 符合：使用卫语句
  func testFuncDepthAndParams5() {
      a := 1
      b := 2
      c := 3
      d := 4
      e := 5
      f := 6
      fmt.Println(b)
      if test1 != a {
          return
      }
      if test2 != b {
          return
      }
      if test3 != c {
          return
      }
      if test4 != d {
          return
      }
      if test5 != e {
          return
      }
      if test6 != f {
          return
      }
      fmt.Println(a)
  }
  ```
##### 场景2：函数过长
- 修复示例：可以进行进一步拆分或分层

    ```go
    func testFunc() {
        func1()
        func2()
        ... // 符合：不超过50行的函数内容
    }
    ```
##### 场景3：函数参数超过5个
- 修复示例：对方法进行抽象或重构，或者将参数合并成一个结构体传入。

    ```go
    type data {
        a int,
        b int,
        c int,
        d int,
        e int,
        f int
    }

    func testFunc(d data) { // 符合
        ...
    }
    ```
##### 场景4：函数返回值超过3个
- 修复示例：将返回值合并成一个结构体返回

    ```go
    type data {
        a int,
        b int,
        c int,
        d int,
    }

    func testFunc() data { // 符合
        ...
        return data{
            a: xx,
            b: xx,
            c: xx,
            d: xx,
        }
    }
    ```

### 错误示例：
##### 场景1：函数的代码块嵌套深度超过4层
- 错误示例：
  ```go
  // 不符合
  func testFuncDepthAndParams5() {
      a := 1
      b := 2
      c := 3
      d := 4
      e := 5
      f := 6
      if test1 == a {
          if test2 == b {
              if test3 == c {
                  if test4 == d {
                      if test5 == e {
                          if test6 == f {
                              fmt.Println(a)
                          }
                      }
                  }
              }
          }
      }
      fmt.Println(b)
  }
  ```
##### 场景2：函数过长
- 错误示例：
  ```go
  func testFunc() {
      ... // 不符合：超过50行的函数内容
  }
  ```
##### 场景3：函数参数超过5个
- 错误示例：
  ```go
  func testFunc(a, b, c, d, e, f int) { // 不符合：参数过多
      ...
  }
  ```
##### 场景4：函数返回值超过3个
- 错误示例：
  ```go
  func testFunc() (a, b, c, d int) { // 不符合：返回值过多
      ...
  }
  ```

### 修改建议：
针对描述中的四个维度，合理规划函数。

---

## 70. 规则名称：禁止直接使用外部数据构造格式化字符串
**问题级别**：严重

### 正确示例：
```go
func IncorrectPassword(user string) {
    msgFormat := "%s cannot be authenticated.\n";
    msg := fmt.Sprintf(msgFormat, user)
    fmt.Println(msg); // 使用Println函数代替Printf函数
}
```

### 错误示例：
```go
func IncorrectPassword(user string) {
    msgFormat := "%s cannot be authenticated.\n";
    msg := fmt.Sprintf(msgFormat, user)
    fmt.Printf(msg); // msg中有来自未验证的外部数据，存在格式化漏洞
}
```

### 修改建议：
不直接使用外部数据构造格式化字符串。

---

## 71. 规则名称：冗余代码[GO]
**问题级别**：一般

---

## 72. 规则名称：禁止重复释放资源
**问题级别**：严重

### 正确示例：
```go
func foo(c chan int) {
    defer close(c) // 【修改】使用defer延迟关闭channel
    for {
        err := processBusiness()
        if err != nil {
            c <- 0
            // 【修改】删除重复关闭channel方法
            return
        }
        c <- 1
    }
}
```

### 错误示例：
```go
func foo(c chan int) {
    defer close(c)
    for {
        err := processBusiness()
        if err != nil {
            c <- 0
            close(c) // 【错误】重复释放channel
            return
        }
        c <- 1
    }
}
```

### 修改建议：
不要重复关闭chan。

---

## 73. 规则名称：禁止硬编码访问密钥
**问题级别**：一般

### 错误示例：
硬编码AK Credentials
```Golang
func foo() string {
    c := "1AHFKDSHFJKHDKJSFHQ1"
    return c
}
```

### 修改建议：
不要硬编码AccessKey Credentials。

---

## 74. 规则名称：不能用"=+"代替"+="
**问题级别**：一般

### 正确示例：
```GO

var target, num = -5, 3

target = -num  // 符合，明确表示要指定num的倒数值
target += num

```

### 错误示例：
```GO

var target, num = -5, 3

target =- num  // 不符合，target = -3.真的是这个意思吗？
target =+ num // 不符合，target = 3

```

---

## 75. 规则名称：禁止硬编码API密钥
**问题级别**：一般

### 错误示例：
硬编码密码
```Golang
func foo() string {
    api_key := "fdslagjffdsagas"
    return api_key
}
```

### 修改建议：
使用返回值替换使用输出参数。

---

## 76. 规则名称：避免过多的time.After函数调用导致消耗大量的资源
**问题级别**：一般

### 正确示例：
##### 场景1：在循环语句中使用time.After函数。
- 修复示例：不要在循环语句中使用time.After函数，应该只使用（并复用）一个Timer值。

```go
func longRunning(messages <-chan string) {
    // 符合：为了避免太多的Timer值被创建，应该只使用（并复用）一个Timer值
    timer := time.NewTimer(time.Minute)
    defer timer.Stop()
    for {
        select {
        case <-timer.C:
            return
        case msg, ok := <-messages:
            if !ok {
                return
            }
            fmt.Println(msg)
            if !timer.Stop() {
                <-timer.C // 重要：丢弃当前的过期通知，避免影响下一次循环选择
            }
        }
        // 重要：必须重置以复用
        timer.Reset(time.Minute)
    }
}
```

### 错误示例：
##### 场景1：在循环语句中使用time.After函数。
- 错误示例：

```go
func longRunning(messages <-chan string) {
    for {
        select {
        case <-time.After(time.Minute): // 不符合
            return // 如果某两个连续的消息的间隔大于一分钟，此函数将返回
        case msg := <-messages:
            fmt.Println(msg)
        }
    }
}
```

### 修改建议：
不在循环结构中使用time.After。

---

## 77. 规则名称：所有导出的标识符都应该有注释
**问题级别**：提示

### 正确示例：
##### 场景1：标识符注释只简单重复符号名称
- 修复示例：使用注释说明导出符号（函数）的作用。

  ```go
  // 符合：注释说明了导出符号（函数）的作用

  // LookupHostAddress try enumerate the addresses of all interfaces in the host, get error if enumerate failed.
  func LookupHostAddress(host string) ([]string, error)
  ```

### 错误示例：
##### 场景1：标识符注释只简单重复符号名称
- 错误示例：

  ```go
  // 不符合：该标识符注释只简单重复符号名称

  // LookupHostAddress
  func LookupHostAddress(host string) ([]string, error)
  ```

### 修改建议：
增加包内导出的标识符的注释，以所描述的标识符开头，说明导出标识符的作用，使用时的注意事项等信息。

---

## 78. 规则名称：一个文件只定义一个init函数
**问题级别**：一般

### 正确示例：
##### 场景1：单文件定义多个init函数
- 修复示例：修改逻辑使只存在一个init函数

  ```go
  import "fmt"

  // 符合：一个文件只定义一个init函数
  func init() {
      fmt.Println("init")
  }
  ```

### 错误示例：
##### 场景1：单文件定义多个init函数

### 修改建议：
一个文件只定义一个init函数。

---

## 79. 规则名称：重复文件[GO]
**问题级别**：一般

---

## 80. 规则名称：chan类型作函数参数时限定类型
**问题级别**：一般

### 正确示例：
##### 场景1：chan的参数类型非最小权限
- 修复示例：函数中的chan操作权限与参数权限一致，符合。

  ```go
  func testBidirectionalChanAssignStmt1(ch <-chan int) {  // 符合
      // AssignStmt
      s := <-ch
      fmt.Println(s)
  }
  ```

### 错误示例：
##### 场景1：chan的参数类型非最小权限
- 错误示例：函数中的chan只进行读操作，但chan参数权限是`可读可写`，导致chan参数权限过大

  ```go
  func testBidirectionalChanAssignStmt1(ch chan int) {
      // AssignStmt
      s := <-ch // chan参数只读
      fmt.Println(s)
  }
  ```

### 修改建议：
根据函数中chan的使用权限（`只读`，`只写`，`可读可写`），限定chan参数的类型。

---

## 81. 规则名称：禁止代码中包含公网地址
**问题级别**：一般

### 正确示例：
##### 场景1：硬编码Ipv4公网地址。
- 修复示例：公网地址可以存储在配置文件或数据库中，通过读取配置文件或数据库来获取信息。

```go
// 如果IP地址存储到数据库中，可通过查询操作获取数据
...
rows, err := db.Query("SELECT ip FROM table_name")
if err != nil {
    fmt.Println("Query USER table error.", err)
    return ""
}
defer rows.Close()

// 遍历查询结果
for rows.Next() {
    var ip string
    err := rows.Scan(&ip) // 符合：从数据库来获取IP地址
    if err != nil {
        fmt.Println("Query Scan error.", err)
    }
    ...
}
...
```

##### 场景2：硬编码Ipv6公网地址。
- 修复示例：公网地址可以存储在配置文件或数据库中，通过读取配置文件或数据库来获取信息。

```go
// 如果IP地址存储到数据库中，可通过查询操作获取数据
...
rows, err := db.Query("SELECT ip FROM table_name")
if err != nil {
    fmt.Println("Query USER table error.", err)
    return ""
}
defer rows.Close()

// 遍历查询结果
for rows.Next() {
    var ip string
    err := rows.Scan(&ip) // 符合：从数据库来获取IP地址
    if err != nil {
        fmt.Println("Query Scan error.", err)
    }
    ...
}
...
```

### 错误示例：
##### 场景1：硬编码Ipv4公网地址。
- 错误示例：

```go
ip := "128.0.0.1:100" // 不符合： 硬编码IP地址
```

##### 场景2：硬编码Ipv6公网地址。
- 错误示例：

```go
ip := "fa80:0:0:ec61:c1d1:9827:82be:3" // 不符合： 硬编码IP地址
```

### 修改建议：
禁止硬编码公网IP。

---

## 82. 规则名称：使用工具对代码进行格式化
**问题级别**：一般

### 正确示例：
##### 场景1：对冗余括号进行格式化
- 修复示例1：使用gofmt对代码进行格式化

  ```go
  func AddAtEnd(val int) {
      n := 1
      // 符合：gofmt后冗余括号消除
      if val == 0 {
          val = n
          return
      }
  }
  ```

### 错误示例：
##### 场景1：对冗余括号进行格式化
- 错误示例：

  ```go
  func AddAtEnd(val int) {
      n := 1
      // 不符合：存在冗余括号
      if (val == 0) {
          val = n
          return
      }
  }
  ```

### 修改建议：
直接使用Go语言官方的工具gofmt。

---

## 83. 规则名称："if ... else if"结构应该以"else"语句结束
**问题级别**：提示

### 正确示例：
```GO

if x == 0 {
	doSomething()
} else if x == 1 {
	doSomethingElse()
} else {
	return errors.New("unsupported int")
}

```

### 错误示例：
```GO

if x == 0 {
	doSomething()
} else if x == 1 {
	doSomethingElse()
}

```

---

## 84. 规则名称：超大函数[GO]
**问题级别**：一般

---

## 85. 规则名称：包外可见的函数禁止向外抛出panic
**问题级别**：严重

### 正确示例：
##### 场景1：包外可见函数主动调用panic, 可能导致调用者被意外退出。
- 修复示例1：

```go
package cube

const MaxLen = 10

type cube struct {
    length int
    width  int
    height int
}
func (c *cube) GetVolume() int {
    if c.isLessMaxLen() {
        return c.height * c.width * c.length
    } else {
        c.Reduce(MaxLen)
        return c.height * c.width * c.length
    }
}
// 在当前包中只可使用newCube的构建函数
func NewCube(l, w, h int) *cube {
    if l < w {
        l, w = w, l
    }
    if l < h {
        l, h = h, l
    }
    return &cube{length: l, width: w, height: h}
}
func (c *cube) Increase(num int) {
    if c.length+num < MaxLen {
        c.height += num
        c.width += num
        c.length += num
    }
}
func (c *cube) Reduce(num int) {
    if c.length-num > 0 {
        c.height -= num
        c.width -= num
        c.length -= num
    }
}
func (c *cube) isLessMaxLen() bool {
    return c.length < MaxLen // 符合：使isLessMaxLen()功能单一
}
```

### 错误示例：
##### 场景1：包外可见函数主动调用panic, 可能导致调用者被意外退出。
- 错误示例：

```go
package cube

const MaxLen = 10

type cube struct {
    length int
    width  int
    height int
}

func (c *cube) GetVolume() int {
    // 如果长度小于MAXLEN直接计算体积，否则长度回滚到原来长度再计算体积
    if c.isLessMaxLen() { // 不符合，包外可见函数调用了抛出panic的函数，且未处理
        return c.height * c.width * c.length
    } else {
    c.Reduce(MaxLen)
    return c.height * c.width * c.length
    }
}
func (c *cube) Increase(num int) {
    c.height += num
    c.width += num
    c.length += num
}
func (c *cube) Reduce(num int) {
    c.height -= num
    c.width -= num
    c.length -= num
}
func (c *cube) isLessMaxLen() bool {
    if c.length > MaxLen {
        panic("The length more than max width.")
    }
    return true
}
```

### 修改建议：
包外可见的函数不向外抛panic。

---

## 86. 规则名称：避免使用Go的内置标识符
**问题级别**：一般

### 正确示例：
##### 场景1：下面的代码中使用内置名称string作为变量名称，使得代码难以理解，难以维护。
- 修复示例：修改变量名

  ```go
  func foo() string {
      // 符合：未使用内置标识符
      var str = "hello world"
      return str
  }
  ```

### 错误示例：
##### 场景1：下面的代码中使用内置名称string作为变量名称，使得代码难以理解，难以维护。
- 错误示例：

  ```go
  func foo() string {
      // 不符合：使用了内置标识符string作为变量名
      var string string = "hello world"
      return string
  }
  ```

### 修改建议：
命名时应规避go语言内置标识符。

---

## 87. 规则名称：表达式不应该太复杂
**问题级别**：严重

### 正确示例：
```GO

if ( (myFirstCondition() || mySecondCondition()) && myLastCondition()) { ... }

```

### 错误示例：
默认阈值为3： 

```GO

if (((condition1 && condition2) || (condition3 && condition4)) && condition5) { ... }

```

---

## 88. 规则名称：使用C虚拟包辅助函数C.CString和C.CBytes返回的类型转换参数使用完后需要手动释放内存
**问题级别**：严重

### 正确示例：
##### 场景1：C.CString函数返回值未释放。
- 修复示例：

```go
/*
#include "stdio.h"
#include "stdlib.h"
*/
import "C"
import "unsafe"

func main() {
    cs := C.CString("Hello, World\
")
    defer C.free(unsafe.Pointer(cs)) // 符合：使用C.CString后调用C.free函数释放对应内存
    C.puts(cs)
}
```

##### 场景2：CBytes函数返回值未释放。
- 修复示例：

```go
/*
#include "stdio.h"
#include "stdlib.h"
*/
import "C"
import "unsafe"

func main() {
    bytes := []byte{1, 2}
    cbytes := C.CBytes(bytes)
    defer C.free(unsafe.Pointer(cbytes))// 符合：使用C.CBytes后调用C.free函数释放对应内存
    C.puts((*C.char)(cbytes))
}
```

### 错误示例：
##### 场景1：C.CString函数返回值未释放。
- 错误示例：

```go
// #include "stdio.h"
import "C"

func main() {
    C.puts(C.CString("Hello, World\
")) // 不符合：使用C.CString后未调用C.free函数释放内存
}
```

##### 场景2：CBytes函数返回值未释放。
- 错误示例：

```go
// #include "stdio.h"
import "C"

func main() {
    bytes := []byte{1, 2}
    C.puts((*C.char)C.CBytes(bytes)) // 不符合：使用C.CBytes后未调用C.free函数释放内存
}
```

### 修改建议：
在调用完CString或者Cbytes后使用C.free()等类似的函数进行相关资源的释放。

---

## 89. 规则名称：超大目录[GO]
**问题级别**：一般

---

## 90. 规则名称：确保校验文件路径前对其进行标准化
**问题级别**：严重

### 正确示例：
```go
func Foo() {
    var path string
    flag.StringVar(&path, "filepath", "", "Get user input file path")
    flag.Parse()
    relpath, err := filepath.Abs(path)   //filepath.Abs方法对文件路径进行标准化
    if err != nil {
        return
    }
    file, err := os.Open(path)
    if err != nil {
        fmt.Println("It's error when converted to an absolute path.")
    }
    file.Close()
}
```

### 错误示例：
```go
// 验证文件路径是否在安全目录/opt/pwm下
func Foo() {
    var path string
    flag.StringVar(&path, "filepath", "", "Get user input file path")
    flag.Parse()
    file, err := os.Open(path)
    if err != nil {
        fmt.Println("It's error when converted to an absolute path.")
    }
    file.Close()
}
```

### 修改建议：
文件路径验证前对其进行标准化。

---

## 91. 规则名称：无README
**问题级别**：一般

---

## 92. 规则名称：确保安全地从压缩包中提取文件
**问题级别**：一般

### 正确示例：
##### 场景1：未校验文件数量、文件大小
- 修复示例：

```go
func Unsafe_Unzip_good1(m *BadCase.MyError) {
    r, err := zip.OpenReader("/user/demo/demo.zip")
    if err != nil {
        fmt.Println("ziperr")
    }
    if len(r.File) > TooManyFiles { // 符合：校验文件数量
        fmt.Println("zip file is too big.")
        return
    }
    var total int64
    destPath := "C:/Users/ywx591438/Desktop/"
    if !strings.HasPrefix(destPath, "poi") {
        return
    }
    for _, f := range r.File {
        if f.UncompressedSize64 > TooBigSingleSize || total > TooBigTotalSize {  // 符合：校验文件大小
            return
        }
        rdz, err := f.Open()
        defer rdz.Close()
        if err != nil {
            return
        }
        if f.FileInfo().IsDir() {
            err := os.MkdirAll(destPath, f.Mode())
            if err != nil {
                return
            }
            continue
        }
        destFile, err := os.Create(destPath + "/" + f.Name)
        if err != nil {
            return
        }
        // 符合：io.CopyN 拷贝已知大小，防止数据被篡改
        wt, err := io.CopyN(destFile, rdz, int64(f.UncompressedSize64))
        if err != nil {
            return
        }
        total += wt
    }
}
```
##### 场景二：未校验文件的个数
- 修复示例：
```go
// 将要解压的zipfile文件的压缩包解压到destpath目的路径
// 在调用前，确保destpath是合法的（在允许应用&用户访问的目录范围内，而非外部恶意输入）
func unzipFileSafe(zipfile, destpath string) error {
    // 打开压缩包
    r, err := zip.OpenReader(zipfile)
    if err != nil {
        return errors.New("open the zipfile error")
    }
    defer r.Close()
    // 符合：增加解压文件的数量限制(最大为1024)，防止内存耗尽
    if len(r.File) > TooManyFiles {
        return errors.New("too many file will be unzip")
    }
    destpath = filepath.Clean(destpath) // 符合：将目的路径标准化
    var totalWrote int64
    // 遍历压缩包内的目录和文件
    for _, file := range r.File {
        rc, err := file.Open()
        if err != nil {
            err = errors.New("open file which in zip archive error")
            break
        }
        if file.UncompressedSize64 > TooBigSingleSize || totalWrote > TooBigTotalSize { // 符合：检查单个以及累积解压文件大小
            ... // error process
            // err = ...
            return err
        }
        // 拼装指定的解压路径
        var targetpath = filepath.Join(destpath, file.Name)
        if !strings.HasPrefix(targetpath, destpath) { // 符合：检查输出到目标路径
            ... // error process
            // err = ...
            return err
        }
        // 如果是目录则创建；如果是文件则把文件复制到指定的目的地
        if file.FileInfo().IsDir() {
            e := os.MkdirAll(targetpath, file.Mode())
            if e != nil {
                ... // error process
                // err = ...
                return err
            }
        } else {
            f, e := os.OpenFile(targetpath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, file.Mode())
            if e != nil {
                ... // error process
                err = errors.New("open the dest file error")
                break
            }
            // 从压缩流中把内容复制到目的文件中
            wt, e := io.CopyN(f, rc, int64(file.UncompressedSize64)) // 符合：io.CopyN 拷贝已知大小，防止数据被篡改
            if e != nil {
                ... // error process
                err = errors.New("copy file content error")
                break
            }
            totalWrote += wt
            fmt.Println("wt:", wt)
            f.Close()
        }
        rc.Close()
    }
    return err
}
```

### 错误示例：
##### 场景1：未校验文件数量、文件大小
- 错误示例：

```go
func Unsafe_Unzip_bad1() {
    r, err := zip.OpenReader("/user/demo/demo.zip")
    if err != nil {
        fmt.Println("ziperr")
        return
    }
    if len(r.File) > TooManyFiles {
        fmt.Println("zip file is too big.")
        return
    }
    var total int64
    destPath := "/user/demo/"
    if !strings.HasPrefix(destPath, "poi") {
        return
    }
    for _, f := range r.File {
        rdz, err := f.Open()
        defer rdz.Close()
        if err != nil {
            return
        }
        if f.FileInfo().IsDir() {
            err := os.MkdirAll(destPath, f.Mode())
            if err != nil {
                return
            }
            continue
        }
        destFile, err := os.Create(destPath + "/" + f.Name)
        if err != nil {
            return
        }
        // 不符合: 未校验文件大小
        wt, err := io.Copy(destFile, rdz)

        if err != nil {
            return
        }
        total += wt
    }
}
```

##### 场景二：未校验文件的个数
- 错误示例：
```go
func Unsafe_Unzip_bad2() {
    // 不符合: 未校验文件的个数
    r, err := zip.OpenReader("/user/demo/demo.zip")
    if err != nil {
        fmt.Println("ziperr")
    }
    var total int64
    destPath := "/user/demo/"
    if !strings.HasPrefix(destPath, "poi") {
        return
    }
    for _, f := range r.File {
        if f.UncompressedSize64 > 100000 || total > 300000 {
            return
        }
        rdz, err := f.Open()
        defer rdz.Close()
        if err != nil {
            return
        }
        if f.FileInfo().IsDir() {
            err := os.MkdirAll(destPath, f.Mode())
            if err != nil {
                return
            }
            continue
        }
        destFile, err := os.Create(destPath + "/" + f.Name)
        if err != nil {
            return
        }
        wt, err := io.CopyN(destFile, rdz, int64(f.UncompressedSize64))
        if err != nil {
            return
        }
        total += wt
        }
    }
```

### 修改建议：
校验文件大小，校验文件个数。

---

## 93. 规则名称：函数的返回值应避免使用命名返回值
**问题级别**：一般

### 正确示例：
##### 场景1：使用命名返回值，导致存在冗余的信息
- 修复示例：

  ```go
  func (n *Node) Good() (*Node, error) // 符合
  ```

### 错误示例：
##### 场景1：使用命名返回值，导致存在冗余的信息
- 错误示例：

  ```go
  func (n *Node) Bad() (node *Node, err error) // 不符合：命名的返回值与其类型存在冗余的信息
  ```

### 修改建议：
避免使用命名返回值，除非存在多个返回值类型相同的情况。

---

## 94. 规则名称：代码注释置于对应代码的上方或右边
**问题级别**：提示

### 正确示例：
##### 场景1：代码上方的注释不对齐
- 修复示例1：
  ```go
  // 符合：注释行格式对齐

  // MATCH /Unify the indent of comments/
  // Foo 函数的注释1
  // Foo 函数的注释2
  // Foo 函数的注释3
  func CMT05(a int, aa int, aaa int) {
  ```
##### 场景2：右边注释与代码之间没有空一格
- 修复示例1：
  ```go
  // 符合：右边注释与代码之间空一格

  func CMT05(a int, aa int, aaa int) { // Foo 函数的注释1
  ```

### 错误示例：
##### 场景1：代码上方的注释不对齐
- 错误示例：

  ```go
  // 不符合：相邻注释行格式不对齐

  // MATCH /Unify the indent of comments/
      // Foo 函数的注释1
     // Foo 函数的注释2
  // Foo 函数的注释3
  func CMT05(a int, aa int, aaa int) {
  ```
##### 场景2：右边注释与代码之间没有空一格
- 错误示例：

  ```go
  // 不符合：右边注释与代码之间没有空一格

  func CMT05(a int, aa int, aaa int) {// Foo 函数的注释1
  ```

### 修改建议：
修改注释的位置，是注释置于对应代码的上方或右边。

---

## 95. 规则名称：行宽不超过120个字符
**问题级别**：提示

### 正确示例：
##### 场景1：超长函数签名未断行
- 修复示例1：在适当处断行

  ```go
  // 符合：行宽不超过120个字符
  func testCode121Characters(TAddressStringFor40Digits int, TAddressStringFor40Digits1 int,
      TAddressStrigits2 int, i int) {
      // Code
  }
  ```

### 错误示例：
##### 场景1：超长函数签名未断行
- 错误示例：

  ```go
  // 不符合：行宽超过120个字符
  func testCode121Characters(TAddressStringFor40Digits int, TAddressStringFor40Digits1 int, TAddressStrigits2 int, i int) {
      // Code
  }
  ```

### 修改建议：
进行适当的缩进控制代码行宽。

---

## 96. 规则名称：在CGO中避免使用C代码调用Go函数
**问题级别**：提示

### 正确示例：
##### 场景1：在CGO中C代码调用Go函数。
- 修复示例：建议在CGO中C代码不要调用Go函数。

```go
// my.h
int add(int a, int b);
// my.c
#include "my.h"

int addimpl(int a, int b);

int add(int a, int b) {
    return addimpl(a, b); // 符合：C代码没有调用Go函数
}

int addimpl(int a, int b) {
    return a + b
}
// main.go
package main

import "fmt"

// #cgo CFLAGS: -I.
// #cgo LDFLAGS: -L. -lmy
// #include "my.h"
import "C"

func main() {
    fmt.Printf("%v", C.add(1, 2)) // 符合：仅Go调用了C代码
}
```

### 错误示例：
##### 场景1：在CGO中C代码调用Go函数。
- 错误示例：Go调用了C代码，C代码又反过来调用了Go。

```go
// my.h
int add(int a, int b);
// my.c
#include "my.h"

int addimpl(int a, int b);
int add(int a, int b) {
    return addimpl(a, b); // 不符合：调用Go函数
}
// main.go
package main

import "fmt"

// #cgo CFLAGS: -I.
// #cgo LDFLAGS: -L. -lmy
// #include "my.h"
import "C"

//export addimpl
func addimpl(a, b C.int) C.int {  // 不符合：addimpl函数通过export导出并在C函数add中调用，导致出现嵌套调用
    return a + b
}

func main() {
    fmt.Printf("%v", C.add(1, 2))
}
```

### 修改建议：
不在C代码中调用Go函数。

---

## 97. 规则名称：变量不应该自赋值
**问题级别**：一般

### 正确示例：
```GO

func (user *User) rename(name string) {
  user.name = name
}

```

### 错误示例：
```GO

func (user *User) rename(name string) {
  name = name  // 不符合
}

```

---

## 98. 规则名称：合理使用defer
**问题级别**：一般

### 正确示例：
##### 场景1：简单场景非必要使用defer
- 修复示例：

  ```go
  var lock sync.Mutex
  var counter int

  func increase() {
     lock.Lock()
     counter++
     lock.Unlock()  // 符合：中间没有过多的异常处理，可以不用引入 defer
  }
  ```

### 错误示例：
##### 场景1：简单场景非必要使用defer
- 错误示例：

  ```go
  var lock sync.Mutex
  var counter int

  func increase() {
      lock.Lock()
      defer lock.Unlock()  // 不符合：这里没有必要引入 defer 功能

      counter++
  }
  ```

### 修改建议：
建议按规则描述中的推荐场景使用defer。

---

## 99. 规则名称：不要使用难以理解的字面量
**问题级别**：一般

### 正确示例：
##### 场景1：使用难以理解的字面量
- 修复示例1：用常量定义字面量，并通过常量名说明字面量含义

  ```go
  // 符合：用常量定义字面量，并通过常量名说明字面量含义
  const WeekHours = 7*24
  totalHours.Add(WeekHours)
  ```
- 修复示例2：用注释说明字面量的含义

  ```go
  // 符合：用注释说明字面量的含义
  totalHours.Add(168) // 一周的时间
  ```

### 错误示例：
##### 场景1：使用难以理解的字面量
- 错误示例：

  ```go
  // 不符合：字面量含义难以理解
  totalHours.Add(168)
  ```

### 修改建议：
尽量不要使用字面量，在必须使用时说明字面量的含义。

---

## 100. 规则名称：禁止硬编码密码
**问题级别**：一般

### 错误示例：
通过传入的指针来修改外部实参值
```Golang
func foo() string {
    passwd := "qwe123"
    return passwd
}
```

### 修改建议：
不要硬编码密码。

---

## 101. 规则名称：导出的错误变量采用ErrXxx格式命名
**问题级别**：提示

### 正确示例：
##### 场景1：错误格式的类型命名
- 修复示例：按规范建议修改命名格式

  ```go
  // 符合：错误格式的类型名符合规则描述要求
  type FooError struct {
      ...
  }
  ```
##### 场景2：错误格式的变量命名
- 修复示例：按规范建议修改命名格式

  ```go
  // 符合：错误格式变量名符合规则描述要求
  var ErrSomething = errors.New("something wrong")
  ```

### 错误示例：
##### 场景1：错误格式的类型命名
- 错误示例：

  ```go
  // 不符合：导出的错误类型的命名格式：类型名+Error（即：XxxError）
  type ErrorFoo struct {
      ...
  }
  ```
##### 场景2：错误格式的变量命名
- 错误示例：

  ```go
  // 不符合：导出的错误类型的命名格式：Error+变量名（即：ErrorXxx）
  var ErrorSomethingWrong = errors.New("something wrong")
  ```

### 修改建议：
修改错误变量名并使之符合ErrXxx格式。

---

## 102. 规则名称：凭证不应硬编码
**问题级别**：致命

### 正确示例：
```GO

func connect()  {
  user := getEncryptedUser()
  password:= getEncryptedPass() // 符合

  url := "login=" + user + "&passwd=" + password
}

```

### 错误示例：
```GO

func connect()  {
  user := "root"
  password:= "supersecret" // Sensitive

  url := "login=" + user + "&passwd=" + password
}

```

### 修改建议：
*将凭据存储在未推送到代码仓库的配置文件中。

  *将凭据存储在数据库中。

  *使用云提供商的服务来管理机密。

  *如果密码已通过源代码泄露，更改密码。

---

## 103. 规则名称：初始化结构变量时，尽可能采用复合字面量的方式
**问题级别**：一般

### 正确示例：
##### 场景1：初始化结构体时未采用字面量的方式
- 修复示例：

    ```go
    type Stock struct {
        amount int
        name string
        price big.Rat
        original bool
    }

    func New() *Stock {
        // 符合：使用复合字面量的方式，并指定字段名称
        return &Stock {
            amount: 1,
            name: "foo",
            price: *big.NewRat(200, 100), // price is 2.00
            original: true,
        }
    }
    ```
##### 场景2：初始化结构体时没有指定字段名称
- 修复示例：

    ```go
    type Stock struct {
        amount int
        name string
        price big.Rat
        original bool
    }

    func bar() {
        // 符合：使用复合字面量的方式，并指定字段名称
        s := Stock {
            amount: 1,
            name: "foo",
            price: *big.NewRat(200, 100), // price is 2.00
            original: true,
        }
        ...
    }
    ```

### 错误示例：
##### 场景1：初始化结构体时未采用字面量的方式
- 错误示例：

    ```go
    type Stock struct {
        amount int
        name string
        price big.Rat
        original bool
    }

    func New() *Stock {
        s := new(Stock)
        // 不符合：在初始化结构变量时，使用下面的单独给指定字段赋值比较累赘，此种方式比较适用于给已构造完成的结构体的特定字段赋值
        s.amount = 1
        s.name = "foo"
        s.price = *big.NewRat(200, 100) // price is 2.00
        s.original = true
        return s
    }
    ```
##### 场景2：初始化结构体时没有指定字段名称
- 错误示例：

    ```go
    type Stock struct {
        amount int
        name string
        price big.Rat
        original bool
    }

    func bar() {
        s := Stock { 1, "foo", *big.NewRat(200, 100), true } // 不符合：没有指定字段名称
        ...
    }
    ```

### 修改建议：
优先采用复合字面量的方式，并指定字段名称，以提升代码的可读性。

---

## 104. 规则名称：特殊词汇
**问题级别**：严重

---

## 105. 规则名称：许可证兼容性
**问题级别**：致命

---

## 106. 规则名称：确保正确处理函数的错误返回值
**问题级别**：一般

### 正确示例：
##### 场景1：忽略函数的错误返回值。
- 修复示例：正确处理函数的错误返回值。

```go
if b, err = user.Foo(a); err != nil { // 符合：判断错误值并正确处理这个错误
    fmt.Print("Error")
    return
}

// 如果Foo方法存在约定：当此函数返回的错误值 err != nil时，b一定不会为nil。
// 那么如果前面的语句已经判断了无错误，按约定，可直接调用 b.Bar()；否则应判断返回值b有效
b.Bar()
```

##### 场景2：不接收函数的错误返回值。
- 修复示例：正确处理函数的错误返回值。

```go
func foo() error {
    ...
}

func bar() {
    err = foo()
    if err != nil { // 符合：判断错误值并正确处理这个错误
        fmt.Print("Error")
        return
    }
}
```

### 错误示例：
##### 场景1：忽略函数的错误返回值。
- 错误示例：

```go
b, _ := user.Foo(a) // 不符合: 忽略错误返回，b可能为nil，从而导致代码运行出错
b.Bar()
```
##### 场景2：不接收函数的错误返回值。
- 错误示例：

```go
func foo() error {
    ...
}

func bar() {
    foo() // 不符合: 不接收错误返回值
}
```

### 修改建议：
正确处理函数返回值。

---

## 107. 规则名称：标识符避免使用包名作为前缀
**问题级别**：提示

### 正确示例：
##### 场景1：导出的标识符使用了包名作为前缀
- 修复示例：删除包名前缀

  ```go
  // stream.go
  package stream

  type Buffer struct { // 符合：导出的标识符直接表明含义且不包含包名前缀
      ...
  }
  ```

### 错误示例：
##### 场景1：导出的标识符使用了包名作为前缀
- 错误示例：

  ```go
  // stream.go
  package stream

  type StreamBuffer struct { // 不符合：导出的标识符使用了包名"stream"作为前缀
      ...
  }
  ```

### 修改建议：
修改标识符避免使用包名作为前缀。

---

## 108. 规则名称：包名采用全小写单词，允许包含数字，无下划线
**问题级别**：一般

### 正确示例：
##### 场景1：包名包含下划线
- 修复示例：按规范要求修改包名

  ```go
  // 符合：包名全小写
  package tablecodec
  ```

### 错误示例：
##### 场景1：包名包含下划线
- 错误示例：

  ```go
  // 不符合：包含了下划线
  package table_codec
  ```

### 修改建议：
修改包名使之符合规范要求。

---

## 109. 规则名称：switch语句要有default分支
**问题级别**：一般

### 正确示例：
##### 场景1：switch无default分支
- 修复示例1：

```go
func foo(x int) {
    switch x {
    case 1:
        fmt.Println(1)
    case 6:
        fmt.Println(6)
    default: // 符合：增加default分支
        fmt.Println("default")
    }
}
```

### 错误示例：
##### 场景1：switch无default分支
- 错误示例：

```go
func foo(x int) {
    switch x { // 不符合：没有default分支
    case 1:
        fmt.Println(1)
    case 6:
        fmt.Println(6)
    }
}
```

### 修改建议：
switch语句要有default分支。

---

## 110. 规则名称：二进制运算符两侧不应使用相同表达式
**问题级别**：一般

### 正确示例：
```GO

func main() {
  v1 := (true && false) // 符合
}

```

### 错误示例：
```GO

func main() {
  v1 := (true && false) && (true && false) // 不符合
}

```

---

## 111. 规则名称：目录名采用全小写单词，允许包含数字，允许加中划线‘-’组合方式，但是头尾不应为中划线
**问题级别**：一般

### 正确示例：
##### 场景1：目录名包含大写字母
- 修复示例：按规则建议修改目录名

  ```go
  // 符合：目录名可以包含小写字母和数字
  md5
  ```

### 错误示例：
##### 场景1：目录名包含大写字母
- 错误示例：

  ```go
  // 不符合：目录名不能包含大写字母
  Md5
  ```

### 修改建议：
修改目录名使之符合规范要求。

---

## 112. 规则名称：文件名采用全小写单词，允许包含数字，允许加下划线‘_’组合方式
**问题级别**：一般

### 正确示例：
##### 场景1：文件名使用了驼峰命名
- 修复示例：

  ```go
  // 符合：小写单词加下划线‘_’组合方式
  http_client.go
  ```
##### 场景2：文件名使用了字符‘-’
- 修复示例：

  ```go
  // 符合：小写单词加下划线‘_’组合方式
  http_client.go
  ```

### 错误示例：
##### 场景1：文件名使用了驼峰命名
- 错误示例：

  ```go
  // 不符合：使用了驼峰命名
  HttpClient.go
  ```

### 修改建议：
修改文件名使之符合规范要求。

---

## 113. 规则名称：避免使用unsafe包
**问题级别**：一般

### 正确示例：
##### 场景1：使用unsafe.Pointer进行类型强转
- 修复示例：

```go
func main() {
    var ptra = new(uint8)
    *ptra = 1
    fmt.Printf("ptra=%d\
", *ptra)
    Foo(*ptra)
}

// 符合: 不使用unsafe包，值传递，不用进行类型转换
func Foo(val uint8) {
    ...
    var trb uint32 = val // 值拷贝
    fmt.Printf("trb=%d", trb)
}
```
##### 场景2：使用uintptr和unsafe.Pointer
- 修复示例：

```go
// 符合：不使用unsafe包
func main() {
    inputPara := flag.String("inputPara", "0", "")
    flag.Parse()
    foo(*inputPara)
}
func foo(str string) {
    var tempBuf1 [8]byte
    var isOverWrite = []byte{0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE}
    fmt.Printf("isOverWrite数组元素值全部为：\
%v\
", isOverWrite)
    var strLen = len(str)
    if strLen < 8 {
        for index, ch := range str{
            tempBuf1[index] = byte(ch)
        }
    }
    fmt.Printf("tempBuf1=%p\
",&tempBuf1)
    fmt.Printf("\
执行后，isOverWrite数组元素值实际上为：\
%v", isOverWrite)
}
```

##### 场景3：uintptr值转换为unsafe.Pointer时地址可能已经无效
- 修复示例：

```go
func Safe_Uintptr_good1()   {
    x := new(int)
    p := unsafe.Pointer(x)
    *(*int)(unsafe.Pointer(uintptr(p) + 5)) = 6 // 符合：地址还有效
}
```
##### 场景4：解引用unsafe.Pointer时存在内存越界访问
- 修复示例：

```go
type Mem struct{
    addr unsafe.Pointer // 避免外部直接读写addr缺少地址偏移校验
    length uintptr
}
var intLen = unsafe.Sizeof(int(1))
func (m *Mem)getAddrWithOffset(offset uintptr, dataLen uintptr)(unsafe.Pointer, error){ // 符合：封装安全函数
    addr := offset + dataLen
    if addr > uintptr(m.addr) + m.length{
        return nil, fmt.Errorf("err addr %d", addr)
    }
    return (unsafe.Pointer)(uintptr(m.addr) + addr), nil
}
// SetInt int写接口，保证不越界
func (m *Mem)SetInt(offset uintptr, value int)error{
    addr, err := m.getAddrWithOffset(offset, intLen)
    if err != nil {
        return err
    }
    *(*int)(addr) = value
    return nil
}
```

### 错误示例：
##### 场景1：使用unsafe.Pointer进行类型强转
- 错误示例：

```go
func main() {
    var ptra = new(uint8)
    *ptra = 1
    fmt.Printf("ptra=%d\
", *ptra)
    Prohibited_UintptrPointer_bad1(unsafe.Pointer(ptra))
}

// 不符合: 使用了unsafe包
func Prohibited_UintptrPointer_bad1(buf unsafe.Pointer) {
    ptrb := (*uint32)(buf)
    fmt.Printf("ptrb=%d", *ptrb)
}
```

##### 场景2：使用uintptr和unsafe.Pointer
- 错误示例：

```go
func main() {
    inputPara := flag.String("inputPara", "0", "")
    flag.Parse()
    foo(*inputPara)
}

func foo(str string) {
    var tempBuf1 [8]byte
    var isOverWrite = []byte{0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE, 0xFE}
    fmt.Printf("isOverWrite数组元素值全部为：\
%v\
", isOverWrite)
    var strLen = uint8(len(str)) // 模拟的示例，数据截断使得以下判断失效
    if strLen < 8 {
        tempBuf1Addr := uintptr(unsafe.Pointer(&tempBuf1[0]))
        for index, ch := range str { // 上面数据截断后，可导致越界
            tempPwdPTR := (*byte)(unsafe.Pointer(tempBuf1Addr + uintptr(index)))
            *tempPwdPTR = byte(ch)
        }
    }
    fmt.Printf("tempBuf1=%p\
",&tempBuf1)
    fmt.Printf("\
执行后，isOverWrite数组元素值实际上为：\
%v", isOverWrite)
}
```

##### 场景3：uintptr值转换为unsafe.Pointer时地址可能已经无效
- 错误示例：

```go
func Safe_Uintptr_bad1()   {
    x := new(int)
    p := uintptr(unsafe.Pointer(x))
    // 不符合: 此时地址可能已经无效
    *(*int)(unsafe.Pointer(p))= 5
}
```
##### 场景4：解引用unsafe.Pointer时存在内存越界访问
- 错误示例：

```go
func UnsafePointer_OutRange_bad2() {
    var x int8 = 10
    // 不符合: 指针类型转换将导致数据发生错误
    var y = (*int16)(unsafe.Pointer(uintptr(unsafe.Pointer(&x)) + 5))
    fmt.Println(y)
}

```

### 修改建议：
避免使用uintptr与unsafe.Pointer。在规范例外场景下使用unsafe包，要确保：解引用unsafe.Pointer时不存在内存越界访问。规范例外场景下使用要确保uintptr值转换为unsafe.Pointer时地址有效。

---

## 114. 规则名称：禁止拷贝锁或同步的对象
**问题级别**：提示

### 正确示例：
##### 场景1：递归造成了死锁现象
- 修复示例：

```go
func Lock_Unreleased_good1(s []int) {
    for i := 0; i < len(s); i++ { // 没有递归
        m.Lock()
        ...
        m.Unlock() // 符合，正确释放锁
    }
}
```

##### 场景2：连续请求同一个锁
- 修复示例：

```go
var m sync.Mutex

func Obtain_Same_Lock_Consecutive_good1() {
    s := "the test string"
    m.Lock()
    defer m.Unlock() // 符合：释放锁
    ...
}
```

### 错误示例：
##### 场景1：递归造成了死锁现象
- 错误示例：

```go
var m2 sync.Mutex

func Lock_Unreleased_bad2() {
    // 不符合: 递归造成了死锁现象
    m2.Lock()
    Lock_Unreleased_bad2() // 会发生死锁现象
    m2.Unlock()
}
```

##### 场景2：连续请求同一个锁
- 错误示例：

```go
var m sync.Mutex

func Obtain_Same_Lock_Consecutive_bad1() {
    var x sync.Mutex
    x.Lock()
    // 不符合: x一直未解锁，且连续Lock
    x.Lock()
}
```

### 修改建议：
正确使用锁，不连续请求或释放同一个锁。

---

## 115. 规则名称：导入的包按照稳定度排序
**问题级别**：一般

### 正确示例：
##### 场景1：导入包未按要求分组
- 修复示例：先用gofmt格式化文件，再按建议的顺序调整导入包顺序并分组。

  ```go
  import (  // 符合
      "fmt"
      "hash/adler32"
      "os"

      "github.com/foo/bar"
      "rsc.io/goversion/version"

      "project/foo"
      "project/user"
  )
  ```

### 错误示例：
##### 场景1：导入包未按要求分组
- 错误示例：

  ```go
  import (  // 不符合
      "fmt"
      "hash/adler32"
      "os"
      "github.com/foo/bar"
      "rsc.io/goversion/version"
      "project/foo"
      "project/user"
  )
  ```

### 修改建议：
建议按稳定度分组导入。

---

## 116. 规则名称：不写空有格式的函数和方法头注释
**问题级别**：提示

### 正确示例：
##### 场景1：函数头注释信息冗余
- 修复示例：按需写函数和方法注释，删除不必要的内容

  ```go
  // 符合：函数和方法注释，简要说明其功能即可

  // WriteData 写入字符串
  func WriteData(buf []byte, len uint32) {
  ```

### 错误示例：
##### 场景1：函数头注释信息冗余
- 错误示例：

  ```go
  // 不符合：函数和方法注释，简要说明其功能即可，不需要过于复杂

  /*
   * 函数名：WriteData
   * 功能：写入字符串
   * 参数：
   * 返回值：
   */
  func WriteData(buf []byte, len uint32) {
  ```

### 修改建议：
尽量通过函数名自注释其功能，避免写空有格式的函数头注释和无用、信息冗余的函数头注释。

---

## 117. 规则名称：跟踪"TODO"标签的使用
**问题级别**：提示

### 错误示例：
```GO

func foo() {
  // TODO
}

```

---

