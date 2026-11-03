# 错误码管理
restful接口响应信息建议如下：
1. 结构
```
{
    Code: int32,
    Msg: string,
    Data:{}
}
```
2. 正常响应

单对象
```
{
    Code: 0,
    Data:{}
}
```
列表
```
{
    Code: 0,
    Data:[{}]
}
```
3. 错误响应

```
{
    Code: （服务+模块）识别码,
    Data:[{
      Code:服务+模块+错误码,
      Msg:XXX
    }]
}
```
4. 批量操作
```
{
    Code: int32,
    Msg: string,
    Data:{
        success:[{}],
        failed:[{
            Code:服务+模块+错误码,
            Msg:XXX,
            Object:{}
        }]
    }
}
```
## 码表管理
对restful中response中的Code进行分类管理，Code为int32类型，前8位为服务掩码，后面24位各服务自定义。（建议：再8位为服务内模块掩码，后面16位为错误码）

| 服务 | 服务掩码 | 服务掩码十进制 |
| - | - | - |
| console-service | 0000,0001 | 1 |
| monitoring-service | 00000010 | 2 |
| logging-operator | 00000011 | 3 |
| oauth-server | 00000100 | 4 |
| oauth-webhook | 00000101 | 5 |
| oauth-proxy | 00000110 | 6 |
## 样例
以console-service为例：
- 前8位0000,0001为console-service的服务掩码
- 再8位
  - 0000,0001为通用错误；
  - 0000,0010为应用管理;
  - 0000,0011为应用市场
  - 0000,0100为XXXX
- 后面16位由模块自己定义相应错误

