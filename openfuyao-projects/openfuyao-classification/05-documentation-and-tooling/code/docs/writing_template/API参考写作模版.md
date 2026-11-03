# OpenAPI 编写指南

1. 根对象

```yaml
openapi: 3.0.0  # 必填 - 指定OpenAPI版本
info:            # 必填 - API元数据
servers:         # 可选 - API服务器配置
paths:           # 必填 - API端点定义
components:      # 可选 - 可重用组件
security:        # 可选 - 全局安全方案
tags:            # 可选 - 端点分组标签
externalDocs:    # 可选 - 外部文档链接
```

2. info 对象

```yaml
info:
  title: API标题               # 必填 - API名称
  version: 1.0.0              # 必填 - API版本
  description: API详细描述      
  termsOfService: https://...  # 可选 - 服务条款URL
  contact:                    # 可选 - 联系人信息
    name: 支持团队
    url: https://support.example.com
    email: support@example.com
  license:                    # 可选 - 许可证信息
    name: Apache 2.0
    url: https://www.apache.org/licenses/LICENSE-2.0.html
```

3. 路径操作

```yaml
paths:
  /endpoint:
    get|post|put|patch|delete|options|head|trace:
      summary: 简短描述       # 简短描述
      description: 详细说明   # 说明
      operationId: uniqueId  # 必填，唯一标识
      tags: [tag1]           # 分组
      parameters:            # 可选
        - $ref: '#/components/parameters/Param'
      requestBody:           # 请求体
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/Model'
      responses:             # 必填
        '200':
          $ref: '#/components/responses/Success'
```

4. 参数定义规范

```yaml
paths:parameters:
  - name: paramName
    in: path|query|header|cookie  # 位置
    description: 参数描述         
    required: true|false          # 路径参数必填true
    deprecated: false             # 可选
    schema:                       # 数据类型
      type: string|integer|boolean...
      format: date-time|int32|uuid...
      pattern: 'regex'            # 正则验证
    example: value                # 示例
```

5. 请求体定义

```yaml
requestBody:
  description: 请求体描述       
  required: true|false         # 是否必填
  content:
    application/json:
      schema:
        $ref: '#/components/schemas/Model'
      examples:                # 多个示例
        example1:
          value:
            key: value
```

6. 响应定义

```yaml
responses:
  '200':
    description: 成功响应描述
    headers:                   # 响应头
      Header-Name:
        schema: {type: string}
    content:
      application/json:
        schema:
          $ref: '#/components/schemas/Model'
  '4xx':                       # 错误响应
    $ref: '#/components/responses/Error'
```

7. 组件重用

```yaml
components:
  schemas:        # 数据模型
    User:
      type: object
      properties:
        id: {type: integer}
        name: {type: string}

  parameters:     # 可重用参数
    UserIdParam:
      name: userId
      in: path
      schema: {type: integer}

  responses:      # 可重用响应
    Success:
      description: 成功响应
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/Success'

  securitySchemes: # 安全方案
    BearerAuth:
      type: http
      scheme: bearer
```