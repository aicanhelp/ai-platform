<!--
 Copyright (c) 2026 Huawei Technologies Co., Ltd.
 openFuyao is licensed under Mulan PSL v2.
 You can use this software according to the terms and conditions of the Mulan PSL v2.
 You may obtain a copy of Mulan PSL v2 at:
          http://license.coscl.org.cn/MulanPSL2
 THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 See the Mulan PSL v2 for more details.
 -->

<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>时钟同步检查</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 25px;
            margin: 20px 0;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 25px;
        }
        .timestamp {
            color: #7f8c8d;
            font-size: 0.9em;
            margin-bottom: 20px;
            font-style: italic;
        }
        .result {
            display: inline-block;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 16px;
            margin: 10px auto;
        }
        .result-pass {
            background-color: #28a745;
            color: white;
        }
        .result-fail {
            background-color: #dc3545;
            color: white;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
            font-weight: 500;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        .sync-status {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        .sync-true {
            background-color: #28a745;
            color: white;
        }
        .sync-false {
            background-color: #dc3545;
            color: white;
        }
        .local-status {
            font-weight: bold;
        }
        .local-true {
            color: #17a2b8;
        }
        .role-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            background-color: #e9ecef;
            color: #495057;
            margin-right: 5px;
            font-size: 14px;
        }
        .error {
            color: #e74c3c;
            word-break: break-all;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>时钟同步检查报告</h1>
        <div class="timestamp">检查时间: {{.Timestamp.Format "2006-01-02 15:04:05"}}</div>
        <div class="result result-{{.Result}}">检测结果: {{if eq .Result "pass"}}通过{{else}}失败{{end}}</div>

        <h2 style="color: #2c3e50; margin-bottom: 15px;">
            检查详细信息
        </h2>
        <table>
            <thead>
                <tr>
                    <th>主机地址</th>
                    <th>角色</th>
                    <th>是否同步</th>
                    <th>时间差(s)</th>
                    <th>是否是执行机</th>
                    <th>报错信息</th>
                </tr>
            </thead>
            <tbody>
                {{range .Clocks}}
                <tr>
                    <td><strong>{{.Host}}</strong></td>
                    <td>
                        {{range .Role}}
                        <span class="role-badge">
                            {{.}}
                        </span>
                        {{end}}
                    </td>
                    <td>
                        <span class="sync-status sync-{{.IsSynced}}">
                            {{if .IsSynced}}
                            是
                            {{else}}
                            否
                            {{end}}
                        </span>
                    </td>
                    <td>{{.TimeDiff}}</td>
                    <td>
                        {{if .IsLocal}}
                        <span class="local-status local-true">是</span>
                        {{else}}
                        否
                        {{end}}
                    </td>
                    <td>
                        {{if .Error}}
                            <span class="error">{{.Error}}</span>
                        {{else}}
                            无
                        {{end}}
                    </td>
                </tr>
                {{end}}
            </tbody>
        </table>
    </div>
</body>
</html>