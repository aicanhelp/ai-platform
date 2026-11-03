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
    <title>默认路由检查结果</title>
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
        .summary {
            background: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 4px 4px 0;
        }
        .summary h2 {
            color: #2980b9;
            margin-top: 0;
            font-size: 1.2em;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .summary-item {
            background: white;
            padding: 8px 12px;
            border-radius: 4px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .summary-item span {
            display: block;
            font-weight: bold;
            color: #2c3e50;
            font-size: 1.1em;
        }
        .routes h2 {
            color: #2c3e50;
            margin-bottom: 15px;
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
        .status-pass {
            color: #27ae60;
            font-weight: bold;
        }
        .status-fail {
            color: #e74c3c;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>默认路由检查结果</h1>
        <div class="timestamp">检查时间: {{.Timestamp.Format "2006-01-02 15:04:05"}}</div>

        <div class="summary">
            <h2>检查摘要</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    默认路由数
                    <span>{{.Summary.TotalRoutes}}</span>
                </div>
                <div class="summary-item">
                    检查结果
                    <span class="{{if .Summary.CheckPassed}}status-pass{{else}}status-fail{{end}}">
                        {{if .Summary.CheckPassed}}通过{{else}}失败{{end}}
                    </span>
                </div>
                <div class="summary-item">
                    消息
                    <span>{{.Summary.Message}}</span>
                </div>
            </div>
        </div>

        {{if gt (len .DefaultRoutes) 0}}
        <div class="routes">
            <h2>默认路由详情</h2>
            <table>
                <thead>
                    <tr>
                        <th>目标</th>
                        <th>网关</th>
                        <th>接口</th>
                        <th>Metric</th>
                    </tr>
                </thead>
                <tbody>
                    {{range .DefaultRoutes}}
                    <tr>
                        <td><strong>{{.Destination}}</strong></td>
                        <td>{{.Gateway}}</td>
                        <td>{{.Interface}}</td>
                        <td>{{.Metric}}</td>
                    </tr>
                    {{end}}
                </tbody>
            </table>
        </div>
        {{else}}
        <div class="routes">
            <h2>默认路由详情</h2>
            <p style="color: #e74c3c;">未找到默认路由</p>
        </div>
        {{end}}
    </div>
</body>
</html>