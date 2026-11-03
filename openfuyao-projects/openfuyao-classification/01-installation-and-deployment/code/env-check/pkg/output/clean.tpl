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
    <title>清理结果报告</title>
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
            font-size: 1.1em;
        }
        .status-deleted { color: #27ae60; background: #e8f5e9; }
        .status-failed { color: #e74c3c; background: #fdedec; }
        .status-skipped { color: #f39c12; background: #fff8e1; }
        .section {
            margin: 25px 0;
        }
        .section h2 {
            color: #2c3e50;
            padding-left: 10px;
            border-left: 3px solid;
        }
        .section-deleted h2 { border-left-color: #27ae60; }
        .section-failed h2 { border-left-color: #e74c3c; }
        .section-skipped h2 { border-left-color: #f39c12; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
            font-weight: 600;
            color: #2c3e50;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        .permissions-dir { color: #3498db; }
        .permissions-file { color: #7f8c8d; }
        .empty-list {
            padding: 15px;
            background: #f8f9fa;
            border-radius: 4px;
            text-align: center;
            color: #7f8c8d;
            font-style: italic;
        }
        .status-badge {
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>清理结果报告</h1>
        <div class="timestamp">清理时间: {{.Timestamp.Format "2006-01-02 15:04:05"}}</div>

        <div class="summary">
            <h2>汇总信息</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    总检查数
                    <span>{{.Summary.TotalChecked}}</span>
                </div>
                <div class="summary-item status-deleted">
                    已删除
                    <span>{{.Summary.TotalDeleted}}</span>
                </div>
                <div class="summary-item status-failed">
                    失败
                    <span>{{.Summary.TotalFailed}}</span>
                </div>
                <div class="summary-item status-skipped">
                    跳过
                    <span>{{.Summary.TotalSkipped}}</span>
                </div>
            </div>
        </div>

        <div class="section section-deleted">
            <h2>已删除项目 ({{len .Deleted}})</h2>
            {{if .Deleted}}
            <table>
                <thead>
                    <tr>
                        <th>路径</th>
                        <th>类型</th>
                        <th>权限</th>
                        <th>状态</th>
                    </tr>
                </thead>
                <tbody>
                    {{range .Deleted}}
                    <tr>
                        <td>{{.Path}}</td>
                        <td>
                            {{if .IsDir}}
                                <span class="status-badge status-deleted">目录</span>
                            {{else}}
                                <span class="status-badge status-deleted">文件</span>
                            {{end}}
                        </td>
                        <td>
                            {{if .IsDir}}
                                <span class="permissions-dir">{{.Permissions}}</span>
                            {{else}}
                                <span class="permissions-file">{{.Permissions}}</span>
                            {{end}}
                        </td>
                        <td>
                            <span class="status-badge status-deleted">已删除</span>
                        </td>
                    </tr>
                    {{end}}
                </tbody>
            </table>
            {{else}}
            <div class="empty-list">没有已删除的项目</div>
            {{end}}
        </div>

        <div class="section section-failed">
            <h2>失败项目 ({{len .Failed}})</h2>
            {{if .Failed}}
            <table>
                <thead>
                    <tr>
                        <th>路径</th>
                        <th>类型</th>
                        <th>权限</th>
                        <th>错误信息</th>
                    </tr>
                </thead>
                <tbody>
                    {{range .Failed}}
                    <tr>
                        <td>{{.Path}}</td>
                        <td>
                            {{if .IsDir}}
                                <span class="status-badge status-failed">目录</span>
                            {{else}}
                                <span class="status-badge status-failed">文件</span>
                            {{end}}
                        </td>
                        <td>{{.Permissions}}</td>
                        <td>
                            {{if .Error}}
                                <span class="error">{{.Error}}</span>
                            {{else}}
                                无错误信息
                            {{end}}
                        </td>
                    </tr>
                    {{end}}
                </tbody>
            </table>
            {{else}}
            <div class="empty-list">没有失败的项目</div>
            {{end}}
        </div>

        <div class="section section-skipped">
            <h2>跳过项目 ({{len .Skipped}})</h2>
            {{if .Skipped}}
            <table>
                <thead>
                    <tr>
                        <th>路径</th>
                        <th>类型</th>
                        <th>权限</th>
                        <th>跳过原因</th>
                    </tr>
                </thead>
                <tbody>
                    {{range .Skipped}}
                    <tr>
                        <td>{{.Path}}</td>
                        <td>
                            {{if .Exists}}
                                {{if .IsDir}}
                                    <span class="status-badge status-skipped">目录</span>
                                {{else}}
                                    <span class="status-badge status-skipped">文件</span>
                                {{end}}
                            {{else}}
                                <span class="status-badge status-skipped">不存在</span>
                            {{end}}
                        </td>
                        <td>
                            {{if .Permissions}}
                                {{if .IsDir}}
                                    <span class="permissions-dir">{{.Permissions}}</span>
                                {{else}}
                                    <span class="permissions-file">{{.Permissions}}</span>
                                {{end}}
                            {{else}}
                                N/A
                            {{end}}
                        </td>
                        <td>
                            {{if .Error}}
                                <span class="error">{{.Error}}</span>
                            {{else}}
                                无具体原因
                            {{end}}
                        </td>
                    </tr>
                    {{end}}
                </tbody>
            </table>
            {{else}}
            <div class="empty-list">没有跳过的项目</div>
            {{end}}
        </div>
    </div>
</body>
</html>