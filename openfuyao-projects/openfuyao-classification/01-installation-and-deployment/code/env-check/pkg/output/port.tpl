<!--
 Copyright (c) 2026 Bocloud Technologies Co., Ltd.
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
<html>
<head>
    <meta charset="UTF-8">
    <title>Port Check Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .free { color: green; font-weight: bold; }
        .occupied { color: red; font-weight: bold; }
        .info { background-color: #e7f3fe; padding: 15px; margin: 10px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Port Check Report</h1>
    <div class="info">
        <strong>Check Time:</strong> {{.Timestamp.Format "2006-01-02 15:04:05"}}
    </div>

    <h2>Port Status</h2>
    <table>
        <tr>
            <th>Port</th>
            <th>Protocol</th>
            <th>Status</th>
            <th>Process</th>
        </tr>
        {{range .Ports}}
        <tr>
            <td>{{.Port}}</td>
            <td>{{.Protocol}}</td>
            <td>
                {{if .IsUsed}}
                <span class="occupied">Occupied</span>
                {{else}}
                <span class="free">Free</span>
                {{end}}
            </td>
            <td>{{if .Process}}{{.Process}}{{else}}-{{end}}</td>
        </tr>
        {{end}}
    </table>

    <h2>Summary</h2>
    <table>
        <tr>
            <th>Total Checked</th>
            <th>Occupied</th>
            <th>Free</th>
        </tr>
        <tr>
            <td>{{.Summary.TotalChecked}}</td>
            <td class="occupied">{{.Summary.Used}}</td>
            <td class="free">{{.Summary.Free}}</td>
        </tr>
    </table>
</body>
</html>
