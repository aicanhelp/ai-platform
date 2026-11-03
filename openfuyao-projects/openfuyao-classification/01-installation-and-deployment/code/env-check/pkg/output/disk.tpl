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
    <title>Disk Check Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin-top: 20px; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .pass { color: green; font-weight: bold; }
        .fail { color: red; font-weight: bold; }
        .error { color: orange; font-weight: bold; }
        .info { background-color: #e7f3fe; padding: 15px; margin: 10px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Disk Check Report</h1>
    <div class="info">
        <strong>Check Time:</strong> {{.Timestamp.Format "2006-01-02 15:04:05"}}
    </div>

    <h2>Disk Space Status</h2>
    <table>
        <tr>
            <th>Path</th>
            <th>Filesystem</th>
            <th>Total (GB)</th>
            <th>Free (GB)</th>
            <th>Used (GB)</th>
            <th>Used %</th>
            <th>Status</th>
        </tr>
        {{range .Spaces}}
        <tr>
            <td>{{.Path}}</td>
            <td>{{if .Filesystem}}{{.Filesystem}}{{else}}-{{end}}</td>
            <td>{{printf "%.1f" (divideFloat .Total 1073741824)}}</td>
            <td>{{printf "%.1f" (divideFloat .Free 1073741824)}}</td>
            <td>{{printf "%.1f" (divideFloat .Used 1073741824)}}</td>
            <td>{{printf "%.1f%%" .UsedPercent}}</td>
            <td>
                {{if .Error}}
                <span class="error">ERROR</span>
                {{else if .IsSufficient}}
                <span class="pass">PASS</span>
                {{else}}
                <span class="fail">FAIL</span>
                {{end}}
            </td>
        </tr>
        {{end}}
    </table>

    <h2>Summary</h2>
    <table>
        <tr>
            <th>Total Checked</th>
            <th>Sufficient</th>
            <th>Insufficient</th>
        </tr>
        <tr>
            <td>{{.Summary.TotalChecked}}</td>
            <td class="pass">{{.Summary.SufficientPath}}</td>
            <td class="fail">{{.Summary.InsufficientPath}}</td>
        </tr>
    </table>
</body>
</html>
