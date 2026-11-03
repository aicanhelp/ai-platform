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
    <title>Dispatch Check Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        h2 { color: #555; margin-top: 30px; }
        h3 { color: #666; margin-top: 20px; }
        h4 { color: #777; margin-top: 15px; margin-left: 10px; }
        table { border-collapse: collapse; width: 100%; margin-top: 10px; }
        th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .success { color: green; font-weight: bold; }
        .failed { color: red; font-weight: bold; }
        .running { color: orange; font-weight: bold; }
        .pass { color: green; }
        .fail { color: red; }
        .info { background-color: #e7f3fe; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .detail-table th { background-color: #666; }
        .sub-table { margin-top: 5px; margin-left: 20px; width: 90%; }
        .sub-table th { background-color: #888; font-size: 0.9em; }
        .sub-table td { font-size: 0.9em; padding: 6px; }
    </style>
</head>
<body>
    <h1>Dispatch Check Report</h1>
    <div class="info">
        <strong>Check Time:</strong> {{.Timestamp.Format "2006-01-02 15:04:05"}}<br>
        <strong>Total Duration:</strong> {{.Duration}}<br>
        <strong>Overall Result:</strong>
        {{if eq .Summary.Result "PASS"}}
        <span class="success">PASS</span>
        {{else}}
        <span class="failed">FAIL</span>
        {{end}}
    </div>

    <h2>Node Results</h2>
    <table>
        <tr>
            <th>IP</th>
            <th>Role</th>
            <th>Status</th>
            <th>Start Time</th>
            <th>End Time</th>
            <th>Error</th>
        </tr>
        {{range .Nodes}}
        <tr>
            <td>{{.IP}}</td>
            <td>{{range .Role}}{{.}} {{end}}</td>
            <td>
                {{if eq .Status "success"}}
                <span class="success">SUCCESS</span>
                {{else if eq .Status "failed"}}
                <span class="failed">FAILED</span>
                {{else}}
                <span class="running">RUNNING</span>
                {{end}}
            </td>
            <td>{{.StartTime.Format "2006-01-02 15:04:05"}}</td>
            <td>{{if .EndTime.IsZero}}-{{else}}{{.EndTime.Format "2006-01-02 15:04:05"}}{{end}}</td>
            <td>{{if .Error}}{{.Error}}{{else}}-{{end}}</td>
        </tr>
        {{end}}
    </table>

    <h2>Summary</h2>
    <table>
        <tr>
            <th>Total Nodes</th>
            <th>Success</th>
            <th>Failed</th>
            <th>Running</th>
        </tr>
        <tr>
            <td>{{.Summary.TotalNodes}}</td>
            <td class="success">{{.Summary.SuccessNodes}}</td>
            <td class="failed">{{.Summary.FailedNodes}}</td>
            <td class="running">{{.Summary.RunningNodes}}</td>
        </tr>
    </table>

    <h2>Detailed Results</h2>
    {{range .Nodes}}
    {{if .Results}}
    <h3>Node: {{.IP}} ({{range .Role}}{{.}} {{end}})</h3>
    <table class="detail-table">
        <tr>
            <th>Check Type</th>
            <th>Status</th>
            <th>Summary</th>
        </tr>
        {{range .Results}}
        <tr>
            <td>{{.CheckType}}</td>
            <td>
                {{if eq .Status "pass"}}
                <span class="success">PASS</span>
                {{else if eq .Status "fail"}}
                <span class="failed">FAIL</span>
                {{else}}
                <span class="running">{{.Status}}</span>
                {{end}}
            </td>
            <td>{{.Detail}}</td>
        </tr>
        {{end}}
    </table>

    {{range .Results}}
    {{if eq .CheckType "port"}}
    {{$portDetails := getPortDetails .}}
    {{if $portDetails}}
    <h4>Port Details</h4>
    <table class="sub-table">
        <tr>
            <th>Port</th>
            <th>Protocol</th>
            <th>Status</th>
            <th>Process</th>
        </tr>
        {{range $portDetails}}
        <tr>
            <td>{{.Port}}</td>
            <td>{{.Protocol}}</td>
            <td>
                {{if .IsUsed}}
                <span class="fail">Occupied</span>
                {{else}}
                <span class="pass">Free</span>
                {{end}}
            </td>
            <td>{{if .Process}}{{.Process}}{{else}}-{{end}}</td>
        </tr>
        {{end}}
    </table>
    {{end}}
    {{end}}

    {{if eq .CheckType "disk"}}
    {{$diskDetails := getDiskDetails .}}
    {{$diskSummary := getDiskSummary .}}
    {{if $diskDetails}}
    <h4>Disk Details</h4>
    <table class="sub-table">
        <tr>
            <th>Path</th>
            <th>Total (GB)</th>
            <th>Free (GB)</th>
            <th>Used %</th>
            <th>Required (GB)</th>
            <th>Status</th>
        </tr>
        {{range $diskDetails}}
        <tr>
            <td>{{.Path}}</td>
            <td>{{printf "%.1f" (divideFloat .Total 1073741824)}}</td>
            <td>{{printf "%.1f" (divideFloat .Free 1073741824)}}</td>
            <td>{{printf "%.1f%%" .UsedPercent}}</td>
            <td>{{printf "%.1f" (divideFloat .MinFree 1073741824)}}</td>
            <td>
                {{if .Error}}
                <span class="running">ERROR</span>
                {{else if .IsSufficient}}
                <span class="pass">PASS</span>
                {{else}}
                <span class="fail">FAIL</span>
                {{end}}
            </td>
        </tr>
        {{end}}
    </table>
    {{end}}
    {{end}}

    {{if eq .CheckType "fileQuery"}}
    {{$fileDetails := getFileQueryDetails .}}
    {{if $fileDetails}}
    <h4>File Query Details</h4>
    <table class="sub-table">
        <tr>
            <th>Path</th>
            <th>Exists</th>
            <th>Type</th>
            <th>Owner</th>
            <th>Group</th>
            <th>Permissions</th>
            <th>Error</th>
        </tr>
        {{range $fileDetails}}
        <tr>
            <td>{{.Path}}</td>
            <td>
                {{if .Exists}}
                <span class="pass">Yes</span>
                {{else}}
                <span class="fail">No</span>
                {{end}}
            </td>
            <td>
                {{if .IsDir}}
                Directory
                {{else}}
                File
                {{end}}
            </td>
            <td>{{if .Owner}}{{.Owner}}{{else}}-{{end}}</td>
            <td>{{if .Group}}{{.Group}}{{else}}-{{end}}</td>
            <td>{{if .Permissions}}{{.Permissions}}{{else}}-{{end}}</td>
            <td>{{if .Error}}{{.Error}}{{else}}-{{end}}</td>
        </tr>
        {{end}}
    </table>
    {{end}}
    {{end}}

    {{if eq .CheckType "programCheck"}}
    {{$programDetails := getProgramCheckDetails .}}
    {{if $programDetails}}
    <h4>Program Check Details</h4>
    <table class="sub-table">
        <tr>
            <th>Program</th>
            <th>Installed</th>
            <th>Version</th>
            <th>Path</th>
        </tr>
        {{range $programDetails}}
        <tr>
            <td>{{.Name}}</td>
            <td>
                {{if .Installed}}
                <span class="pass">Yes</span>
                {{else}}
                <span class="fail">No</span>
                {{end}}
            </td>
            <td>{{if .Version}}{{.Version}}{{else}}-{{end}}</td>
            <td>{{if .Path}}{{.Path}}{{else}}-{{end}}</td>
        </tr>
        {{end}}
    </table>
    {{end}}
    {{end}}

    {{if eq .CheckType "route"}}
    {{$routeDetails := getRouteDetails .}}
    {{if $routeDetails}}
    <h4>Default Route Details</h4>
    <table class="sub-table">
        <tr>
            <th>Destination</th>
            <th>Gateway</th>
            <th>Interface</th>
            <th>Metric</th>
        </tr>
        {{range $routeDetails}}
        <tr>
            <td>{{.Destination}}</td>
            <td>{{.Gateway}}</td>
            <td>{{.Interface}}</td>
            <td>{{.Metric}}</td>
        </tr>
        {{end}}
    </table>
    {{end}}
    {{end}}

    {{if eq .CheckType "iptables"}}
    {{$policy := getIptablesPolicy .}}
    {{if $policy.Chain}}
    <h4>iptables FORWARD Policy</h4>
    <table class="sub-table">
        <tr>
            <th>Chain</th>
            <th>Policy</th>
            <th>Is Accept</th>
        </tr>
        <tr>
            <td>{{$policy.Chain}}</td>
            <td>{{$policy.Policy}}</td>
            <td>
                {{if $policy.IsAccept}}
                <span class="pass">Yes</span>
                {{else}}
                <span class="fail">No</span>
                {{end}}
            </td>
        </tr>
    </table>
    {{end}}
    {{end}}
    {{end}}
    {{end}}
    {{end}}
</body>
</html>
