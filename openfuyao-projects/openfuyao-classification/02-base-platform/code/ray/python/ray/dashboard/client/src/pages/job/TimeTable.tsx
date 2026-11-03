import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";
import React from "react";
import { HelpInfo } from "../../components/Tooltip";
import { UnifiedJob } from "../../type/job";

const formatTime = (seconds: number) => {
  if (seconds < 120) {
    return `${seconds.toFixed(2)} s`;
  } else if (seconds < 7200) {
    const minutes = seconds / 60;
    return `${minutes.toFixed(2)} min`;
  } else {
    const hours = seconds / 3600;
    return `${hours.toFixed(2)} h`;
  }
};

export type TimeTableProps = {
  job?: UnifiedJob;
};

export const TimeTable = ({ job }: TimeTableProps) => {
  if (job === null || job === undefined) {
    return null;
  }
  const total = job.resource_usage.total;
  const reprs = job.resource_usage.reprs;
  const columns_1 = [
    { label: "ReprName", rowSpan: 2 },
    { label: "Type", rowSpan: 2 },
    { label: "Actor/Task Number", rowSpan: 2 },
    { label: "Actor Time", rowSpan: 2 },
    { label: "CPU Request Time", rowSpan: 2 },
    { label: "Accelerator Request Time", rowSpan: 2 },
    {
      label: "Profile",
      colSpan: 6,
      helpInfo:
        "Use 'ray job submit --profile -- python your_script.py' to enable the Resources Profiler.",
    },
  ];
  const columns_2 = [
    {
      label: "Subprocess Num",
      helpInfo:
        "Instantaneous maximum number of subprocesses / Instantaneous maximum number of NPU subprocesses (May not be accurate)",
    },
    { label: "Required Resources" },
    { label: "CPU", helpInfo: "Average / Max / Q3" },
    { label: "Memory", helpInfo: "Average / Max / Q3 Unit: MB" },
    { label: "NPU AICore", helpInfo: "Average / Max / Q3" },
    { label: "NPU HBM", helpInfo: "Average / Max / Q3" },
  ];

  return (
    <Table>
      <TableHead>
        <TableRow>
          {columns_1.map(({ label, rowSpan, colSpan, helpInfo }) => (
            <TableCell
              align="center"
              rowSpan={rowSpan}
              colSpan={colSpan}
              key={label}
            >
              <Box display="flex" justifyContent="center" alignItems="center">
                {label}
                {helpInfo && (
                  <HelpInfo sx={{ marginLeft: 1 }}>{helpInfo}</HelpInfo>
                )}
              </Box>
            </TableCell>
          ))}
        </TableRow>
        <TableRow>
          {columns_2.map(({ label, helpInfo }) => (
            <TableCell align="center" key={label}>
              <Box display="flex" justifyContent="center" alignItems="center">
                {label}
                {helpInfo && (
                  <HelpInfo sx={{ marginLeft: 1 }}>{helpInfo}</HelpInfo>
                )}
              </Box>
            </TableCell>
          ))}
        </TableRow>
      </TableHead>
      <TableBody>
        <TableRow>
          <TableCell align="center" component="th" scope="row">
            <Typography
              sx={{ alignItems: "center", fontWeight: 500 }}
              variant="h4"
            >
              Total
            </Typography>
          </TableCell>
          <TableCell align="center"></TableCell>
          <TableCell align="center">{total.instNum}</TableCell>
          <TableCell align="center">{formatTime(total.instTime)}</TableCell>
          <TableCell align="center">{formatTime(total.cpuTime)}</TableCell>
          <TableCell align="center">
            {["GPU", "NPU", "HPU", "TPU"].map((dev) => {
              if (total?.accelerators[dev] === undefined) {
                return null;
              }
              return (
                <p>
                  {dev}: {formatTime(total.accelerators[dev])}
                </p>
              );
            })}
          </TableCell>
          <TableCell align="center"></TableCell>
          <TableCell align="center"></TableCell>
          <TableCell align="center"></TableCell>
          <TableCell align="center"></TableCell>
          <TableCell align="center"></TableCell>
          <TableCell align="center"></TableCell>
        </TableRow>
        {Object.keys(reprs).map((key) => {
          const item = reprs[key];
          const profile = item.profileResult;
          const npu_ai_core = profile?.npuProfile?.aiCore;
          const npu_hbm = profile?.npuProfile?.HBM;
          return (
            <TableRow>
              <TableCell align="center" component="th" scope="row">
                <Typography
                  sx={{ alignItems: "center", fontWeight: 500 }}
                  variant="h4"
                >
                  {item.name}
                </Typography>
              </TableCell>
              <TableCell align="center">{item.type}</TableCell>
              <TableCell align="center">{item.instNum}</TableCell>
              <TableCell align="center">{formatTime(item.instTime)}</TableCell>
              <TableCell align="center">{formatTime(item.cpuTime)}</TableCell>
              <TableCell align="center">
                {["GPU", "NPU", "HPU", "TPU"].map((dev) => {
                  if (item.accelerators[dev] === undefined) {
                    return null;
                  }
                  return (
                    <p>
                      {dev}: {formatTime(item.accelerators[dev])}
                    </p>
                  );
                })}
              </TableCell>
              <TableCell align="center">
                {profile
                  ? profile.childProcessNum + " / " + profile.npuChildProcessNum
                  : ""}
              </TableCell>
              <TableCell align="center">
                {profile
                  ? Object.entries(profile.originalRequiredResources)
                      .map(([key, value]) => `${key}: ${value}`)
                      .join("; ")
                  : ""}
              </TableCell>
              <TableCell align="center">
                {profile?.cpuProfile
                  ? profile.cpuProfile.average_usage +
                    " / " +
                    profile.cpuProfile.max_usage +
                    " / " +
                    profile.cpuProfile.Q3
                  : ""}
              </TableCell>
              <TableCell align="center">
                {profile
                  ? profile.memoryProfile.average_usage +
                    " / " +
                    profile.memoryProfile.max_usage +
                    " / " +
                    profile.memoryProfile.Q3
                  : ""}
              </TableCell>
              <TableCell align="center">
                {npu_ai_core
                  ? npu_ai_core.average_usage +
                    " / " +
                    npu_ai_core.max_usage +
                    " / " +
                    npu_ai_core.Q3
                  : ""}
              </TableCell>
              <TableCell align="center">
                {npu_hbm
                  ? npu_hbm.average_usage +
                    " / " +
                    npu_hbm.max_usage +
                    " / " +
                    npu_hbm.Q3
                  : ""}
              </TableCell>
            </TableRow>
          );
        })}
      </TableBody>
    </Table>
  );
};
