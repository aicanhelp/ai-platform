import { Box, Tooltip, Typography } from "@mui/material";
import React from "react";
import { RightPaddedTypography } from "../../common/CustomTypography";
import PercentageBar from "../../components/PercentageBar";
import PercentageBarLogicalNpu from "../../components/PercentageBarLogicalNpu";
import { NodeDetail } from "../../type/node";

const HBM_COL_WIDTH = 120;

const getMiBRatioNoPercent = (used: number, total: number) =>
  `${used}MiB/${total}MiB`;

type HBMEntryProps = {
  npuName: string;
  slot: number;
  utilization: number;
  total: number;
  percentages?: { percentage: number }[];
  isLogicalNpu?: boolean;
};

const HBMEntry: React.FC<HBMEntryProps> = ({
  npuName,
  slot,
  utilization,
  total,
  percentages = [],
  isLogicalNpu = false,
}) => {
  const ratioStr = getMiBRatioNoPercent(utilization, total);

  return (
    <Box display="flex" flexWrap="nowrap" style={{ minWidth: HBM_COL_WIDTH }}>
      <Tooltip title={npuName}>
        <Box display="flex" flexWrap="nowrap" flexGrow={1}>
          <RightPaddedTypography variant="body1">
            [{slot}]:{" "}
          </RightPaddedTypography>
          {isLogicalNpu ? (
            <PercentageBarLogicalNpu bars={percentages}>
              {ratioStr}
            </PercentageBarLogicalNpu>
          ) : (
            <PercentageBar num={utilization} total={total}>
              {ratioStr}
            </PercentageBar>
          )}
        </Box>
      </Tooltip>
    </Box>
  );
};

export const NodeLogicalNpu = ({ node }: { node: NodeDetail }) => {
  const nodeHBMEntries = (node.logicalNpu ?? []).map((npu, i) => {
    const npuName = npu.npuName;
    const total = npu.total;
    const utilization = Number(
      (
        npu.processes.reduce((sum, process) => sum + process.logicalNpu, 0) *
        total
      ).toFixed(2),
    );
    const slot = npu.index;

    const percentages = npu.processes.map((process) => {
      return { percentage: process.logicalNpu };
    });

    const props = {
      key: npu.index,
      npuName: npuName,
      utilization: utilization,
      total: total,
      slot: slot,
      percentages: percentages,
      isLogicalNpu: true,
    };

    return <HBMEntry {...props} />;
  });
  return (
    <div style={{ minWidth: 60 }}>
      {nodeHBMEntries.length === 0 ? (
        <Typography color="textSecondary" component="span" variant="inherit">
          N/A
        </Typography>
      ) : (
        <div style={{ minWidth: HBM_COL_WIDTH }}>{nodeHBMEntries}</div>
      )}
    </div>
  );
};
