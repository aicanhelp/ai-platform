import { Box } from "@mui/material";
import React, { PropsWithChildren } from "react";

const PercentageBar = (
  props: PropsWithChildren<{ bars: Array<{ percentage: number }> }>,
) => {
  const { bars } = props;
  const totalPercentage = bars.reduce(
    (total, bar) => total + bar.percentage,
    0,
  );
  const remainingPercentage = (1 - totalPercentage).toFixed(2);

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        minWidth: "150px",
        border: "1px solid #bbf2ff",
      }}
    >
      {bars.map((bar, index) => (
        <Box
          key={index}
          sx={{
            background: "#bbf2ff",
            height: "18px",
            borderLeft: index === 0 ? "1px solid #42ecff" : "none",
            borderRight:
              index === bars.length - 1 ? "1px solid #42ecff" : "none",
            textAlign: "center",
            fontSize: "12px",
            width: `${bar.percentage * 100}%`,
          }}
        >
          {bar.percentage >= 0.1 ? bar.percentage : ""}
        </Box>
      ))}
      {totalPercentage < 1 && (
        <Box
          sx={{
            height: "18px",
            textAlign: "center",
            fontSize: "12px",
            width: `${parseFloat(remainingPercentage) * 100}%`,
          }}
        >
          {parseFloat(remainingPercentage)}
        </Box>
      )}
    </Box>
  );
};

export default PercentageBar;
